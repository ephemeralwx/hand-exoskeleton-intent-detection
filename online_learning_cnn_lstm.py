#!/usr/bin/env python3

import argparse
import numpy as np
import scipy.io as sio
from sklearn.metrics import (accuracy_score, f1_score, classification_report,
                             precision_score, recall_score)
import tensorflow as tf
from tensorflow.keras import layers, models, callbacks
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import os
import time

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')

# force cpu — model is too small for gpu to actually help, just adds overhead
os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')


def load_data(path='labeled data 4 subjects.mat'):
    mat = sio.loadmat(path)
    sX, sy = [], []
    for i in range(4):
        d = mat['data'][0, i]
        l = mat['labels'][0, i].flatten().astype(int)
        sX.append(np.array([d[j, 0] for j in range(d.shape[0])]))
        sy.append(l)
    return sX, sy


def load_data_chrono(path='4 subjects ordered chronologically.mat'):
    mat = sio.loadmat(path)
    sX, sy, s_info = [], [], []
    for i in range(4):
        d = mat['data_chrono'][0, i]
        l = mat['labels_chrono'][0, i].flatten().astype(int)
        sX.append(np.array([d[j, 0] for j in range(d.shape[0])]))
        sy.append(l)
        info_i = mat['info'][0, i]
        s_info.append(info_i.astype(int))
    return sX, sy, s_info


def compute_opening_metrics(y_true, y_pred, opening_ids):
    y_true = np.asarray(y_true)
    y_pred = np.asarray(y_pred)
    opening_ids = np.asarray(opening_ids)

    unique_oids = np.unique(opening_ids)

    detected = 0
    total_pos_instances = 0

    for oid in unique_oids:
        mask = opening_ids == oid
        gt = y_true[mask]
        pr = y_pred[mask]

        if np.any(gt == 1):
            total_pos_instances += 1
            if np.any(pr[gt == 1] == 1):
                detected += 1

    fp_windows = int(np.sum((y_pred == 1) & (y_true == 0)))
    neg_windows = int(np.sum(y_true == 0))
    detection_rate = detected / total_pos_instances if total_pos_instances > 0 else 0.0
    window_fpr = fp_windows / neg_windows if neg_windows > 0 else 0.0

    return {
        'detection_rate': detection_rate,
        'detected': detected,
        'total_pos_instances': total_pos_instances,
        'fp_windows': fp_windows,
        'neg_windows': neg_windows,
        'window_fpr': window_fpr,
    }


def threshold_sweep(probs, y_true, opening_ids,
                    thresholds=np.arange(0.30, 0.91, 0.05)):
    probs = np.asarray(probs)
    y_true = np.asarray(y_true)
    opening_ids = np.asarray(opening_ids)

    rows = []
    for th in thresholds:
        preds = (probs > th).astype(int)
        om = compute_opening_metrics(y_true, preds, opening_ids)
        f1 = f1_score(y_true, preds, zero_division=0)
        rows.append({
            'threshold': float(th),
            'det_rate': om['detection_rate'],
            'detected': om['detected'],
            'total_pos': om['total_pos_instances'],
            'fp_windows': om['fp_windows'],
            'window_fpr': om['window_fpr'],
            'f1': float(f1),
        })
    return rows


def build_cnn_lstm(input_shape=(100, 8)):
    # recurrent_dropout only works on cpu, thats fine since we force cpu anyway
    return models.Sequential([
        layers.Conv1D(64, 5, activation='relu', input_shape=input_shape, name='conv1'),
        layers.MaxPooling1D(2, name='pool1'),
        layers.BatchNormalization(name='bn1'),
        layers.Conv1D(128, 3, activation='relu', name='conv2'),
        layers.MaxPooling1D(2, name='pool2'),
        layers.BatchNormalization(name='bn2'),
        layers.LSTM(100, dropout=0.3, recurrent_dropout=0.3, name='lstm'),
        layers.Dense(32, activation='relu', name='dense1'),
        layers.Dropout(0.3, name='drop'),
        layers.Dense(1, activation='sigmoid', name='output'),
    ])


def cw(y):
    n0 = max(np.sum(y == 0), 1)
    n1 = max(np.sum(y == 1), 1)
    t = len(y)
    return {0: t / (2 * n0), 1: t / (2 * n1)}


def fmt_time(seconds):
    if seconds < 60:
        return f'{seconds:.1f}s'
    m, s = divmod(int(seconds), 60)
    return f'{m}m {s:02d}s'


def train_base(X_tr, y_tr, X_val, y_val, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    m = build_cnn_lstm()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(X_tr, y_tr, epochs=50, batch_size=256,
          validation_data=(X_val, y_val),
          class_weight=cw(y_tr),
          callbacks=[callbacks.EarlyStopping('val_loss', patience=10,
                                            restore_best_weights=True)],
          verbose=2)
    return m


class OnlineLearner:

    def __init__(self, model, *, update_every=50, warmup=50,
                 ft_epochs=3, ft_bs=64, ft_lr=5e-4,
                 freeze_early=True, adapt_thresh=True, buf_cap=300,
                 chronological=False):
        self.model = model
        self.update_every = update_every
        self.warmup = warmup
        self.ft_epochs = ft_epochs
        self.ft_bs = ft_bs
        self.freeze_early = freeze_early
        self.adapt_thresh = adapt_thresh
        self.buf_cap = buf_cap
        self.chronological = chronological
        self.threshold = 0.5

        self.buf_X, self.buf_y = [], []
        self.preds, self.trues, self.probs_list = [], [], []
        self.oid_list = []
        self.update_pts = []

        # freeze early conv layers — they learn general features, dont need to adapt those online
        if freeze_early:
            for layer in model.layers:
                if layer.name in ('conv1', 'bn1'):
                    layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(ft_lr),
                      loss='binary_crossentropy', metrics=['accuracy'])

    def run(self, X, y, seed=0, opening_ids=None):
        n = len(X)

        if self.chronological:
            Xs, ys = X, y
            oids = opening_ids
        else:
            rng = np.random.RandomState(seed)
            order = rng.permutation(n)
            Xs, ys = X[order], y[order]
            oids = opening_ids[order] if opening_ids is not None else None

        sched = set(range(self.warmup, n + 1, self.update_every))
        bounds = sorted(set([0] + list(sched) + [n]))

        for b in range(len(bounds) - 1):
            s, e = bounds[b], bounds[b + 1]
            if s >= e:
                continue

            bX, by = Xs[s:e], ys[s:e]

            bp = self.model(tf.constant(bX, dtype=tf.float32), training=False).numpy().flatten()
            bpred = (bp > self.threshold).astype(int)

            self.preds.extend(bpred.tolist())
            self.trues.extend(by.tolist())
            self.probs_list.extend(bp.tolist())
            if oids is not None:
                self.oid_list.extend(oids[s:e].tolist())
            self.buf_X.extend(bX)
            self.buf_y.extend(by.tolist())

            if e in sched:
                self._finetune()
                if self.adapt_thresh:
                    self._opt_thresh()
                self.update_pts.append(e)

        return self._results()

    def _finetune(self):
        X = np.array(self.buf_X)
        y = np.array(self.buf_y)

        if len(X) > self.buf_cap:
            # 60/40 recent/old split — bias toward recent data but keep some old to avoid catastrophic forgetting
            recent_n = int(self.buf_cap * 0.6)
            old_n = self.buf_cap - recent_n
            old_pool = len(X) - recent_n
            old_idx = np.random.choice(old_pool, min(old_n, old_pool), replace=False)
            recent_idx = np.arange(len(X) - recent_n, len(X))
            idx = np.concatenate([old_idx, recent_idx])
            X, y = X[idx], y[idx]

        weights = cw(y)
        sample_weights = np.where(y == 1, weights[1], weights[0]).astype(np.float32)
        X_t = tf.constant(X, dtype=tf.float32)
        y_t = tf.constant(y, dtype=tf.float32)
        sw_t = tf.constant(sample_weights, dtype=tf.float32)
        for _ in range(self.ft_epochs):
            self.model.train_on_batch(X_t, y_t, sample_weight=sw_t)

    def _opt_thresh(self):
        X = np.array(self.buf_X[-self.buf_cap:])
        y = np.array(self.buf_y[-self.buf_cap:])
        ps = self.model(tf.constant(X, dtype=tf.float32), training=False).numpy().flatten()

        # below 0.15 is basically always false positive city, above 0.85 kills recall
        thresholds = np.arange(0.15, 0.85, 0.02)
        preds_all = (ps[np.newaxis, :] > thresholds[:, np.newaxis]).astype(int)
        y_broad = y[np.newaxis, :]

        tp = np.sum((preds_all == 1) & (y_broad == 1), axis=1).astype(float)
        fp = np.sum((preds_all == 1) & (y_broad == 0), axis=1).astype(float)
        fn = np.sum((preds_all == 0) & (y_broad == 1), axis=1).astype(float)
        precision = np.where(tp + fp > 0, tp / (tp + fp), 0.0)
        recall = np.where(tp + fn > 0, tp / (tp + fn), 0.0)
        f1s = np.where(precision + recall > 0,
                       2 * precision * recall / (precision + recall), 0.0)

        best_idx = np.argmax(f1s)
        self.threshold = thresholds[best_idx]

    def _results(self):
        p, t = np.array(self.preds), np.array(self.trues)
        n = len(p)
        q = n // 4

        def m(t_, p_):
            return {'acc': float(accuracy_score(t_, p_)),
                    'f1': float(f1_score(t_, p_, zero_division=0)),
                    'prec': float(precision_score(t_, p_, zero_division=0)),
                    'rec': float(recall_score(t_, p_, zero_division=0))}

        # 100 window sliding avg — smaller is too noisy, larger hides when adaptation actually kicks in
        win = 100
        run_acc, run_f1 = [], []
        for i in range(n):
            s = max(0, i + 1 - win)
            run_acc.append(accuracy_score(t[s:i+1], p[s:i+1]))
            run_f1.append(f1_score(t[s:i+1], p[s:i+1], zero_division=0))

        result = {
            'overall': m(t, p),
            'q1': m(t[:q], p[:q]),
            'q2': m(t[q:2*q], p[q:2*q]),
            'q3': m(t[2*q:3*q], p[2*q:3*q]),
            'q4': m(t[3*q:], p[3*q:]),
            'first_half': m(t[:n//2], p[:n//2]),
            'second_half': m(t[n//2:], p[n//2:]),
            'run_acc': run_acc, 'run_f1': run_f1,
            'updates': self.update_pts,
            'threshold': self.threshold, 'n': n,
            'probs': list(self.probs_list),
            'preds_list': p.tolist(),
            'trues': t.tolist(),
            'oids': list(self.oid_list) if self.oid_list else [],
        }

        if self.oid_list:
            oids = np.array(self.oid_list)
            result['opening'] = compute_opening_metrics(t, p, oids)

            for label, sl in [('q1', slice(0, q)), ('q2', slice(q, 2*q)),
                               ('q3', slice(2*q, 3*q)), ('q4', slice(3*q, None))]:
                result[f'opening_{label}'] = compute_opening_metrics(
                    t[sl], p[sl], oids[sl])

        return result


def run_experiment(subjects_X, subjects_y, config, n_seeds=3,
                   subjects_info=None, chronological=False):
    results = {}
    subject_labels = ['healthy', 'healthy', 'healthy', 'stroke']
    total_subjects = 4
    # chrono is deterministic so one seed is enough
    effective_seeds = 1 if chronological else n_seeds
    total_steps = total_subjects * (1 + effective_seeds)
    current_step = 0
    exp_start = time.time()
    step_times = []

    for ts in range(4):
        name = f'Subject {ts+1}'
        tag = subject_labels[ts]
        print(f'\n{"="*60}')
        print(f'  {name} ({tag})')
        print(f'{"="*60}')

        X_tr = np.concatenate([subjects_X[i] for i in range(4) if i != ts])
        y_tr = np.concatenate([subjects_y[i] for i in range(4) if i != ts])
        X_te, y_te = subjects_X[ts], subjects_y[ts]
        oi = subjects_info[ts][:, 2] if subjects_info is not None else None

        print(f'  train: {len(y_tr)} windows ({np.sum(y_tr==1)} pos, {np.sum(y_tr==0)} neg)')
        print(f'  test:  {len(y_te)} windows ({np.sum(y_te==1)} pos, {np.sum(y_te==0)} neg)')

        step_start = time.time()
        print('  training base model...', end=' ', flush=True)
        base = train_base(X_tr, y_tr, X_te, y_te, seed=42)
        bp = base(tf.constant(X_te, dtype=tf.float32), training=False).numpy().flatten()
        bpred = (bp > 0.5).astype(int)
        b_acc = accuracy_score(y_te, bpred)
        b_f1 = f1_score(y_te, bpred, zero_division=0)
        b_rec = recall_score(y_te, bpred, zero_division=0)
        b_prec = precision_score(y_te, bpred, zero_division=0)
        step_elapsed = time.time() - step_start
        step_times.append(step_elapsed)
        current_step += 1
        elapsed = time.time() - exp_start
        avg_step = elapsed / current_step
        remaining = avg_step * (total_steps - current_step)
        print(f'done ({fmt_time(step_elapsed)}).  baseline: acc={b_acc:.4f}, F1={b_f1:.4f}, '
              f'prec={b_prec:.4f}, rec={b_rec:.4f}')

        b_opening = None
        if oi is not None:
            b_opening = compute_opening_metrics(y_te, bpred, oi)
            print(f'  baseline opening detection: {b_opening["detected"]}/{b_opening["total_pos_instances"]} '
                  f'({b_opening["detection_rate"]:.1%}), '
                  f'FP windows: {b_opening["fp_windows"]}/{b_opening["neg_windows"]} '
                  f'({b_opening["window_fpr"]:.1%})')

        print(f'  [time] elapsed: {fmt_time(elapsed)} | '
              f'est. remaining: {fmt_time(remaining)} | '
              f'step {current_step}/{total_steps}')

        seeds = []
        for seed in range(effective_seeds):
            step_start = time.time()
            fresh = build_cnn_lstm()
            fresh.build((None, 100, 8))
            fresh.set_weights(base.get_weights())
            learner = OnlineLearner(fresh, **config, chronological=chronological)
            r = learner.run(X_te, y_te, seed=seed, opening_ids=oi)
            seeds.append(r)
            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed)
            current_step += 1
            elapsed = time.time() - exp_start
            avg_step = elapsed / current_step
            remaining = avg_step * (total_steps - current_step)

            seed_label = f'  seed {seed}' if not chronological else '  chrono'
            det_str = ''
            if 'opening' in r:
                om = r['opening']
                det_str = f' | det={om["detected"]}/{om["total_pos_instances"]} ({om["detection_rate"]:.1%})'
            print(f'{seed_label}: overall acc={r["overall"]["acc"]:.4f}, '
                  f'F1={r["overall"]["f1"]:.4f} | '
                  f'Q4 acc={r["q4"]["acc"]:.4f}, F1={r["q4"]["f1"]:.4f} | '
                  f'thresh={r["threshold"]:.2f}{det_str} ({fmt_time(step_elapsed)})')
            print(f'  [time] elapsed: {fmt_time(elapsed)} | '
                  f'est. remaining: {fmt_time(remaining)} | '
                  f'step {current_step}/{total_steps}')

        avg = lambda key, sub='overall': np.mean([s[sub][key] for s in seeds])
        print(f'\n  avg online overall:  acc={avg("acc"):.4f}, F1={avg("f1"):.4f}')
        print(f'  avg online Q4:       acc={avg("acc","q4"):.4f}, F1={avg("f1","q4"):.4f}')
        print(f'  improvement (Q4 vs baseline): '
              f'acc {avg("acc","q4")-b_acc:+.4f}, F1 {avg("f1","q4")-b_f1:+.4f}')

        sweep = None
        if oi is not None:
            best_seed = max(seeds, key=lambda s: s['overall']['f1'])
            sweep = threshold_sweep(
                best_seed['probs'], best_seed['trues'], best_seed['oids'])
            print(f'\n  threshold vs detection/FP tradeoff ({name}, {tag}):')
            print(f'  {"thresh":<8} {"det_rate":<9} {"detected":<10} '
                  f'{"fp_win":<8} {"win_fpr":<9} {"f1":<6}')
            print(f'  {"-"*50}')
            for row in sweep:
                print(f'  {row["threshold"]:<8.2f} {row["det_rate"]:<9.1%} '
                      f'{row["detected"]}/{row["total_pos"]:<7} '
                      f'{row["fp_windows"]:<8} {row["window_fpr"]:<9.1%} '
                      f'{row["f1"]:<6.3f}')

        results[name] = {
            'baseline': {'acc': b_acc, 'f1': b_f1, 'prec': b_prec, 'rec': b_rec},
            'baseline_opening': b_opening,
            'seeds': seeds,
            'tag': tag,
            'sweep': sweep,
        }

    return results, time.time() - exp_start


def print_summary(results):
    has_opening = any(r.get('baseline_opening') is not None for r in results.values())

    print('\n' + '='*90)
    print('  online learning vs loso baseline')
    print('='*90)

    if has_opening:
        hdr = (f'{"subject":<14} {"base acc":<10} {"base f1":<10} '
               f'{"online acc":<12} {"online f1":<11} {"q4 f1":<9} '
               f'{"base det%":<10} {"online det%":<12} {"df1":<8}')
    else:
        hdr = (f'{"subject":<14} {"base acc":<10} {"base f1":<10} '
               f'{"online acc":<12} {"online f1":<11} {"q4 acc":<9} '
               f'{"q4 f1":<9} {"df1":<8}')
    print(hdr)
    print('-'*90)

    ba_all, bf_all, oa_all, of_all, qa_all, qf_all = [], [], [], [], [], []
    bd_all, od_all = [], []
    for sub, r in results.items():
        ba, bf = r['baseline']['acc'], r['baseline']['f1']
        oa = np.mean([s['overall']['acc'] for s in r['seeds']])
        of_ = np.mean([s['overall']['f1'] for s in r['seeds']])
        qa = np.mean([s['q4']['acc'] for s in r['seeds']])
        qf = np.mean([s['q4']['f1'] for s in r['seeds']])
        delta = qf - bf
        tag = f' ({r["tag"]})' if r['tag'] == 'stroke' else ''

        if has_opening:
            bd = r['baseline_opening']['detection_rate'] if r['baseline_opening'] else 0
            od_vals = [s['opening']['detection_rate'] for s in r['seeds'] if 'opening' in s]
            od = np.mean(od_vals) if od_vals else 0
            print(f'{sub+tag:<14} {ba:<10.4f} {bf:<10.4f} {oa:<12.4f} '
                  f'{of_:<11.4f} {qf:<9.4f} {bd:<10.1%} {od:<12.1%} {delta:+.4f}')
            bd_all.append(bd); od_all.append(od)
        else:
            print(f'{sub+tag:<14} {ba:<10.4f} {bf:<10.4f} {oa:<12.4f} '
                  f'{of_:<11.4f} {qa:<9.4f} {qf:<9.4f} {delta:+.4f}')

        ba_all.append(ba); bf_all.append(bf)
        oa_all.append(oa); of_all.append(of_)
        qa_all.append(qa); qf_all.append(qf)

    print('-'*90)
    d = np.mean(qf_all) - np.mean(bf_all)
    if has_opening:
        print(f'{"average":<14} {np.mean(ba_all):<10.4f} {np.mean(bf_all):<10.4f} '
              f'{np.mean(oa_all):<12.4f} {np.mean(of_all):<11.4f} '
              f'{np.mean(qf_all):<9.4f} {np.mean(bd_all):<10.1%} '
              f'{np.mean(od_all):<12.1%} {d:+.4f}')
    else:
        print(f'{"average":<14} {np.mean(ba_all):<10.4f} {np.mean(bf_all):<10.4f} '
              f'{np.mean(oa_all):<12.4f} {np.mean(of_all):<11.4f} '
              f'{np.mean(qa_all):<9.4f} {np.mean(qf_all):<9.4f} {d:+.4f}')
    print('='*90)
    print('\nbase = loso baseline (no adaptation)')
    print('online = full sim, q4 = last quarter (most adapted)')
    print('df1 = q4 f1 - baseline f1')
    if has_opening:
        print('det% = fraction of hand openings where >= 1 positive window was caught')


def plot_curves(results, path='online_learning_curves.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Online Learning Adaptation Curves (CNN-LSTM)', fontsize=14, y=1.01)

    for idx, (sub, r) in enumerate(results.items()):
        ax = axes[idx // 2][idx % 2]
        sr = r['seeds'][0]
        x = np.arange(len(sr['run_acc']))

        ax.plot(x, sr['run_acc'], alpha=0.7, label='Running Accuracy',
                color='steelblue', linewidth=1)
        ax.plot(x, sr['run_f1'], alpha=0.7, label='Running F1',
                color='darkorange', linewidth=1)

        if sr.get('oids'):
            run_det = _running_detection_rate(
                np.array(sr['trues']), np.array(sr['preds_list']),
                np.array(sr['oids']))
            if run_det is not None:
                ax.plot(x, run_det, alpha=0.7, label='Running Det. Rate',
                        color='green', linewidth=1)

        for i, up in enumerate(sr['updates']):
            if up < len(x):
                kw = {'label': 'Model updates'} if i == 0 else {}
                ax.axvline(x=up, color='green', linestyle='--', alpha=0.15, **kw)

        ax.axhline(y=r['baseline']['acc'], color='steelblue', linestyle=':',
                   alpha=0.6, label=f'Baseline acc={r["baseline"]["acc"]:.3f}')
        ax.axhline(y=r['baseline']['f1'], color='darkorange', linestyle=':',
                   alpha=0.6, label=f'Baseline F1={r["baseline"]["f1"]:.3f}')

        tag = f' ({r["tag"]})' if r['tag'] == 'stroke' else ''
        det_str = ''
        if 'opening' in sr:
            det_str = f'  Det={sr["opening"]["detection_rate"]:.1%}'
        ax.set_title(f'{sub}{tag}{det_str}')
        ax.set_xlabel('Window #')
        ax.set_ylabel('Metric (sliding window=100)')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'\nplot saved: {path}')


def _running_detection_rate(trues, preds, oids):
    n = len(trues)
    if n == 0:
        return None

    # bail if not monotonic — means data was shuffled so running det rate is meaningless
    if not np.all(np.diff(oids) >= 0):
        return None

    unique_oids = np.unique(oids)
    instance_detected = {}
    instance_last_idx = {}
    for oid in unique_oids:
        mask = oids == oid
        gt = trues[mask]
        pr = preds[mask]
        if np.any(gt == 1):
            instance_detected[oid] = bool(np.any(pr[gt == 1] == 1))
            instance_last_idx[oid] = int(np.where(mask)[0][-1])

    if not instance_detected:
        return None

    events = sorted((idx, det) for oid, det in instance_detected.items()
                    for idx in [instance_last_idx[oid]])

    run_det = np.full(n, np.nan)
    ev_ptr = 0
    cum_detected = 0
    cum_total = 0
    for i in range(n):
        while ev_ptr < len(events) and events[ev_ptr][0] <= i:
            cum_total += 1
            if events[ev_ptr][1]:
                cum_detected += 1
            ev_ptr += 1
        if cum_total > 0:
            run_det[i] = cum_detected / cum_total

    first_valid = np.argmax(~np.isnan(run_det))
    run_det[:first_valid] = run_det[first_valid]

    return run_det


def plot_threshold_tradeoff(results, path='threshold_tradeoff.png'):
    subjects_with_sweep = [(sub, r) for sub, r in results.items() if r.get('sweep')]
    if not subjects_with_sweep:
        return

    n = len(subjects_with_sweep)
    fig, axes = plt.subplots(1, n, figsize=(5 * n, 4.5), squeeze=False)
    fig.suptitle('Opening Detection Rate vs False Positive Rate (Threshold Sweep)',
                 fontsize=13, y=1.03)

    for idx, (sub, r) in enumerate(subjects_with_sweep):
        ax = axes[0][idx]
        sweep = r['sweep']
        threshs = [row['threshold'] for row in sweep]
        det_rates = [row['det_rate'] for row in sweep]
        fprs = [row['window_fpr'] for row in sweep]
        f1s = [row['f1'] for row in sweep]

        ax.plot(fprs, det_rates, 'o-', color='#2d8a4e', markersize=4, linewidth=1.5,
                label='Det Rate vs FPR')

        for i, th in enumerate(threshs):
            if i % 2 == 0:
                ax.annotate(f'{th:.2f}', (fprs[i], det_rates[i]),
                            textcoords='offset points', xytext=(5, 5),
                            fontsize=7, color='#555')

        adapted_th = np.mean([s['threshold'] for s in r['seeds']])
        closest_idx = np.argmin(np.abs(np.array(threshs) - adapted_th))
        ax.plot(fprs[closest_idx], det_rates[closest_idx], '*',
                color='red', markersize=12, label=f'Adapted thresh ({adapted_th:.2f})')

        ax2 = ax.twinx()
        ax2.plot(fprs, f1s, 's--', color='darkorange', markersize=3,
                 linewidth=1, alpha=0.7, label='F1')
        ax2.set_ylabel('F1 Score', fontsize=9, color='darkorange')
        ax2.tick_params(axis='y', labelcolor='darkorange')
        ax2.set_ylim(0, 1.05)

        tag = f' ({r["tag"]})' if r['tag'] == 'stroke' else ''
        ax.set_title(f'{sub}{tag}', fontsize=10, fontweight='bold')
        ax.set_xlabel('Window False Positive Rate', fontsize=9)
        ax.set_ylabel('Opening Detection Rate', fontsize=9)
        ax.set_xlim(-0.02, max(fprs) * 1.1 + 0.02)
        ax.set_ylim(0, 1.05)
        ax.grid(True, alpha=0.2)

        lines1, labels1 = ax.get_legend_handles_labels()
        lines2, labels2 = ax2.get_legend_handles_labels()
        ax.legend(lines1 + lines2, labels1 + labels2, fontsize=7, loc='lower left')

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'threshold tradeoff plot saved: {path}')


def generate_report(results, total_time, path='online_learning_report.png'):
    has_opening = any(r.get('baseline_opening') is not None for r in results.values())

    fig = plt.figure(figsize=(14, 12) if has_opening else (12, 10))
    fig.patch.set_facecolor('white')
    font_family = 'Times New Roman'

    fig.text(0.5, 0.97, 'Summary',
             ha='center', va='top', fontsize=18, fontweight='bold',
             color='#1a1a2e', fontfamily=font_family)
    fig.text(0.5, 0.945, f'Total training time: {fmt_time(total_time)}',
             ha='center', va='top', fontsize=11, color='#555555',
             fontfamily=font_family)

    if has_opening:
        col_labels = ['Subject', 'Type', 'Base Acc', 'Base F1',
                      'Online Acc', 'Online F1', 'Q4 F1',
                      'Base Det%', 'Online Det%', '\u0394F1']
    else:
        col_labels = ['Subject', 'Type', 'Base Acc', 'Base F1',
                      'Online Acc', 'Online F1', 'Q4 F1', '\u0394F1']

    table_data = []
    ba_all, bf_all, oa_all, of_all, qf_all = [], [], [], [], []
    bd_all, od_all = [], []

    for idx, (sub, r) in enumerate(results.items()):
        ba, bf = r['baseline']['acc'], r['baseline']['f1']
        oa = np.mean([s['overall']['acc'] for s in r['seeds']])
        of_ = np.mean([s['overall']['f1'] for s in r['seeds']])
        qf = np.mean([s['q4']['f1'] for s in r['seeds']])
        delta = qf - bf

        row = [sub, r['tag'].capitalize(),
               f'{ba:.4f}', f'{bf:.4f}',
               f'{oa:.4f}', f'{of_:.4f}', f'{qf:.4f}']

        if has_opening:
            bd = r['baseline_opening']['detection_rate'] if r['baseline_opening'] else 0
            od_vals = [s['opening']['detection_rate'] for s in r['seeds'] if 'opening' in s]
            od = np.mean(od_vals) if od_vals else 0
            row.extend([f'{bd:.1%}', f'{od:.1%}'])
            bd_all.append(bd); od_all.append(od)

        row.append(f'{delta:+.4f}')
        table_data.append(row)
        ba_all.append(ba); bf_all.append(bf)
        oa_all.append(oa); of_all.append(of_)
        qf_all.append(qf)

    avg_delta = np.mean(qf_all) - np.mean(bf_all)
    avg_row = ['Average', '',
               f'{np.mean(ba_all):.4f}', f'{np.mean(bf_all):.4f}',
               f'{np.mean(oa_all):.4f}', f'{np.mean(of_all):.4f}',
               f'{np.mean(qf_all):.4f}']
    if has_opening:
        avg_row.extend([f'{np.mean(bd_all):.1%}', f'{np.mean(od_all):.1%}'])
    avg_row.append(f'{avg_delta:+.4f}')
    table_data.append(avg_row)

    ax_table = fig.add_axes([0.03, 0.55, 0.94, 0.35])
    ax_table.axis('off')
    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(9)
    table.scale(1.0, 1.6)

    for _, cell in table.get_celld().items():
        cell.get_text().set_fontfamily(font_family)

    for j in range(len(col_labels)):
        cell = table[0, j]
        cell.set_facecolor('#1a1a2e')
        cell.set_text_props(color='white', fontweight='bold')

    last_row = len(table_data)
    for j in range(len(col_labels)):
        cell = table[last_row, j]
        cell.set_facecolor('#e8e8e8')
        cell.set_text_props(fontweight='bold')

    delta_col = len(col_labels) - 1
    for i in range(1, len(table_data) + 1):
        val = float(table_data[i-1][delta_col])
        cell = table[i, delta_col]
        if val > 0:
            cell.set_text_props(color='#2d8a4e')
        elif val < 0:
            cell.set_text_props(color='#c0392b')

    row_y = [0.30, 0.04]
    axes_curves = []
    for idx in range(4):
        ax = fig.add_axes([0.07 + (idx % 2) * 0.48, row_y[idx // 2], 0.42, 0.20])
        axes_curves.append(ax)

    for idx, (sub, r) in enumerate(results.items()):
        ax = axes_curves[idx]
        sr = r['seeds'][0]
        x = np.arange(len(sr['run_acc']))
        ax.plot(x, sr['run_acc'], alpha=0.7, color='steelblue', linewidth=0.8, label='Acc')
        ax.plot(x, sr['run_f1'], alpha=0.7, color='darkorange', linewidth=0.8, label='F1')
        ax.axhline(y=r['baseline']['f1'], color='darkorange', linestyle=':', alpha=0.5, linewidth=0.8)
        ax.axhline(y=r['baseline']['acc'], color='steelblue', linestyle=':', alpha=0.5, linewidth=0.8)
        tag = f' ({r["tag"]})' if r['tag'] == 'stroke' else ''
        det_str = ''
        if 'opening' in sr:
            det_str = f'  Det={sr["opening"]["detection_rate"]:.1%}'
        ax.set_title(f'{sub}{tag}{det_str}', fontsize=9, fontweight='bold', fontfamily=font_family)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Window #', fontsize=7, fontfamily=font_family)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15)
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')

    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'report saved: {path}')


if __name__ == '__main__':
    parser = argparse.ArgumentParser(
        description='Online Learning CNN-LSTM for Hand Exoskeleton Intent Detection')
    parser.add_argument('--mode', choices=['chrono', 'random', 'both'], default='chrono',
                        help='chrono = chronological order (1 seed), '
                             'random = shuffled (3 seeds), '
                             'both = run both and compare')
    args = parser.parse_args()

    total_start = time.time()

    print('='*60)
    print('  online learning cnn-lstm — hand exo intent detection')
    print('='*60)
    print('\n running on cpu')

    config = dict(
        update_every=50,
        warmup=50,
        ft_epochs=3,
        ft_bs=64,
        ft_lr=5e-4,
        freeze_early=True,
        adapt_thresh=True,
        buf_cap=300,
    )
    print(f'\nonline learning config:')
    for k, v in config.items():
        print(f'  {k}: {v}')

    modes_to_run = ['chrono', 'random'] if args.mode == 'both' else [args.mode]

    all_results = {}
    for mode in modes_to_run:
        chronological = (mode == 'chrono')
        n_seeds = 1 if chronological else 3

        print(f'\n{"#"*60}')
        print(f'  mode: {"chronological" if chronological else "randomized"} '
              f'({n_seeds} seed{"s" if n_seeds > 1 else ""})')
        print(f'{"#"*60}')

        print('\nloading data...')
        sX, sy, s_info = load_data_chrono()
        for i in range(4):
            tag = 'stroke' if i == 3 else 'healthy'
            n_instances = len(np.unique(s_info[i][:, 2]))
            print(f'  subject {i+1} ({tag}): {len(sy[i])} windows '
                  f'({np.sum(sy[i]==1)} open, {np.sum(sy[i]==0)} not), '
                  f'{n_instances} opening instances')

        results, experiment_time = run_experiment(
            sX, sy, config, n_seeds=n_seeds,
            subjects_info=s_info, chronological=chronological)

        suffix = f'_{mode}'
        print_summary(results)
        plot_curves(results, path=f'online_learning_curves{suffix}.png')
        generate_report(results, experiment_time, path=f'online_learning_report{suffix}.png')
        plot_threshold_tradeoff(results, path=f'threshold_tradeoff{suffix}.png')

        all_results[mode] = results

    if args.mode == 'both' and len(all_results) == 2:
        print(f'\n{"="*80}')
        print('  chrono vs randomized comparison')
        print(f'{"="*80}')
        print(f'{"subject":<14} {"chrono f1":<11} {"random f1":<11} '
              f'{"chrono det%":<13} {"random det%":<13} {"df1 (c-r)":<10}')
        print('-'*80)
        for sub in all_results['chrono']:
            c = all_results['chrono'][sub]
            r = all_results['random'][sub]
            c_f1 = np.mean([s['overall']['f1'] for s in c['seeds']])
            r_f1 = np.mean([s['overall']['f1'] for s in r['seeds']])
            c_det = np.mean([s['opening']['detection_rate'] for s in c['seeds'] if 'opening' in s])
            r_det = np.mean([s['opening']['detection_rate'] for s in r['seeds'] if 'opening' in s])
            tag = f' ({c["tag"]})' if c['tag'] == 'stroke' else ''
            print(f'{sub+tag:<14} {c_f1:<11.4f} {r_f1:<11.4f} '
                  f'{c_det:<13.1%} {r_det:<13.1%} {c_f1-r_f1:+.4f}')
        print('='*80)

    total_time = time.time() - total_start
    print(f'\n  total time: {fmt_time(total_time)}')
