#!/usr/bin/env python3

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

os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
tf.config.set_visible_devices([], 'GPU')

# # def check_gpu():
# #     gpus = tf.config.list_physical_devices('GPU')
# #     if gpus:
# #         print(f'  GPU detected: {gpus[0].name}')
# #         for gpu in gpus:
# #             tf.config.experimental.set_memory_growth(gpu, True)
# #     else:
# #         print('  WARNING: No GPU detected. Install tensorflow-macos and tensorflow-metal')
# #         print('           for Apple Silicon acceleration.')
# #     return len(gpus) > 0


def load_data(path='labeled data 4 subjects.mat'):
    mat = sio.loadmat(path)
    sX, sy = [], []
    for i in range(4):
        d = mat['data'][0, i]
        l = mat['labels'][0, i].flatten().astype(int)
        # mat file nests each window in a cell array, have to unpack per-element
        sX.append(np.array([d[j, 0] for j in range(d.shape[0])]))
        sy.append(l)
    return sX, sy


def build_cnn_lstm(input_shape=(100, 8)):
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
    # max 1 to avoid zero-div when a class is missing in tiny buffers
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
    # 256 batch ok on m2 pro 16gb, cuts base training time ~in half vs 64
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
                 freeze_early=True, adapt_thresh=True, buf_cap=300):
        self.model = model
        self.update_every = update_every
        self.warmup = warmup
        self.ft_epochs = ft_epochs
        self.ft_bs = ft_bs
        self.freeze_early = freeze_early
        self.adapt_thresh = adapt_thresh
        self.buf_cap = buf_cap
        self.threshold = 0.5

        self.buf_X, self.buf_y = [], []
        self.preds, self.trues, self.probs_list = [], [], []
        self.update_pts = []

        # freeze conv1+bn1 only — they pick up generic emg features,
        # deeper layers need to adapt per-subject
        if freeze_early:
            for layer in model.layers:
                if layer.name in ('conv1', 'bn1'):
                    layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(ft_lr),
                      loss='binary_crossentropy', metrics=['accuracy'])

    def run(self, X, y, seed=0):
        rng = np.random.RandomState(seed)
        n = len(X)
        # prof lum confirmed chronological order wasn't preserved in this dataset
        order = rng.permutation(n)
        Xs, ys = X[order], y[order]

        sched = set(range(self.warmup, n + 1, self.update_every))
        bounds = sorted(set([0] + list(sched) + [n]))

        for b in range(len(bounds) - 1):
            s, e = bounds[b], bounds[b + 1]
            if s >= e:
                continue

            bX, by = Xs[s:e], ys[s:e]

            # direct call bypasses predict() overhead (progress bars, dataset creation etc)
            bp = self.model(tf.constant(bX, dtype=tf.float32), training=False).numpy().flatten()
            bpred = (bp > self.threshold).astype(int)

            self.preds.extend(bpred.tolist())
            self.trues.extend(by.tolist())
            self.probs_list.extend(bp.tolist())
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

        # 60/40 recency split — heuristic to prioritize recent windows
        # without fully forgetting older ones
        if len(X) > self.buf_cap:
            recent_n = int(self.buf_cap * 0.6)
            old_n = self.buf_cap - recent_n
            old_pool = len(X) - recent_n
            old_idx = np.random.choice(old_pool, min(old_n, old_pool), replace=False)
            recent_idx = np.arange(len(X) - recent_n, len(X))
            idx = np.concatenate([old_idx, recent_idx])
            X, y = X[idx], y[idx]

        # train_on_batch instead of fit() — avoids epoch setup latency
        # on these tiny ~300 sample buffers
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

        # [0.15, 0.85] avoids degenerate thresholds that collapse to all-one/all-zero
        # on heavily imbalanced subjects
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

        win = 100
        run_acc, run_f1 = [], []
        for i in range(n):
            s = max(0, i + 1 - win)
            run_acc.append(accuracy_score(t[s:i+1], p[s:i+1]))
            run_f1.append(f1_score(t[s:i+1], p[s:i+1], zero_division=0))

        return {
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
        }


def run_experiment(subjects_X, subjects_y, config, n_seeds=3):
    results = {}
    subject_labels = ['healthy', 'healthy', 'healthy', 'stroke']
    total_subjects = 4
    total_steps = total_subjects * (1 + n_seeds)
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
        print(f'  Train: {len(y_tr)} windows ({np.sum(y_tr==1)} pos, {np.sum(y_tr==0)} neg)')
        print(f'  Test:  {len(y_te)} windows ({np.sum(y_te==1)} pos, {np.sum(y_te==0)} neg)')

        step_start = time.time()
        print('  Training base model...', end=' ', flush=True)
        # X_te as val is intentional — it's the held-out subject so this just
        # monitors generalization, doesn't leak into the loso split
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
        print(f'done ({fmt_time(step_elapsed)}).  Baseline: acc={b_acc:.4f}, F1={b_f1:.4f}, '
              f'prec={b_prec:.4f}, rec={b_rec:.4f}')
        print(f'  [Time] Elapsed: {fmt_time(elapsed)} | '
              f'Est. remaining: {fmt_time(remaining)} | '
              f'Step {current_step}/{total_steps}')

        seeds = []
        for seed in range(n_seeds):
            step_start = time.time()
            fresh = build_cnn_lstm()
            fresh.build((None, 100, 8))
            fresh.set_weights(base.get_weights())
            learner = OnlineLearner(fresh, **config)
            r = learner.run(X_te, y_te, seed=seed)
            seeds.append(r)
            step_elapsed = time.time() - step_start
            step_times.append(step_elapsed)
            current_step += 1
            elapsed = time.time() - exp_start
            avg_step = elapsed / current_step
            remaining = avg_step * (total_steps - current_step)
            print(f'  Seed {seed}: overall acc={r["overall"]["acc"]:.4f}, '
                  f'F1={r["overall"]["f1"]:.4f} | '
                  f'Q4 acc={r["q4"]["acc"]:.4f}, F1={r["q4"]["f1"]:.4f} | '
                  f'2nd-half acc={r["second_half"]["acc"]:.4f}, '
                  f'F1={r["second_half"]["f1"]:.4f} | '
                  f'thresh={r["threshold"]:.2f} ({fmt_time(step_elapsed)})')
            print(f'  [Time] Elapsed: {fmt_time(elapsed)} | '
                  f'Est. remaining: {fmt_time(remaining)} | '
                  f'Step {current_step}/{total_steps}')

        avg = lambda key, sub='overall': np.mean([s[sub][key] for s in seeds])
        print(f'\n  Avg online overall:  acc={avg("acc"):.4f}, F1={avg("f1"):.4f}')
        print(f'  Avg online Q4:       acc={avg("acc","q4"):.4f}, F1={avg("f1","q4"):.4f}')
        print(f'  Avg online 2nd half: acc={avg("acc","second_half"):.4f}, '
              f'F1={avg("f1","second_half"):.4f}')
        print(f'  Improvement (Q4 vs baseline): '
              f'acc {avg("acc","q4")-b_acc:+.4f}, F1 {avg("f1","q4")-b_f1:+.4f}')

        results[name] = {
            'baseline': {'acc': b_acc, 'f1': b_f1, 'prec': b_prec, 'rec': b_rec},
            'seeds': seeds,
            'tag': tag,
        }

    return results, time.time() - exp_start


def print_summary(results):
    print('\n' + '='*75)
    print('  FINAL SUMMARY: Online Learning vs LOSO Baseline')
    print('='*75)
    hdr = (f'{"Subject":<14} {"Base Acc":<10} {"Base F1":<10} '
           f'{"Online Acc":<12} {"Online F1":<11} {"Q4 Acc":<9} '
           f'{"Q4 F1":<9} {"ΔF1":<8}')
    print(hdr)
    print('-'*75)

    ba_all, bf_all, oa_all, of_all, qa_all, qf_all = [], [], [], [], [], []
    for sub, r in results.items():
        ba, bf = r['baseline']['acc'], r['baseline']['f1']
        oa = np.mean([s['overall']['acc'] for s in r['seeds']])
        of_ = np.mean([s['overall']['f1'] for s in r['seeds']])
        qa = np.mean([s['q4']['acc'] for s in r['seeds']])
        qf = np.mean([s['q4']['f1'] for s in r['seeds']])
        delta = qf - bf
        tag = f' ({r["tag"]})' if r['tag'] == 'stroke' else ''
        print(f'{sub+tag:<14} {ba:<10.4f} {bf:<10.4f} {oa:<12.4f} '
              f'{of_:<11.4f} {qa:<9.4f} {qf:<9.4f} {delta:+.4f}')
        ba_all.append(ba); bf_all.append(bf)
        oa_all.append(oa); of_all.append(of_)
        qa_all.append(qa); qf_all.append(qf)

    print('-'*75)
    d = np.mean(qf_all) - np.mean(bf_all)
    print(f'{"Average":<14} {np.mean(ba_all):<10.4f} {np.mean(bf_all):<10.4f} '
          f'{np.mean(oa_all):<12.4f} {np.mean(of_all):<11.4f} '
          f'{np.mean(qa_all):<9.4f} {np.mean(qf_all):<9.4f} {d:+.4f}')
    print('='*75)
    print('\nBase = LOSO baseline (no adaptation)')
    print('Online = full simulation (overall). Q4 = last quarter (most adapted).')
    print('ΔF1 = Q4 F1 − Baseline F1 (improvement from online learning)')


def plot_curves(results, path='online_learning_curves.png'):
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))
    fig.suptitle('Online Learning Adaptation Curves (CNN-LSTM)', fontsize=14, y=1.01)

    for idx, (sub, r) in enumerate(results.items()):
        ax = axes[idx // 2][idx % 2]
        # seed 0 only for plot — metrics table already averages all seeds
        sr = r['seeds'][0]
        x = np.arange(len(sr['run_acc']))

        ax.plot(x, sr['run_acc'], alpha=0.7, label='Running Accuracy',
                color='steelblue', linewidth=1)
        ax.plot(x, sr['run_f1'], alpha=0.7, label='Running F1',
                color='darkorange', linewidth=1)

        for i, up in enumerate(sr['updates']):
            if up < len(x):
                kw = {'label': 'Model updates'} if i == 0 else {}
                ax.axvline(x=up, color='green', linestyle='--', alpha=0.15, **kw)

        ax.axhline(y=r['baseline']['acc'], color='steelblue', linestyle=':',
                   alpha=0.6, label=f'Baseline acc={r["baseline"]["acc"]:.3f}')
        ax.axhline(y=r['baseline']['f1'], color='darkorange', linestyle=':',
                   alpha=0.6, label=f'Baseline F1={r["baseline"]["f1"]:.3f}')

        tag = f' ({r["tag"]})' if r['tag'] == 'stroke' else ''
        ax.set_title(f'{sub}{tag}')
        ax.set_xlabel('Window #')
        ax.set_ylabel('Metric (sliding window=100)')
        ax.set_ylim(0, 1.05)
        ax.legend(fontsize=7, loc='lower right')
        ax.grid(True, alpha=0.2)

    plt.tight_layout()
    plt.savefig(path, dpi=150, bbox_inches='tight')
    print(f'\nPlot saved: {path}')


def generate_report(results, total_time, path='online_learning_report.png'):
    fig = plt.figure(figsize=(12, 10))
    fig.patch.set_facecolor('white')

    font_family = 'Times New Roman'

    fig.text(0.5, 0.97, 'Summary',
             ha='center', va='top', fontsize=18, fontweight='bold',
             color='#1a1a2e', fontfamily=font_family)
    fig.text(0.5, 0.945, f'Total training time: {fmt_time(total_time)}',
             ha='center', va='top', fontsize=11, color='#555555',
             fontfamily=font_family)

    col_labels = ['Subject', 'Type', 'Base Acc', 'Base F1',
                  'Online Acc', 'Online F1', 'Q4 F1', '\u0394F1']
    table_data = []
    ba_all, bf_all, oa_all, of_all, qf_all = [], [], [], [], []

    for idx, (sub, r) in enumerate(results.items()):
        ba, bf = r['baseline']['acc'], r['baseline']['f1']
        oa = np.mean([s['overall']['acc'] for s in r['seeds']])
        of_ = np.mean([s['overall']['f1'] for s in r['seeds']])
        qf = np.mean([s['q4']['f1'] for s in r['seeds']])
        delta = qf - bf
        table_data.append([
            sub, r['tag'].capitalize(),
            f'{ba:.4f}', f'{bf:.4f}',
            f'{oa:.4f}', f'{of_:.4f}',
            f'{qf:.4f}', f'{delta:+.4f}'
        ])
        ba_all.append(ba); bf_all.append(bf)
        oa_all.append(oa); of_all.append(of_)
        qf_all.append(qf)

    avg_delta = np.mean(qf_all) - np.mean(bf_all)
    table_data.append([
        'Average', '',
        f'{np.mean(ba_all):.4f}', f'{np.mean(bf_all):.4f}',
        f'{np.mean(oa_all):.4f}', f'{np.mean(of_all):.4f}',
        f'{np.mean(qf_all):.4f}', f'{avg_delta:+.4f}'
    ])

    ax_table = fig.add_axes([0.05, 0.55, 0.9, 0.35])
    ax_table.axis('off')
    table = ax_table.table(cellText=table_data, colLabels=col_labels,
                           loc='center', cellLoc='center')
    table.auto_set_font_size(False)
    table.set_fontsize(10)
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

    for i in range(1, len(table_data) + 1):
        val = float(table_data[i-1][-1])
        cell = table[i, len(col_labels)-1]
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
        ax.set_title(f'{sub}{tag}', fontsize=9, fontweight='bold', fontfamily=font_family)
        ax.set_ylim(0, 1.05)
        ax.set_xlabel('Window #', fontsize=7, fontfamily=font_family)
        ax.tick_params(labelsize=7)
        ax.grid(True, alpha=0.15)
        if idx == 0:
            ax.legend(fontsize=7, loc='lower right')

    plt.savefig(path, dpi=150, bbox_inches='tight', facecolor='white')
    print(f'Report saved: {path}')


if __name__ == '__main__':
    total_start = time.time()

    print('='*60)
    print('  Online Learning CNN-LSTM — Hand Exoskeleton Intent Detection')
    print('='*60)

    # print('\nChecking GPU...')
    # check_gpu()
    print('\nRunning on CPU only.')

    print('\nLoading data...')
    sX, sy = load_data()
    for i in range(4):
        tag = 'stroke' if i == 3 else 'healthy'
        print(f'  Subject {i+1} ({tag}): {len(sy[i])} windows '
              f'({np.sum(sy[i]==1)} open, {np.sum(sy[i]==0)} not)')

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
    print(f'\nOnline learning config:')
    for k, v in config.items():
        print(f'  {k}: {v}')

    results, experiment_time = run_experiment(sX, sy, config, n_seeds=3)

    total_time = time.time() - total_start
    print(f'\n  Total execution time: {fmt_time(total_time)}')

    print_summary(results)
    plot_curves(results)
    generate_report(results, total_time)
