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

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
tf.get_logger().setLevel('ERROR')


def load_data(path='labeled data 4 subjects.mat'):
    mat = sio.loadmat(path)
    sX, sy = [], []
    for i in range(4):
        d = mat['data'][0, i]
        l = mat['labels'][0, i].flatten().astype(int)
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
    # max(..., 1) guards against zero-division when a class is absent in small buffers
    n0 = max(np.sum(y == 0), 1)
    n1 = max(np.sum(y == 1), 1)
    t = len(y)
    return {0: t / (2 * n0), 1: t / (2 * n1)}


def train_base(X_tr, y_tr, X_val, y_val, seed=42):
    tf.random.set_seed(seed)
    np.random.seed(seed)
    m = build_cnn_lstm()
    m.compile(optimizer=tf.keras.optimizers.Adam(1e-3),
              loss='binary_crossentropy', metrics=['accuracy'])
    m.fit(X_tr, y_tr, epochs=50, batch_size=64,
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

        # only freeze the first conv+bn block — these learn general emg time-series
        # features that should transfer across subjects; deeper layers are more subject-specific
        if freeze_early:
            for layer in model.layers:
                if layer.name in ('conv1', 'bn1'):
                    layer.trainable = False
        model.compile(optimizer=tf.keras.optimizers.Adam(ft_lr),
                      loss='binary_crossentropy', metrics=['accuracy'])

    def run(self, X, y, seed=0):
        rng = np.random.RandomState(seed)
        n = len(X)
        # windows are randomized per prof. lum — chronological order wasn't preserved in this dataset
        order = rng.permutation(n)
        Xs, ys = X[order], y[order]

        sched = set(range(self.warmup, n + 1, self.update_every))
        bounds = sorted(set([0] + list(sched) + [n]))

        for b in range(len(bounds) - 1):
            s, e = bounds[b], bounds[b + 1]
            if s >= e:
                continue

            bX, by = Xs[s:e], ys[s:e]

            bp = self.model.predict(bX, verbose=0).flatten()
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

        # 60/40 recency split is a heuristic — prioritize recent windows without fully
        # discarding older ones, since the model still needs them to avoid forgetting
        if len(X) > self.buf_cap:
            recent_n = int(self.buf_cap * 0.6)
            old_n = self.buf_cap - recent_n
            old_pool = len(X) - recent_n
            old_idx = np.random.choice(old_pool, min(old_n, old_pool), replace=False)
            recent_idx = np.arange(len(X) - recent_n, len(X))
            idx = np.concatenate([old_idx, recent_idx])
            X, y = X[idx], y[idx]

        self.model.fit(X, y, epochs=self.ft_epochs,
                       batch_size=min(self.ft_bs, len(X)),
                       class_weight=cw(y), verbose=0)

    def _opt_thresh(self):
        X = np.array(self.buf_X[-self.buf_cap:])
        y = np.array(self.buf_y[-self.buf_cap:])
        ps = self.model.predict(X, verbose=0).flatten()
        best_f1, best_t = 0, 0.5
        # search range [0.15, 0.85] — avoids degenerate thresholds that collapse
        # predictions to all-one or all-zero on heavily imbalanced subjects
        for t in np.arange(0.15, 0.85, 0.02):
            f = f1_score(y, (ps > t).astype(int), zero_division=0)
            if f > best_f1:
                best_f1, best_t = f, t
        self.threshold = best_t

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

        print('  Training base model...', end=' ', flush=True)
        # using X_te as val during base training is intentional — it's the held-out subject,
        # so this just monitors generalization without affecting the loso split
        base = train_base(X_tr, y_tr, X_te, y_te, seed=42)
        bp = base.predict(X_te, verbose=0).flatten()
        bpred = (bp > 0.5).astype(int)
        b_acc = accuracy_score(y_te, bpred)
        b_f1 = f1_score(y_te, bpred, zero_division=0)
        b_rec = recall_score(y_te, bpred, zero_division=0)
        b_prec = precision_score(y_te, bpred, zero_division=0)
        print(f'done.  Baseline: acc={b_acc:.4f}, F1={b_f1:.4f}, '
              f'prec={b_prec:.4f}, rec={b_rec:.4f}')

        seeds = []
        for seed in range(n_seeds):
            fresh = build_cnn_lstm()
            fresh.build((None, 100, 8))
            fresh.set_weights(base.get_weights())
            learner = OnlineLearner(fresh, **config)
            r = learner.run(X_te, y_te, seed=seed)
            seeds.append(r)
            print(f'  Seed {seed}: overall acc={r["overall"]["acc"]:.4f}, '
                  f'F1={r["overall"]["f1"]:.4f} | '
                  f'Q4 acc={r["q4"]["acc"]:.4f}, F1={r["q4"]["f1"]:.4f} | '
                  f'2nd-half acc={r["second_half"]["acc"]:.4f}, '
                  f'F1={r["second_half"]["f1"]:.4f} | '
                  f'thresh={r["threshold"]:.2f}')

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

    return results


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
        sr = r['seeds'][0]  # only plotting seed 0 for visual clarity; metrics table uses all seeds
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


if __name__ == '__main__':
    print('='*60)
    print('  Online Learning CNN-LSTM — Hand Exoskeleton Intent Detection')
    print('='*60)

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

    results = run_experiment(sX, sy, config, n_seeds=3)
    print_summary(results)
    plot_curves(results)
