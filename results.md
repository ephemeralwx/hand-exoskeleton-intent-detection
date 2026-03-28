# Online Learning CNN-LSTM Results

Total execution time: 16m 37s

## Online Learning vs LOSO Baseline

| Subject | Base Acc | Base F1 | Online Acc | Online F1 | Q4 F1 | Base Det% | Online Det% | ΔF1 |
|---|---|---|---|---|---|---|---|---|
| Subject 1 | 0.6573 | 0.5758 | 0.8373 | 0.7177 | 0.7724 | 89.0% | 93.4% | +0.1966 |
| Subject 2 | 0.6683 | 0.5070 | 0.7580 | 0.6479 | 0.6971 | 64.4% | 93.1% | +0.1901 |
| Subject 3 | 0.8066 | 0.5811 | 0.8573 | 0.7504 | 0.8059 | 66.8% | 95.1% | +0.2248 |
| Subject 4 (stroke) | 0.7587 | 0.5227 | 0.8479 | 0.6505 | 0.7028 | 94.7% | 98.2% | +0.1801 |
| **Average** | **0.7227** | **0.5466** | **0.8251** | **0.6916** | **0.7445** | **78.7%** | **95.0%** | **+0.1979** |

- **Base** = LOSO baseline (no adaptation)
- **Online** = full simulation (overall), **Q4** = last quarter (most adapted)
- **ΔF1** = Q4 F1 - Baseline F1
- **Det%** = fraction of hand openings where at least 1 positive window was detected

## Chronological vs Randomized

| Subject | Chrono F1 | Random F1 | Chrono Det% | Random Det% | ΔF1 (C-R) |
|---|---|---|---|---|---|
| Subject 1 | 0.6730 | 0.7177 | 88.3% | 93.4% | -0.0447 |
| Subject 2 | 0.6093 | 0.6479 | 80.0% | 93.1% | -0.0385 |
| Subject 3 | 0.6670 | 0.7504 | 87.0% | 95.1% | -0.0834 |
| Subject 4 (stroke) | 0.6002 | 0.6505 | 94.7% | 98.2% | -0.0504 |

## Threshold vs Detection/FP Tradeoff (Subject 4, stroke)

| Threshold | Det Rate | Detected | FP Windows | Window FPR | F1 |
|---|---|---|---|---|---|
| 0.30 | 100.0% | 76/76 | 438 | 23.7% | 0.620 |
| 0.35 | 100.0% | 76/76 | 412 | 22.2% | 0.625 |
| 0.40 | 100.0% | 76/76 | 382 | 20.6% | 0.626 |
| 0.45 | 100.0% | 76/76 | 360 | 19.4% | 0.633 |
| 0.50 | 100.0% | 76/76 | 331 | 17.9% | 0.641 |
| 0.55 | 98.7% | 75/76 | 297 | 16.0% | 0.648 |
| 0.60 | 98.7% | 75/76 | 260 | 14.0% | 0.653 |
| 0.65 | 98.7% | 75/76 | 229 | 12.4% | 0.657 |
| 0.70 | 98.7% | 75/76 | 199 | 10.7% | 0.658 |
| 0.75 | 96.1% | 73/76 | 173 | 9.3% | 0.655 |
| 0.80 | 96.1% | 73/76 | 147 | 7.9% | 0.651 |
| 0.85 | 94.7% | 72/76 | 104 | 5.6% | 0.631 |
| 0.90 | 90.8% | 69/76 | 70 | 3.8% | 0.565 |
