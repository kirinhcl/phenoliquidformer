"""Compute all statistical tables for the CEA paper.

Outputs to results/paper_stats/:
    benchmark_stats.json   LT vs Transformer paired tests + bootstrap 95% CI.
    ablation_stats.json    Full vs Drop-X Wilcoxon + Cohen's d + R².
    lora_stats.json        Kept aggregate (per-fold unavailable).

Run:
    python scripts/compute_paper_stats.py
"""
from __future__ import annotations

import json
import os
from pathlib import Path

import numpy as np
from scipy import stats

REPO = Path(__file__).parent.parent
RES = REPO / 'results'
OUT = RES / 'paper_stats'
OUT.mkdir(exist_ok=True)

RNG = np.random.default_rng(42)
N_BOOT = 10_000


def load_per_fold(run: str, exp: str = 'exp01'):
    root = RES / run / exp
    folds = sorted(
        (d for d in os.listdir(root) if d.startswith('fold_')),
        key=lambda s: int(s.split('_')[1]),
    )
    mae, preds, trues = [], [], []
    for f in folds:
        m = json.load(open(root / f / 'metrics.json'))
        mae.append(m['test_mae_g'])
        preds.extend(m['test_preds'])
        trues.extend(m['test_trues'])
    return np.array(mae), np.array(preds), np.array(trues)


def bootstrap_mae_ci(preds, trues, n_boot=N_BOOT):
    abs_err = np.abs(preds - trues)
    n = len(abs_err)
    boot = np.array([abs_err[RNG.integers(0, n, n)].mean() for _ in range(n_boot)])
    return float(abs_err.mean()), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def bootstrap_r2_ci(preds, trues, n_boot=N_BOOT):
    n = len(preds)

    def r2(p, t):
        ss_tot = ((t - t.mean()) ** 2).sum()
        return 1 - ((p - t) ** 2).sum() / ss_tot if ss_tot > 0 else float('nan')

    boot = np.empty(n_boot)
    for i in range(n_boot):
        idx = RNG.integers(0, n, n)
        boot[i] = r2(preds[idx], trues[idx])
    return float(r2(preds, trues)), float(np.quantile(boot, 0.025)), float(np.quantile(boot, 0.975))


def cohens_d_paired(a, b):
    diff = a - b
    sd = diff.std(ddof=1)
    return float(diff.mean() / sd) if sd > 0 else float('nan')


def sig_stars(p):
    return '***' if p < 0.001 else '**' if p < 0.01 else '*' if p < 0.05 else 'ns'


# 1. Benchmark ----------------------------------------------------------------
lt_mae, lt_p, lt_t = load_per_fold('lt_h64_n2_res')
tr_mae, tr_p, tr_t = load_per_fold('yield_transformer')
assert len(lt_mae) == len(tr_mae) == 48

w = stats.wilcoxon(lt_mae, tr_mae)
tt = stats.ttest_rel(lt_mae, tr_mae)
d = cohens_d_paired(lt_mae, tr_mae)

lt_mae_pt, lt_mae_lo, lt_mae_hi = bootstrap_mae_ci(lt_p, lt_t)
lt_r2_pt, lt_r2_lo, lt_r2_hi = bootstrap_r2_ci(lt_p, lt_t)
tr_mae_pt, tr_mae_lo, tr_mae_hi = bootstrap_mae_ci(tr_p, tr_t)
tr_r2_pt, tr_r2_lo, tr_r2_hi = bootstrap_r2_ci(tr_p, tr_t)

benchmark = {
    'liquid_transformer': {
        'fold_mean_mae': float(lt_mae.mean()),
        'fold_std_mae': float(lt_mae.std(ddof=1)),
        'mae_bootstrap': {'point': lt_mae_pt, 'ci95_low': lt_mae_lo, 'ci95_high': lt_mae_hi},
        'r2_bootstrap': {'point': lt_r2_pt, 'ci95_low': lt_r2_lo, 'ci95_high': lt_r2_hi},
    },
    'pure_transformer': {
        'fold_mean_mae': float(tr_mae.mean()),
        'fold_std_mae': float(tr_mae.std(ddof=1)),
        'mae_bootstrap': {'point': tr_mae_pt, 'ci95_low': tr_mae_lo, 'ci95_high': tr_mae_hi},
        'r2_bootstrap': {'point': tr_r2_pt, 'ci95_low': tr_r2_lo, 'ci95_high': tr_r2_hi},
    },
    'lt_vs_transformer': {
        'wilcoxon_statistic': float(w.statistic),
        'wilcoxon_p': float(w.pvalue),
        'wilcoxon_sig': sig_stars(w.pvalue),
        'paired_t_statistic': float(tt.statistic),
        'paired_t_p': float(tt.pvalue),
        'paired_t_sig': sig_stars(tt.pvalue),
        'cohens_d': d,
        'mean_diff_g': float((lt_mae - tr_mae).mean()),
    },
    'n_folds': int(len(lt_mae)),
    'n_boot': N_BOOT,
}

(OUT / 'benchmark_stats.json').write_text(json.dumps(benchmark, indent=2))
print('[benchmark]')
print(f'  LT  MAE={lt_mae_pt:.2f} [{lt_mae_lo:.2f}, {lt_mae_hi:.2f}]  '
      f'R²={lt_r2_pt:.2f} [{lt_r2_lo:.2f}, {lt_r2_hi:.2f}]')
print(f'  TR  MAE={tr_mae_pt:.2f} [{tr_mae_lo:.2f}, {tr_mae_hi:.2f}]  '
      f'R²={tr_r2_pt:.2f} [{tr_r2_lo:.2f}, {tr_r2_hi:.2f}]')
print(f'  Wilcoxon p={w.pvalue:.4f} ({sig_stars(w.pvalue)})  Cohen d={d:+.2f}')

# 2. Ablation -----------------------------------------------------------------
ablations = {
    'full':       'lt_h64_n2_res',
    'drop_image': 'lt_ablation/drop_image',
    'drop_fluor': 'lt_ablation/drop_fluor',
    'image_only': 'lt_ablation/image_only',
    'fluor_only': 'lt_ablation/fluor_only',
}
per_fold, preds_map = {}, {}
for name, run in ablations.items():
    mae, p, t = load_per_fold(run)
    per_fold[name] = mae
    preds_map[name] = (p, t)

full_mae = per_fold['full']
ablation_rows = {}
for name in ['drop_image', 'drop_fluor', 'image_only', 'fluor_only']:
    other = per_fold[name]
    w_abl = stats.wilcoxon(full_mae, other)
    d_abl = cohens_d_paired(other, full_mae)  # positive = variant worse
    p_arr, t_arr = preds_map[name]
    ss_tot = ((t_arr - t_arr.mean()) ** 2).sum()
    r2 = 1 - ((p_arr - t_arr) ** 2).sum() / ss_tot
    delta = (other.mean() - full_mae.mean()) / full_mae.mean() * 100
    mae_pt, mae_lo, mae_hi = bootstrap_mae_ci(p_arr, t_arr)
    ablation_rows[name] = {
        'fold_mean_mae': float(other.mean()),
        'delta_pct': float(delta),
        'r2': float(r2),
        'mae_bootstrap_ci': [mae_lo, mae_hi],
        'wilcoxon_statistic': float(w_abl.statistic),
        'wilcoxon_p': float(w_abl.pvalue),
        'wilcoxon_sig': sig_stars(w_abl.pvalue),
        'cohens_d_variant_minus_full': d_abl,
    }

p_arr, t_arr = preds_map['full']
ss_tot = ((t_arr - t_arr.mean()) ** 2).sum()
full_r2 = 1 - ((p_arr - t_arr) ** 2).sum() / ss_tot
mae_pt, mae_lo, mae_hi = bootstrap_mae_ci(p_arr, t_arr)
ablation_rows['full'] = {
    'fold_mean_mae': float(full_mae.mean()),
    'delta_pct': 0.0,
    'r2': float(full_r2),
    'mae_bootstrap_ci': [mae_lo, mae_hi],
}

(OUT / 'ablation_stats.json').write_text(json.dumps(ablation_rows, indent=2))
print('\n[ablation]')
for k, v in ablation_rows.items():
    extra = ''
    if 'wilcoxon_p' in v:
        extra = (f"  Δ={v['delta_pct']:+.1f}%  p={v['wilcoxon_p']:.4f} "
                 f"({v['wilcoxon_sig']})  d={v['cohens_d_variant_minus_full']:+.2f}")
    print(f'  {k:12s}  MAE={v["fold_mean_mae"]:.2f}  R²={v["r2"]:.2f}{extra}')

# 3. LoRA ---------------------------------------------------------------------
lora_raw = json.load(open(RES / 'lora_lt_transfer_fixed' / 'lora_summary.json'))
lora_out = {
    'note': ('lora_summary.json stores only aggregate MAE/R² (n=40 plants). '
             'Per-fold deltas are not preserved, so paired Wilcoxon across '
             'variants cannot be computed. Use tbl:lora aggregate numbers.'),
    'variants': lora_raw,
}
(OUT / 'lora_stats.json').write_text(json.dumps(lora_out, indent=2))
print('\n[lora] per-fold unavailable; kept aggregate summary only')
print(f'\nWrote to {OUT}')
