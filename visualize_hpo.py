#!/usr/bin/env python3
"""
Make paper-ready figures from an Optuna study (SQLite).

Outputs (in --outdir, default: hpo_viz):
  - fig1_history.png          : Optimization history (semilogy best-so-far) + scatter of completed trials
  - fig2_importance.png       : Hyperparameter importance (fANOVA if available, else Spearman |rho|)
  - fig3_parallel.png         : Parallel coordinates of top-k important params (categoricals encoded)
  - trials.csv                : Flat table of all COMPLETED trials with params and value
  - param_encodings.json      : Mapping of categorical values -> integer codes (used in parallel coords)

Usage:
  python visualize_hpo.py --db sqlite:///lx_optuna.db --study LX-PINN-HPO --outdir hpo_viz
"""

from __future__ import annotations
import argparse
import json
import math
import os
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import optuna
from optuna.trial import TrialState

# ---- optional: fANOVA (best), else fall back to Spearman correlation ----
_HAS_FANOVA = True
try:
    from optuna.importance import get_param_importances, FanovaImportanceEvaluator
except Exception:
    _HAS_FANOVA = False

# ---- optional: Spearman correlation ----
_HAS_SCIPY = True
try:
    from scipy.stats import spearmanr
except Exception:
    _HAS_SCIPY = False


def _load_study(storage: str, study_name: str) -> optuna.study.Study:
    return optuna.load_study(study_name=study_name, storage=storage)


def _df_from_trials(study: optuna.study.Study) -> pd.DataFrame:
    trials = [t for t in study.get_trials(deepcopy=False)
              if t.state == TrialState.COMPLETE and np.isfinite(t.value)]
    if not trials:
        raise RuntimeError("No COMPLETED (finite) trials found.")
    rows = []
    for t in trials:
        row = dict(number=t.number, value=float(t.value))
        # Flatten params; keep raw types (categoricals may be str/bool)
        for k, v in t.params.items():
            row[f"param:{k}"] = v
        # If you logged user attrs (e.g., durations), you can add them here:
        # for k, v in t.user_attrs.items():
        #     row[f"attr:{k}"] = v
        rows.append(row)
    df = pd.DataFrame(rows).sort_values("number").reset_index(drop=True)
    return df


def _make_history_plot(df: pd.DataFrame, out: Path, title: str):
    """
    Fig 1: Optimization History
    - scatter of (trial number, value)
    - best-so-far curve (semilogy)
    """
    x = df["number"].to_numpy()
    y = df["value"].to_numpy().astype(float)

    best = np.minimum.accumulate(y)

    fig = plt.figure(figsize=(7.0, 4.2), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # Scatter all trial outcomes
    ax.scatter(x, y, s=18, alpha=0.55, label="Completed trial")

    # Best-so-far line (semilogy look via yscale, but retain scatter on linear to match journal preferences)
    ax.plot(x, best, linewidth=2.5, label="Best so far")

    # If values are positive, show semilogy; otherwise fall back to linear
    if np.all(best > 0):
        ax.set_yscale("log")
        ax.set_ylabel("Objective (log scale)")
    else:
        ax.set_ylabel("Objective")

    ax.set_xlabel("Trial number")
    ax.set_title(title)
    ax.grid(True, alpha=0.25)
    # Annotation with best value
    best_idx = int(np.argmin(y))
    ax.annotate(
        f"Best={y[best_idx]:.3e} (trial {int(x[best_idx])})",
        xy=(x[best_idx], y[best_idx]),
        xytext=(5, 5),
        textcoords="offset points",
        fontsize=9,
        bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="0.6", alpha=0.9)
    )
    ax.legend()
    fig.savefig(out / "fig1_history.png", dpi=300)
    plt.close(fig)


def _safe_to_numeric(col: pd.Series) -> Tuple[pd.Series, Dict[str, Dict[str, int]]]:
    """
    Convert a param column to numeric:
      - numeric stays numeric
      - bool -> {False:0, True:1}
      - str/category -> codes with mapping
    Returns encoded series and mapping dict { 'mapping': {value: code} } or empty if NA.
    """
    mapping: Dict[str, Dict[str, int]] = {}
    if pd.api.types.is_numeric_dtype(col):
        return col.astype(float), mapping
    if pd.api.types.is_bool_dtype(col):
        enc = col.astype(int).astype(float)
        mapping["mapping"] = {"False": 0, "True": 1}
        return enc, mapping

    # strings / categoricals
    cat = pd.Categorical(col.astype("string"))
    codes = pd.Series(cat.codes, index=col.index).astype(float)
    value_map = {str(v): int(i) for i, v in enumerate(cat.categories)}
    mapping["mapping"] = value_map
    return codes, mapping


def _compute_importance(df: pd.DataFrame, study: optuna.study.Study) -> pd.Series:
    """
    Try fANOVA; fall back to Spearman |rho|; if SciPy absent, use rank corr via numpy.
    Returns a pd.Series indexed by param name with importance scores in [0, 1] (normalized).
    """
    # Collect param names
    pcols = [c for c in df.columns if c.startswith("param:")]
    if not pcols:
        raise RuntimeError("No parameters found in trials table.")

    # Preferred: fANOVA (captures interactions)
    if _HAS_FANOVA:
        try:
            imp = get_param_importances(study, evaluator=FanovaImportanceEvaluator(seed=42))
            s = pd.Series(imp, dtype=float)  # already normalized to sum 1.0
            s.index = [f"param:{k}" for k in s.index]  # match df column naming
            # Some params may be missing if constant; reindex with zeros
            s = s.reindex(pcols).fillna(0.0)
            return s
        except Exception:
            pass

    # Fallback: Spearman |rho| (univariate; still very useful)
    y = df["value"].to_numpy().astype(float)
    vals = []
    for c in pcols:
        x_raw = df[c]
        x_enc, _ = _safe_to_numeric(x_raw)
        x = x_enc.to_numpy()
        if _HAS_SCIPY:
            rho, _ = spearmanr(x, y)
            score = 0.0 if np.isnan(rho) else abs(float(rho))
        else:
            # Simple rank-corr approx via numpy corrcoef on ranks
            xr = pd.Series(x).rank(method="average").to_numpy()
            yr = pd.Series(y).rank(method="average").to_numpy()
            cc = np.corrcoef(xr, yr)[0, 1]
            score = abs(float(cc)) if np.isfinite(cc) else 0.0
        vals.append(score)

    s = pd.Series(vals, index=pcols, dtype=float)
    # Normalize to sum 1.0 for visual comparability; guard all-zero
    denom = s.sum()
    return (s / denom) if denom > 0 else s


def _make_importance_plot(importance: pd.Series, out: Path, title: str, top_k: int = 15):
    """
    Fig 2: Horizontal bars of param importance (top_k).
    """
    s = importance.sort_values(ascending=True)
    if len(s) > top_k:
        s = s.iloc[-top_k:]
    labels = [c.replace("param:", "") for c in s.index]

    fig = plt.figure(figsize=(7.0, 4.6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)
    ax.barh(np.arange(len(s)), s.values)
    ax.set_yticks(np.arange(len(s)))
    ax.set_yticklabels(labels, fontsize=9)
    ax.set_xlabel("Importance (normalized)")
    ax.set_title(title)
    ax.grid(axis="x", alpha=0.25)
    fig.savefig(out / "fig2_importance.png", dpi=300)
    plt.close(fig)


def _normalize_for_parallel(df_num: pd.DataFrame) -> pd.DataFrame:
    # Min-max to [0,1]; if constant, set 0.5
    normed = df_num.copy()
    for c in normed.columns:
        v = normed[c].astype(float).to_numpy()
        vmin, vmax = np.nanmin(v), np.nanmax(v)
        if not np.isfinite(vmin) or not np.isfinite(vmax):
            normed[c] = 0.5
            continue
        if vmax <= vmin + 1e-15:
            normed[c] = 0.5
        else:
            normed[c] = (v - vmin) / (vmax - vmin)
    return normed


def _make_parallel_plot(
    df: pd.DataFrame,
    importance: pd.Series,
    out: Path,
    title: str,
    max_params: int = 6,
    max_lines: int = 600,
):
    """
    Fig 3: Parallel coordinates for top 'max_params' params (by importance).
    - Categorical params get encoded; a JSON mapping is saved.
    - Lines are colored by objective (log-scale if strictly positive).
    """
    # pick top params
    top = importance.sort_values(ascending=False).index.tolist()[:max_params]
    if not top:
        raise RuntimeError("No parameters available for parallel plot.")
    # Build numeric param frame + encodings
    encodings: Dict[str, Dict[str, Dict[str, int]]] = {}
    cols = []
    for c in top:
        x_enc, mapping = _safe_to_numeric(df[c])
        cols.append(x_enc.rename(c))
        if mapping:
            encodings[c.replace("param:", "")] = mapping
    P = pd.concat(cols, axis=1)

    # Add objective column (for color)
    y = df["value"].astype(float).rename("objective")
    data = pd.concat([P, y], axis=1)

    # Normalize params to [0,1] for plotting; keep a copy of raw objective
    Pnorm = _normalize_for_parallel(P)
    data_norm = pd.concat([Pnorm, y], axis=1)

    # Subsample for readability (keep best N/2 and a random subset of the rest)
    N = len(data_norm)
    if N > max_lines:
        # keep ~half best (by objective) + random of remainder
        half = max_lines // 2
        idx_best = data_norm.nsmallest(half, "objective").index
        remainder = data_norm.drop(index=idx_best)
        rng = np.random.default_rng(42)
        idx_rand = remainder.index.to_series().sample(n=max(0, max_lines - half), random_state=42)
        data_norm = data_norm.loc[pd.Index(idx_best.tolist() + idx_rand.tolist())]

    # Color by objective (log scale if >0)
    vals = data_norm["objective"].to_numpy()
    positive = np.all(vals > 0)
    if positive:
        cvals = np.log10(vals)
        cbar_label = "log10(objective)"
    else:
        cvals = vals
        cbar_label = "objective"

    # Plot
    fig = plt.figure(figsize=(1.4 + 1.4*len(top), 4.6), constrained_layout=True)
    ax = fig.add_subplot(1, 1, 1)

    # Draw one polyline per trial
    xs = np.arange(len(top))
    for i, (_, row) in enumerate(data_norm.iterrows()):
        ys = row[top].to_numpy(dtype=float)
        ax.plot(xs, ys, alpha=0.18, linewidth=1.0, zorder=1, color=plt.cm.viridis(
            (cvals[i] - np.nanmin(cvals)) / (np.nanmax(cvals) - np.nanmin(cvals) + 1e-12)
        ))

    # Axes cosmetics
    ax.set_xlim(xs.min(), xs.max())
    ax.set_xticks(xs)
    ax.set_xticklabels([t.replace("param:", "") for t in top], rotation=20, ha="right")
    ax.set_yticks([0.0, 0.5, 1.0])
    ax.set_yticklabels(["min", "mid", "max"])
    ax.set_title(title)
    ax.grid(alpha=0.25)

    # Colorbar
    sm = plt.cm.ScalarMappable(cmap="viridis")
    sm.set_array(cvals)
    cb = fig.colorbar(sm, ax=ax)
    cb.set_label(cbar_label)

    fig.savefig(out / "fig3_parallel.png", dpi=300)
    plt.close(fig)

    # Save encodings used (for reproducibility in the paper)
    with open(out / "param_encodings.json", "w") as f:
        json.dump(encodings, f, indent=2)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--db", type=str, default="sqlite:///lx_optuna.db",
                    help="Optuna storage URI, e.g., sqlite:///lx_optuna.db")
    ap.add_argument("--study", type=str, default="LX-PINN-HPO",
                    help="Study name used in _hyper_opt.py")
    ap.add_argument("--outdir", type=str, default="hpo_viz",
                    help="Output folder for figures and CSV")
    ap.add_argument("--parallel-k", type=int, default=6,
                    help="Number of top parameters in parallel-coordinates figure")
    ap.add_argument("--max-lines", type=int, default=600,
                    help="Max polylines to draw in the parallel-coordinates figure")
    args = ap.parse_args()

    out = Path(args.outdir)
    out.mkdir(parents=True, exist_ok=True)

    study = _load_study(args.db, args.study)
    df = _df_from_trials(study)

    # Separate param columns for convenience
    param_cols = [c for c in df.columns if c.startswith("param:")]
    if not param_cols:
        raise RuntimeError("No parameters found. Did your trials have search space?")

    # Save table for the paper / appendix
    df_out = df.copy()
    # Strip 'param:' prefix in CSV to be reader-friendly
    df_out.rename(columns={c: c.replace("param:", "") for c in param_cols}, inplace=True)
    df_out.to_csv(out / "trials.csv", index=False)

    # === Fig 1: Optimization history ===
    _make_history_plot(df, out, title="Hyperparameter Optimization History")

    # === Fig 2: Importance ===
    imp = _compute_importance(df, study)
    # Normalize again just in case fallback path was zero-sum
    s = imp.copy()
    s = s / s.sum() if s.sum() > 0 else s
    _make_importance_plot(s, out, title="Hyperparameter Importance (fANOVA or Spearman |ρ|)", top_k=15)

    # === Fig 3: Parallel coordinates (top-k by importance) ===
    _make_parallel_plot(df, s, out, title="Top Hyperparameters (Parallel Coordinates)",
                        max_params=args.parallel_k, max_lines=args.max_lines)

    print(f"[OK] Wrote figures and data to: {out.resolve()}")
    print("  - fig1_history.png")
    print("  - fig2_importance.png")
    print("  - fig3_parallel.png")
    print("  - trials.csv")
    print("  - param_encodings.json")


if __name__ == "__main__":
    main()
