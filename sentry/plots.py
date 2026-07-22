"""Publication-quality plots for SENTRY / PSID-8 (matplotlib only).

Produces the figures the paper needs, in the visual idiom Ultralytics users
expect (grid of loss/metric curves), plus two figures Ultralytics does not
provide: the epoch-ablation overfitting plot (train loss vs test F1 on twin
axes) and the event-level confusion matrix.

All functions save a PNG and return the path. Colours are colour-blind safe and
readable in greyscale.
"""
from __future__ import annotations

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from psid8.schema import CLASS_NAMES as CLASSES  # single source of truth; see psid8/schema.py

_C_MAIN = "#1f77b4"      # blue   - primary series
_C_ALT = "#d62728"       # red    - contrasting series
_C_REF = "#7f7f7f"       # grey   - reference lines


def _style(ax, title, xlabel, ylabel):
    ax.set_title(title, fontsize=10)
    ax.set_xlabel(xlabel, fontsize=9)
    ax.set_ylabel(ylabel, fontsize=9)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.tick_params(labelsize=8)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)


def plot_stageb_curves(history, out="stageB_curves.png"):
    """Ultralytics-style grid for Stage B.

    history: list of dicts, one per epoch, with keys "det" and "tc"
             (e.g. [{"epoch":0,"det":15.56,"tc":0.760}, ...]).
    """
    ep = [h["epoch"] for h in history]
    det = [h["det"] for h in history]
    tc = [h["tc"] for h in history]

    fig, axes = plt.subplots(1, 2, figsize=(9, 3.4))
    axes[0].plot(ep, det, "-o", color=_C_MAIN, ms=3.5, lw=1.6)
    _style(axes[0], "Stage B: supervised detection loss", "epoch", r"$L_{box}+L_{cls}+L_{dfl}$")
    axes[1].plot(ep, tc, "-o", color=_C_ALT, ms=3.5, lw=1.6)
    _style(axes[1], "Stage B: temporal consistency", "epoch", r"$L_{tc}$")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_epoch_ablation(rows, baseline_f1, out="epoch_ablation.png"):
    """THE overfitting figure: train loss falls while test F1 does not follow.

    rows: list of dicts with "epochs", "train_det", "test_f1", "fp"
          (the untrained-TFM row may omit "train_det").
    baseline_f1: frame-level reference (horizontal line).
    """
    r = [x for x in rows if x.get("train_det") is not None]
    ep = [x["epochs"] for x in r]
    det = [x["train_det"] for x in r]
    f1 = [x["test_f1"] for x in r]
    fp = [x["fp"] for x in r]

    fig, (ax1, ax3) = plt.subplots(1, 2, figsize=(9.5, 3.6))

    # left: train loss (down) vs test F1 (flat/low) on twin axes
    ax1.plot(ep, det, "-o", color=_C_MAIN, ms=5, lw=1.8, label="train detection loss")
    _style(ax1, "Training improves, test does not", "Stage B epochs", "train detection loss")
    ax1.set_ylabel("train detection loss", color=_C_MAIN, fontsize=9)
    ax1.tick_params(axis="y", labelcolor=_C_MAIN)

    ax2 = ax1.twinx()
    ax2.plot(ep, f1, "-s", color=_C_ALT, ms=5, lw=1.8, label="test event F1")
    ax2.axhline(baseline_f1, color=_C_REF, ls="--", lw=1.4)
    ax2.text(max(ep) * 0.98, baseline_f1 + 0.02, f"frame-level baseline ({baseline_f1:.3f})",
             ha="right", fontsize=8, color=_C_REF)
    ax2.set_ylabel("test event F1", color=_C_ALT, fontsize=9)
    ax2.tick_params(axis="y", labelcolor=_C_ALT, labelsize=8)
    ax2.set_ylim(0, max(baseline_f1 * 1.35, max(f1) * 1.35))
    for s in ("top",):
        ax2.spines[s].set_visible(False)

    # right: false positives explode
    ax3.bar([str(e) for e in ep], fp, color=_C_ALT, alpha=0.75, width=0.55)
    _style(ax3, "False positives on the test split", "Stage B epochs", "false positives")
    for i, v in enumerate(fp):
        ax3.text(i, v + max(fp) * 0.02, str(v), ha="center", fontsize=8)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_confusion_matrix(cm, class_names=None, out="confusion_matrix.png",
                          normalize=True, title="Event-level confusion matrix"):
    """Heatmap of the (n+1, n+1) matrix from metrics.event_confusion_matrix.

    The extra row/column is "background": last column = missed events,
    last row = false alarms. Only classes present in the matrix are drawn.
    """
    names = (class_names or CLASSES) + ["background"]
    cm = np.asarray(cm, dtype=float)
    keep = [i for i in range(cm.shape[0] - 1)
            if cm[i, :].sum() > 0 or cm[:, i].sum() > 0] + [cm.shape[0] - 1]
    sub = cm[np.ix_(keep, keep)]
    labels = [names[i] for i in keep]

    disp = sub.copy()
    if normalize:
        rs = disp.sum(axis=1, keepdims=True)
        disp = np.divide(disp, rs, out=np.zeros_like(disp), where=rs > 0)

    fig, ax = plt.subplots(figsize=(1.05 * len(labels) + 2.4, 1.0 * len(labels) + 2.0))
    im = ax.imshow(disp, cmap="Blues", vmin=0, vmax=disp.max() if disp.max() > 0 else 1)
    ax.set_xticks(range(len(labels))); ax.set_yticks(range(len(labels)))
    ax.set_xticklabels(labels, rotation=45, ha="right", fontsize=8)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel("predicted", fontsize=9)
    ax.set_ylabel("ground truth", fontsize=9)
    ax.set_title(title, fontsize=10)

    thr = (disp.max() or 1) / 2
    for i in range(len(labels)):
        for j in range(len(labels)):
            raw = int(sub[i, j])
            if raw == 0 and not (i == j):
                continue
            txt = f"{raw}" if not normalize else f"{disp[i, j]:.2f}\n({raw})"
            ax.text(j, i, txt, ha="center", va="center", fontsize=7.5,
                    color="white" if disp[i, j] > thr else "black")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04)
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_prompt_comparison(results, out="prompt_comparison.png"):
    """Level E: grouped bars of per-class F1 for each prompt condition, plus a
    panel with AUC and macro-F1.

    results: {condition: {"auc": float, "macro_f1": float,
                          "prf": {class_name: {"F1": ...}}}}
    """
    conds = list(results)
    classes = sorted({c for r in results.values() for c in r["prf"]})
    x = np.arange(len(classes))
    w = 0.8 / len(conds)
    colors = [_C_REF, _C_MAIN, _C_ALT][:len(conds)]

    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(11, 3.6),
                                   gridspec_kw={"width_ratios": [2.1, 1]})
    for k, cond in enumerate(conds):
        vals = [results[cond]["prf"].get(c, {}).get("F1", 0.0) for c in classes]
        ax1.bar(x + k * w - 0.4 + w / 2, vals, w, label=cond, color=colors[k], alpha=0.85)
    ax1.set_xticks(x); ax1.set_xticklabels(classes, rotation=30, ha="right", fontsize=8)
    _style(ax1, "Level E: per-class event F1 by prompt condition", "", "F1")
    ax1.legend(fontsize=8, frameon=False)

    idx = np.arange(len(conds))
    ax2.bar(idx - 0.18, [results[c]["auc"] for c in conds], 0.34,
            label="event AUC", color=_C_MAIN, alpha=0.85)
    ax2.bar(idx + 0.18, [results[c]["macro_f1"] for c in conds], 0.34,
            label="macro-F1", color=_C_ALT, alpha=0.85)
    ax2.set_xticks(idx); ax2.set_xticklabels(conds, rotation=20, ha="right", fontsize=8)
    _style(ax2, "Aggregate metrics", "", "value")
    ax2.legend(fontsize=8, frameon=False)

    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def plot_seed_variance(aggregate, metric="map50", out="seed_variance.png"):
    """Stage A: per-seed values with mean and bootstrap CI (multi-seed protocol)."""
    a = aggregate[metric]
    vals = a["per_seed"]
    seeds = aggregate.get("seeds", list(range(len(vals))))

    fig, ax = plt.subplots(figsize=(5, 3.2))
    ax.bar([str(s) for s in seeds], vals, color=_C_MAIN, alpha=0.8, width=0.55)
    ax.axhline(a["mean"], color=_C_ALT, ls="-", lw=1.6, label=f"mean = {a['mean']:.3f}")
    ax.axhspan(a["ci95"][0], a["ci95"][1], color=_C_ALT, alpha=0.12,
               label=f"95% CI [{a['ci95'][0]:.3f}, {a['ci95'][1]:.3f}]")
    for i, v in enumerate(vals):
        ax.text(i, v + 0.008, f"{v:.3f}", ha="center", fontsize=8)
    _style(ax, f"Stage A: {metric} across pre-registered seeds", "seed", metric)
    ax.set_ylim(0, max(vals) * 1.18)
    ax.legend(fontsize=8, frameon=False, loc="lower right")
    fig.tight_layout()
    fig.savefig(out, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return out


def parse_stageb_log(text):
    """Extract the per-epoch history from Stage B stdout lines of the form
    'epoch 3: det=14.660 | tc=0.5161'."""
    import re
    hist = []
    for m in re.finditer(r"epoch\s+(\d+):\s*det=([\d.]+)\s*\|\s*tc=([\d.]+)", text):
        hist.append({"epoch": int(m.group(1)), "det": float(m.group(2)),
                     "tc": float(m.group(3))})
    return hist
