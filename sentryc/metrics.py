"""Metrics for SENTRY-C.

Every metric that already exists for the physical pipeline is IMPORTED from
`sentry.metrics`, never reimplemented: the temporal-IoU matching, the event
AUC/P-R-F1 and the bootstrap CI are the same machinery, so convergent alerts and
unimodal alerts are scored on identical terms. Duplicating that logic would let
the two drift apart and quietly invalidate every comparison in the thesis.

Three metrics are new, because convergence raises questions the unimodal
pipeline never had to answer:

  * lead time — does fusing detect the incident EARLIER than either modality
    alone? (thesis H3: lead time)
  * false-positive reduction — does fusing suppress the false alarms that each
    modality raises on its own?
  * modality contribution — of the true detections, how many were reachable by
    video alone, by network alone, and only by their conjunction? This is the
    table that shows convergence is doing real work rather than riding on one
    strong modality.
"""
from __future__ import annotations

from typing import Iterable, Sequence

import numpy as np

# Reuse, do not reimplement.
from sentry.metrics import (tiou, match_events, event_auc, event_prf,
                            bootstrap_ci, event_confusion_matrix)

from .alerts import ConvergentAlert

__all__ = [
    "tiou", "match_events", "event_auc", "event_prf", "bootstrap_ci",
    "event_confusion_matrix",
    "convergent_event_prf", "convergent_event_auc",
    "lead_time_distribution", "lead_time_gain",
    "false_positive_reduction", "modality_contribution",
]


def _as_events(alerts: Sequence[ConvergentAlert], class_id_of: dict[str, int]) -> list[dict]:
    """ConvergentAlert -> the event dict that sentry.metrics understands."""
    return [a.as_event(class_id_of[a.cls]) for a in alerts]


def convergent_event_prf(alerts: Sequence[ConvergentAlert], gts: Sequence[dict],
                         class_id_of: dict[str, int], tiou_thr: float = 0.5,
                         n_classes: int = 8):
    """Per-class P/R/F1 of convergent alerts against event ground truth."""
    return event_prf(_as_events(alerts, class_id_of), list(gts), tiou_thr, n_classes)


def convergent_event_auc(alerts: Sequence[ConvergentAlert], gts: Sequence[dict],
                         class_id_of: dict[str, int], tiou_thr: float = 0.5) -> float:
    return event_auc(_as_events(alerts, class_id_of), list(gts), tiou_thr)


# --------------------------------------------------------------------------
# Lead time (H3)
# --------------------------------------------------------------------------
def lead_time_distribution(alerts: Sequence[ConvergentAlert],
                           gts: Sequence[dict],
                           class_id_of: dict[str, int],
                           tiou_thr: float = 0.5) -> list[float]:
    """`time_to_detection` of the alerts that actually matched a real incident.

    Restricting to TRUE positives is deliberate: the lead time of a false alarm
    is meaningless, and averaging it in would let a noisy detector look fast.
    """
    ev = _as_events(alerts, class_id_of)
    pairs, _, _ = match_events(ev, list(gts), tiou_thr)
    return [float(alerts[pi].time_to_detection) for pi, _, _ in pairs]


def lead_time_gain(convergent: Sequence[ConvergentAlert],
                   unimodal: Sequence[ConvergentAlert],
                   gts: Sequence[dict],
                   class_id_of: dict[str, int],
                   tiou_thr: float = 0.5,
                   n_boot: int = 1000) -> dict:
    """Paired comparison of detection latency, convergent vs. a unimodal detector.

    Both sides are matched to the SAME ground-truth events; only incidents both
    detected enter the comparison, so the difference measures latency and not a
    difference in recall (which `convergent_event_prf` reports separately).

    A NEGATIVE mean difference means the convergent detector fires EARLIER — it
    is the direction the thesis predicts.
    """
    ev_c = _as_events(convergent, class_id_of)
    ev_u = _as_events(unimodal, class_id_of)
    pc, _, _ = match_events(ev_c, list(gts), tiou_thr)
    pu, _, _ = match_events(ev_u, list(gts), tiou_thr)
    tc = {gi: convergent[pi].time_to_detection for pi, gi, _ in pc}
    tu = {gi: unimodal[pi].time_to_detection for pi, gi, _ in pu}
    common = sorted(set(tc) & set(tu))
    if not common:
        return {"n_paired": 0, "mean_diff": float("nan"), "ci95": [float("nan")] * 2,
                "convergent_mean": float("nan"), "unimodal_mean": float("nan")}
    diffs = [tc[g] - tu[g] for g in common]
    mean, lo, hi = bootstrap_ci(diffs, n_boot=n_boot)
    return {"n_paired": len(common), "mean_diff": mean, "ci95": [lo, hi],
            "convergent_mean": float(np.mean([tc[g] for g in common])),
            "unimodal_mean": float(np.mean([tu[g] for g in common])),
            "earlier_fraction": float(np.mean([d < 0 for d in diffs]))}


# --------------------------------------------------------------------------
# False-positive reduction
# --------------------------------------------------------------------------
def false_positive_reduction(convergent: Sequence[dict],
                             unimodal: Sequence[dict],
                             gts: Sequence[dict],
                             tiou_thr: float = 0.5) -> dict:
    """False alarms of the convergent detector vs. a unimodal one.

    Both argument lists are event dicts (use `_as_events` or pass the unimodal
    pipeline's alert records directly). Reduction is reported as a fraction of
    the unimodal false-alarm count; `nan` when the unimodal detector raised none.
    """
    _, fp_c, _ = match_events(list(convergent), list(gts), tiou_thr)
    _, fp_u, _ = match_events(list(unimodal), list(gts), tiou_thr)
    n_c, n_u = len(fp_c), len(fp_u)
    red = (n_u - n_c) / n_u if n_u else float("nan")
    return {"fp_convergent": n_c, "fp_unimodal": n_u, "reduction": float(red)}


# --------------------------------------------------------------------------
# Modality contribution
# --------------------------------------------------------------------------
def modality_contribution(convergent: Sequence[ConvergentAlert],
                          gts: Sequence[dict],
                          class_id_of: dict[str, int],
                          tiou_thr: float = 0.5) -> dict:
    """Of the incidents the convergent detector got right, how were they reached?

    physical_only : the matched alert carried no network evidence
    network_only  : it carried no physical evidence beyond the seed's own span
                    (i.e. the seed confidence was below the physical threshold
                    and the network stream carried the decision)
    joint         : both modalities contributed

    A convergence paper that cannot show a non-trivial `joint` count is riding on
    one modality; this metric makes that visible instead of hiding it inside F1.
    """
    ev = _as_events(convergent, class_id_of)
    pairs, _, _ = match_events(ev, list(gts), tiou_thr)
    phys_only = net_only = joint = 0
    for pi, _, _ in pairs:
        a = convergent[pi]
        has_net = len(a.network_evidence) > 0
        has_phys = len(a.physical_evidence) > 0
        if has_net and has_phys:
            joint += 1
        elif has_phys:
            phys_only += 1
        elif has_net:
            net_only += 1
    total = joint + phys_only + net_only
    return {"true_positives": total, "joint": joint,
            "physical_only": phys_only, "network_only": net_only,
            "joint_fraction": (joint / total) if total else float("nan")}
