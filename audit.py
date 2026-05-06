"""
audit.py — Forensic System Pattern Analyser
─────────────────────────────────────────────
You are a forensic AI system architect reviewing test results across multiple
images to find patterns in where the system fails.

Receives a batch of individually labelled test evaluations (each is an
analyze() result dict enriched with a ground_truth label) and produces a
structured diagnosis:

  • False-positive / false-negative breakdown per signal
  • Which signals are over-performing (causing FP noise) vs under-performing
    (missing real forgeries)
  • Concrete weight changes as specific numbers
  • Priority-ordered list of recommended fixes

Usage
─────
    from audit import audit_batch

    evaluations = [
        {"ground_truth": "AUTHENTIC",   **analyze(img1)},
        {"ground_truth": "MANIPULATED", **analyze(img2)},
        ...
    ]
    report = audit_batch(evaluations)

Input schema per evaluation dict
─────────────────────────────────
    ground_truth : "AUTHENTIC" | "MANIPULATED"   — human-verified label
    verdict      : dict  — from compute_verdict()
    ela          : dict  — from compute_ela()
    exif         : dict  — from extract_exif()
    noise        : dict  — from noise_consistency_map()
    fft          : dict  — from fft_spectrum_analysis()

Output schema (audit_batch return value)
─────────────────────────────────────────
    summary          : overall accuracy metrics
    error_analysis   : per-error-type breakdown (FP / FN)
    signal_diagnosis : per-signal over/under-performance assessment
    weight_recommendations : specific new weights to try
    priority_fixes   : ranked list of highest-impact single changes
    systemic_patterns: free-text paragraph on root causes
"""

from __future__ import annotations

import statistics
from typing import Any


# ── Current weights (mirrors analyzer.py — update both if you change them) ──
_CURRENT_WEIGHTS: dict[str, float] = {
    "ela_suspicious_pixels": 0.30,
    "ela_regional_variance":  0.15,
    "exif_risk":              0.30,
    "noise_suspicious_tiles": 0.15,
    "fft_high_freq_ratio":    0.10,
}

# ── Verdict thresholds (mirrors analyzer.py) ────────────────────────────────
_THRESH_AUTHENTIC   = 0.30
_THRESH_MANIPULATED = 0.55


# ─────────────────────────────────────────────────────────────────────────────
# Public API
# ─────────────────────────────────────────────────────────────────────────────

def audit_batch(evaluations: list[dict[str, Any]]) -> dict[str, Any]:
    """
    Analyse a batch of labelled test results and return a structured diagnosis.

    Parameters
    ----------
    evaluations : list of dicts, each containing:
        - ground_truth : "AUTHENTIC" | "MANIPULATED"
        - All keys returned by analyze() (ela, exif, noise, fft, verdict)

    Returns
    -------
    Structured audit report dict (see module docstring for schema).
    """
    if not evaluations:
        return {"error": "No evaluations provided."}

    # ── Separate cases by ground truth ────────────────────────────────────────
    authentic_cases   = [e for e in evaluations if _gt(e) == "AUTHENTIC"]
    manipulated_cases = [e for e in evaluations if _gt(e) == "MANIPULATED"]

    # ── Classify each prediction ──────────────────────────────────────────────
    true_pos  = [e for e in manipulated_cases if _pred(e) == "MANIPULATED"]
    false_neg = [e for e in manipulated_cases if _pred(e) != "MANIPULATED"]
    true_neg  = [e for e in authentic_cases   if _pred(e) == "AUTHENTIC"]
    false_pos = [e for e in authentic_cases   if _pred(e) != "AUTHENTIC"]

    n        = len(evaluations)
    accuracy = round((len(true_pos) + len(true_neg)) / n, 4) if n else 0
    precision = (
        round(len(true_pos) / (len(true_pos) + len(false_pos)), 4)
        if (true_pos or false_pos) else None
    )
    recall = (
        round(len(true_pos) / (len(true_pos) + len(false_neg)), 4)
        if (true_pos or false_neg) else None
    )

    summary = {
        "total_images":   n,
        "authentic":      len(authentic_cases),
        "manipulated":    len(manipulated_cases),
        "true_positives": len(true_pos),
        "false_positives": len(false_pos),
        "true_negatives": len(true_neg),
        "false_negatives": len(false_neg),
        "accuracy":       accuracy,
        "precision":      precision,
        "recall":         recall,
    }

    # ── Per-signal diagnosis ───────────────────────────────────────────────────
    signal_diag = _diagnose_signals(
        authentic_cases, manipulated_cases, false_pos, false_neg
    )

    # ── Weight recommendations ────────────────────────────────────────────────
    weight_recs = _recommend_weights(signal_diag)

    # ── Priority fix list ─────────────────────────────────────────────────────
    priority_fixes = _priority_fixes(signal_diag, false_pos, false_neg)

    # ── Systemic narrative ────────────────────────────────────────────────────
    narrative = _narrative(summary, signal_diag, false_pos, false_neg)

    return {
        "summary":              summary,
        "error_analysis":       _error_analysis(false_pos, false_neg),
        "signal_diagnosis":     signal_diag,
        "weight_recommendations": weight_recs,
        "priority_fixes":       priority_fixes,
        "systemic_patterns":    narrative,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Signal diagnosis
# ─────────────────────────────────────────────────────────────────────────────

def _diagnose_signals(
    authentic:   list[dict],
    manipulated: list[dict],
    false_pos:   list[dict],
    false_neg:   list[dict],
) -> dict[str, dict]:
    """
    For each scoring signal compute:
      - mean score on authentic vs manipulated cases
      - mean score on false positives (signal firing too much on clean images)
      - mean score on false negatives (signal not firing on real forgeries)
      - discrimination power (separation between authentic and manipulated)
      - assessment: "over-performing" | "under-performing" | "reliable" | "unreliable"
    """
    signals = list(_CURRENT_WEIGHTS.keys())
    diag: dict[str, dict] = {}

    for sig in signals:
        auth_scores  = [_component(e, sig) for e in authentic]
        manip_scores = [_component(e, sig) for e in manipulated]
        fp_scores    = [_component(e, sig) for e in false_pos]
        fn_scores    = [_component(e, sig) for e in false_neg]

        mean_auth  = _mean(auth_scores)
        mean_manip = _mean(manip_scores)
        mean_fp    = _mean(fp_scores)
        mean_fn    = _mean(fn_scores)
        separation = round(mean_manip - mean_auth, 4)

        # Over-performing: high score on authentic cases → driving false positives
        fp_contribution = mean_fp if false_pos else 0.0

        # Under-performing: low score on manipulated cases → driving false negatives
        fn_contribution = (1.0 - mean_fn) if false_neg else 0.0

        # Classify signal health
        if fp_contribution > 0.55 and mean_auth > 0.40:
            assessment = "over-performing"
            problem    = (
                f"Fires too strongly on clean images (mean score on FP cases: "
                f"{mean_fp:.3f}). Causes false positives."
            )
        elif fn_contribution > 0.55 and mean_manip < 0.45:
            assessment = "under-performing"
            problem    = (
                f"Fails to fire on manipulated images (mean score on FN cases: "
                f"{mean_fn:.3f}). Causes false negatives."
            )
        elif separation < 0.10 and _CURRENT_WEIGHTS[sig] >= 0.15:
            assessment = "unreliable"
            problem    = (
                f"Separation between authentic and manipulated is only {separation:.3f}. "
                f"Not discriminating well despite carrying {_CURRENT_WEIGHTS[sig]:.0%} weight."
            )
        else:
            assessment = "reliable"
            problem    = None

        diag[sig] = {
            "current_weight":        _CURRENT_WEIGHTS[sig],
            "mean_score_authentic":  mean_auth,
            "mean_score_manipulated": mean_manip,
            "mean_score_fp_cases":   mean_fp,
            "mean_score_fn_cases":   mean_fn,
            "separation":            separation,
            "assessment":            assessment,
            "problem":               problem,
        }

    return diag


# ─────────────────────────────────────────────────────────────────────────────
# Weight recommendations
# ─────────────────────────────────────────────────────────────────────────────

def _recommend_weights(signal_diag: dict[str, dict]) -> dict[str, Any]:
    """
    Produce specific new weight values based on signal diagnosis.

    Strategy
    ────────
    • over-performing  → reduce weight by 40% of current value
    • under-performing → increase weight by 33% of current value
    • unreliable       → reduce weight to 0.05 (minimal sentinel)
    • reliable         → keep current weight
    Then renormalise all weights so they sum to 1.0.
    """
    raw: dict[str, float] = {}
    rationale: dict[str, str] = {}

    for sig, info in signal_diag.items():
        cur = info["current_weight"]
        a   = info["assessment"]
        sep = info["separation"]

        if a == "over-performing":
            new = round(cur * 0.60, 3)
            rationale[sig] = (
                f"Reduced from {cur:.2f} → {new:.3f} (−40%) — signal fires too "
                f"strongly on clean images, driving false positives."
            )
        elif a == "under-performing":
            new = round(min(cur * 1.33, 0.45), 3)
            rationale[sig] = (
                f"Increased from {cur:.2f} → {new:.3f} (+33%) — signal misses "
                f"real forgeries; needs more influence on final score."
            )
        elif a == "unreliable":
            new = 0.05
            rationale[sig] = (
                f"Reduced from {cur:.2f} → 0.05 — separation of only "
                f"{sep:.3f} means this signal is near-random; capping at 0.05."
            )
        else:
            new = cur
            rationale[sig] = f"Kept at {cur:.2f} — signal is discriminating well."

        raw[sig] = new

    # Renormalise to sum = 1.0
    total = sum(raw.values())
    if total == 0:
        total = 1.0
    normalised = {sig: round(v / total, 4) for sig, v in raw.items()}

    # Adjust for floating-point rounding (make sure sum is exactly 1.0)
    diff = round(1.0 - sum(normalised.values()), 4)
    if diff:
        largest = max(normalised, key=normalised.__getitem__)
        normalised[largest] = round(normalised[largest] + diff, 4)

    return {
        "proposed_weights": normalised,
        "rationale":        rationale,
        "note": (
            "Weights are renormalised to sum to 1.0. "
            "Apply to WEIGHTS dict in analyzer.py and re-run the test suite."
        ),
    }


# ─────────────────────────────────────────────────────────────────────────────
# Priority fix list
# ─────────────────────────────────────────────────────────────────────────────

def _priority_fixes(
    signal_diag: dict[str, dict],
    false_pos:   list[dict],
    false_neg:   list[dict],
) -> list[dict[str, str]]:
    """
    Rank fixes by estimated impact (number of errors each fix would address).
    Returns a list of fix dicts sorted by impact descending.
    """
    fixes = []

    # ── False-positive fixes ──────────────────────────────────────────────────
    if false_pos:
        # Find the signal with the highest mean score on FP cases
        fp_driver = max(
            signal_diag.items(),
            key=lambda kv: kv[1]["mean_score_fp_cases"],
        )
        sig, info = fp_driver
        fixes.append({
            "rank":   "1",
            "impact": f"Resolves up to {len(false_pos)} false positive(s)",
            "type":   "FALSE POSITIVE",
            "action": (
                f"Reduce weight of '{sig}' from "
                f"{info['current_weight']:.2f} to "
                f"{round(info['current_weight'] * 0.60, 3):.3f}. "
                f"It is the primary driver of false alarms on clean images "
                f"(mean score on FP cases: {info['mean_score_fp_cases']:.3f})."
            ),
        })

    # ── False-negative fixes ──────────────────────────────────────────────────
    if false_neg:
        fn_driver = min(
            signal_diag.items(),
            key=lambda kv: kv[1]["mean_score_fn_cases"],
        )
        sig, info = fn_driver
        rank = str(len(fixes) + 1)
        fixes.append({
            "rank":   rank,
            "impact": f"Resolves up to {len(false_neg)} false negative(s)",
            "type":   "FALSE NEGATIVE",
            "action": (
                f"Increase weight of '{sig}' from "
                f"{info['current_weight']:.2f} to "
                f"{round(min(info['current_weight'] * 1.33, 0.45), 3):.3f}. "
                f"It is the weakest signal on missed forgeries "
                f"(mean score on FN cases: {info['mean_score_fn_cases']:.3f})."
            ),
        })

    # ── Threshold tuning ──────────────────────────────────────────────────────
    # If FP > FN, the MANIPULATED threshold is too low; raise it.
    # If FN > FP, the MANIPULATED threshold is too high; lower it.
    fp_n = len(false_pos)
    fn_n = len(false_neg)

    if fp_n > fn_n and fp_n > 0:
        rank = str(len(fixes) + 1)
        fixes.append({
            "rank":   rank,
            "impact": f"Would eliminate ~{fp_n - fn_n} false positive(s) without creating new false negatives",
            "type":   "THRESHOLD ADJUSTMENT",
            "action": (
                f"Raise the MANIPULATED decision threshold in analyzer.py from "
                f"{_THRESH_MANIPULATED} to {round(_THRESH_MANIPULATED + 0.05, 2)}. "
                f"FP count ({fp_n}) exceeds FN count ({fn_n}), indicating the "
                f"threshold is set too aggressively low."
            ),
        })
    elif fn_n > fp_n and fn_n > 0:
        rank = str(len(fixes) + 1)
        fixes.append({
            "rank":   rank,
            "impact": f"Would catch ~{fn_n - fp_n} additional forgeries",
            "type":   "THRESHOLD ADJUSTMENT",
            "action": (
                f"Lower the MANIPULATED decision threshold in analyzer.py from "
                f"{_THRESH_MANIPULATED} to {round(_THRESH_MANIPULATED - 0.05, 2)}. "
                f"FN count ({fn_n}) exceeds FP count ({fp_n}), indicating the "
                f"threshold is too conservative for this document population."
            ),
        })

    # ── Unreliable signal removal ─────────────────────────────────────────────
    unreliable = [
        (sig, info) for sig, info in signal_diag.items()
        if info["assessment"] == "unreliable"
    ]
    for sig, info in unreliable:
        rank = str(len(fixes) + 1)
        fixes.append({
            "rank":   rank,
            "impact": f"Removes noise from scoring; redistributes {info['current_weight']:.0%} weight to better signals",
            "type":   "SIGNAL REMOVAL",
            "action": (
                f"Set weight of '{sig}' to 0.05 in analyzer.py. "
                f"Separation score of {info['separation']:.3f} confirms it "
                f"cannot reliably distinguish authentic from manipulated documents."
            ),
        })

    return fixes


# ─────────────────────────────────────────────────────────────────────────────
# Error analysis breakdown
# ─────────────────────────────────────────────────────────────────────────────

def _error_analysis(
    false_pos: list[dict],
    false_neg: list[dict],
) -> dict[str, Any]:
    """
    Summarise what the false positives and false negatives look like in terms
    of which signals fired / failed to fire on them.
    """
    def _common_signals(cases: list[dict], threshold: float = 0.5) -> dict[str, int]:
        """Count how many cases have each signal above threshold."""
        counts: dict[str, int] = {sig: 0 for sig in _CURRENT_WEIGHTS}
        for case in cases:
            for sig in _CURRENT_WEIGHTS:
                if _component(case, sig) >= threshold:
                    counts[sig] += 1
        return {k: v for k, v in sorted(counts.items(), key=lambda x: -x[1]) if v > 0}

    fp_analysis: dict[str, Any] = {}
    fn_analysis: dict[str, Any] = {}

    if false_pos:
        fp_scores = {
            sig: _mean([_component(e, sig) for e in false_pos])
            for sig in _CURRENT_WEIGHTS
        }
        fp_analysis = {
            "count": len(false_pos),
            "description": (
                "Clean documents incorrectly flagged as manipulated. "
                "These cause unnecessary customer friction and analyst workload."
            ),
            "mean_signal_scores": {k: round(v, 4) for k, v in fp_scores.items()},
            "signals_firing_above_0_5": _common_signals(false_pos, 0.5),
            "predicted_verdicts": [_pred(e) for e in false_pos],
            "forgery_scores":     [round(_fscore(e), 4) for e in false_pos],
        }

    if false_neg:
        fn_scores = {
            sig: _mean([_component(e, sig) for e in false_neg])
            for sig in _CURRENT_WEIGHTS
        }
        fn_analysis = {
            "count": len(false_neg),
            "description": (
                "Manipulated documents incorrectly cleared as authentic. "
                "These are the highest-risk failures — a genuine forgery reaches a customer file."
            ),
            "mean_signal_scores": {k: round(v, 4) for k, v in fn_scores.items()},
            "signals_firing_above_0_5": _common_signals(false_neg, 0.5),
            "predicted_verdicts": [_pred(e) for e in false_neg],
            "forgery_scores":     [round(_fscore(e), 4) for e in false_neg],
        }

    return {
        "false_positives": fp_analysis,
        "false_negatives": fn_analysis,
    }


# ─────────────────────────────────────────────────────────────────────────────
# Narrative summary
# ─────────────────────────────────────────────────────────────────────────────

def _narrative(
    summary:    dict,
    signal_diag: dict[str, dict],
    false_pos:  list[dict],
    false_neg:  list[dict],
) -> str:
    """Generate a free-text paragraph identifying the root systemic patterns."""
    parts = []
    n     = summary["total_images"]

    parts.append(
        f"Across {n} test images, the system achieved {summary['accuracy']:.1%} accuracy "
        f"({summary['true_positives']} TP, {summary['true_negatives']} TN, "
        f"{summary['false_positives']} FP, {summary['false_negatives']} FN)."
    )

    over  = [s for s, d in signal_diag.items() if d["assessment"] == "over-performing"]
    under = [s for s, d in signal_diag.items() if d["assessment"] == "under-performing"]
    unrel = [s for s, d in signal_diag.items() if d["assessment"] == "unreliable"]

    if over:
        sigs = ", ".join(f"'{s}'" for s in over)
        parts.append(
            f"Over-performing signals ({sigs}) are the primary root cause of false positives — "
            f"they assign high scores to genuine documents, pushing borderline cases past the "
            f"{_THRESH_MANIPULATED} decision threshold. Reducing their weights is the highest-priority fix."
        )

    if under:
        sigs = ", ".join(f"'{s}'" for s in under)
        parts.append(
            f"Under-performing signals ({sigs}) are systematically missing real forgeries — "
            f"their scores remain low even when ground truth is MANIPULATED. "
            f"This is a false-negative risk. Either increase their weights or investigate "
            f"whether their detection thresholds in the respective core modules need tuning."
        )

    if unrel:
        sigs = ", ".join(f"'{s}'" for s in unrel)
        parts.append(
            f"Signals ({sigs}) show near-zero discrimination power (separation < 0.10) "
            f"despite carrying significant weight. They are adding noise to the scoring without "
            f"contributing predictive value and should be reduced to a minimal weight of 0.05."
        )

    if not over and not under and not unrel:
        parts.append(
            "No signals are consistently mis-calibrated. Failures appear to be case-specific "
            "rather than systemic. Consider reviewing the individual failure cases for "
            "document-type or quality patterns not captured by the current signal set."
        )

    if summary["false_positives"] > summary["false_negatives"]:
        parts.append(
            f"The system is biased toward over-detection: {summary['false_positives']} FP vs "
            f"{summary['false_negatives']} FN. For an NBFC KYC context this means unnecessary "
            f"customer rejections. Raising the MANIPULATED threshold from {_THRESH_MANIPULATED} "
            f"to {round(_THRESH_MANIPULATED + 0.05, 2)} is recommended alongside the weight changes."
        )
    elif summary["false_negatives"] > summary["false_positives"]:
        parts.append(
            f"The system is biased toward under-detection: {summary['false_negatives']} FN vs "
            f"{summary['false_positives']} FP. For an NBFC KYC context this is the more dangerous "
            f"failure mode — forged documents are passing through. Lowering the MANIPULATED "
            f"threshold from {_THRESH_MANIPULATED} to {round(_THRESH_MANIPULATED - 0.05, 2)} "
            f"is recommended alongside the weight changes."
        )

    return " ".join(parts)


# ─────────────────────────────────────────────────────────────────────────────
# Helpers
# ─────────────────────────────────────────────────────────────────────────────

def _gt(e: dict) -> str:
    """Normalise ground truth label."""
    return str(e.get("ground_truth", "")).upper().strip()


def _pred(e: dict) -> str:
    """Extract the system's predicted verdict label."""
    return str(e.get("verdict", {}).get("verdict", "UNKNOWN")).upper().strip()


def _fscore(e: dict) -> float:
    """Extract the forgery score from an evaluation."""
    return float(e.get("verdict", {}).get("forgery_score", 0.0))


def _component(e: dict, signal: str) -> float:
    """Extract the normalised component score for a given signal."""
    comp = e.get("verdict", {}).get("component_scores", {})
    return float(comp.get(signal, 0.0))


def _mean(values: list[float]) -> float:
    """Safe mean that returns 0.0 for empty lists."""
    return round(statistics.mean(values), 4) if values else 0.0
