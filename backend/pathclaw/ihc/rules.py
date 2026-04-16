"""Rule registry for IHC scoring presets.

A `Rule` captures everything the scorer needs for one marker:
    - which cellular compartment the stain lives in (nuclear / membrane / cytoplasm)
    - how to threshold DAB intensity (single cutoff, or bands for HER2)
    - how to aggregate per-cell / per-patch measurements to one slide number
    - a human label + clinical interpretation for the final score

Dynamic rules: callers pass `rule_override={...}` to any scoring entrypoint.
The override follows this dataclass's schema; unknown fields are ignored.

Add a new preset without editing this file by calling `register_rule(name,
Rule(...))` from your code — e.g. in a plugin's `__init__.py`.
"""
from __future__ import annotations

from dataclasses import dataclass, field, asdict
from typing import Any, Callable


@dataclass
class Rule:
    name: str
    marker: str                      # "ki67", "er", "pr", "her2", "pdl1", "ck7", ...
    compartment: str = "nuclear"     # nuclear | membrane | cytoplasm
    # DAB intensity thresholds. Single value = binary +/-; list = multi-band (HER2 0/1+/2+/3+).
    dab_threshold: float | list[float] = 0.15
    # Aggregation across patches/cells → slide-level number.
    # percent_positive : fraction of positive cells (0-100)
    # tps              : tumor proportion score (same math, but on tumor-masked area)
    # allred           : proportion score 0-5 + intensity score 0-3, summed (0-8)
    # her2_score       : 0/1+/2+/3+ from banded membrane DAB + completeness
    # mean_intensity   : continuous mean DAB intensity
    aggregation: str = "percent_positive"
    # Minimum fraction of analyzed area required for a valid score (tissue gate).
    min_tissue_fraction: float = 0.05
    # How many patches to sample per slide (random-tissue, 512x512 @ 20x).
    patches_per_slide: int = 200
    patch_size: int = 512
    target_mpp: float = 0.5          # ~20x for most H&E/IHC scanners
    # Human-friendly interpretation table: score → label.
    # Signature: (score: float) -> str.
    interpret: Callable[[float], str] | None = None
    notes: str = ""

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d["interpret"] = None  # callable not JSON-serialisable; reconstructed in the registry
        return d


# ---------------------------------------------------------------------------
# Interpretation helpers
# ---------------------------------------------------------------------------

def _ki67_interpret(pi: float) -> str:
    if pi < 5:   return "low proliferative index"
    if pi < 20:  return "intermediate proliferative index"
    return "high proliferative index"


def _allred_interpret(score: float) -> str:
    if score <= 2: return "negative"
    if score <= 5: return "weakly positive"
    return "strongly positive"


def _her2_interpret(score: float) -> str:
    # score is the banded 0/1/2/3 value
    s = int(round(score))
    return {0: "negative (0)", 1: "equivocal (1+)", 2: "equivocal (2+) — reflex to ISH",
            3: "positive (3+)"}.get(s, f"score={score:.2f}")


def _pdl1_interpret(tps: float) -> str:
    if tps < 1:   return "negative (<1%)"
    if tps < 50:  return "low (1–49%)"
    return "high (≥50%)"


# ---------------------------------------------------------------------------
# Built-in presets
# ---------------------------------------------------------------------------

REGISTRY: dict[str, Rule] = {
    "ki67_pi": Rule(
        name="ki67_pi",
        marker="ki67",
        compartment="nuclear",
        dab_threshold=0.15,
        aggregation="percent_positive",
        interpret=_ki67_interpret,
        notes="Nuclear Ki-67 proliferation index (%). Dynamic threshold can be lowered for weak antibodies.",
    ),
    "er_allred": Rule(
        name="er_allred",
        marker="er",
        compartment="nuclear",
        dab_threshold=[0.08, 0.20, 0.35],   # weak / moderate / strong → intensity 1/2/3
        aggregation="allred",
        interpret=_allred_interpret,
        notes="ER Allred 0-8. Proportion score 0-5 + intensity score 0-3.",
    ),
    "pr_allred": Rule(
        name="pr_allred",
        marker="pr",
        compartment="nuclear",
        dab_threshold=[0.08, 0.20, 0.35],
        aggregation="allred",
        interpret=_allred_interpret,
        notes="PR Allred 0-8. Same thresholds as ER; tumor-region mask optional.",
    ),
    "her2_membrane": Rule(
        name="her2_membrane",
        marker="her2",
        compartment="membrane",
        dab_threshold=[0.10, 0.20, 0.35],   # 1+ / 2+ / 3+ DAB bands
        aggregation="her2_score",
        interpret=_her2_interpret,
        notes="HER2 membrane score 0/1+/2+/3+. Completeness fraction decides 2+ vs 3+.",
    ),
    "pdl1_tps": Rule(
        name="pdl1_tps",
        marker="pdl1",
        compartment="membrane",
        dab_threshold=0.15,
        aggregation="tps",
        interpret=_pdl1_interpret,
        notes="PD-L1 Tumor Proportion Score. Needs tumor mask for clinical use; falls back to whole tissue.",
    ),
}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def register_rule(name: str, rule: Rule) -> None:
    REGISTRY[name] = rule


def get_rule(name: str, override: dict | None = None) -> Rule:
    """Fetch a preset and optionally patch fields from `override`.

    Unknown fields in `override` are silently dropped — lets callers pass a
    JSON config without having to strip extras.
    """
    if name not in REGISTRY:
        raise KeyError(f"Unknown IHC rule: {name!r}. Known: {sorted(REGISTRY)}")
    base = REGISTRY[name]
    if not override:
        return base
    allowed = {f for f in base.__dataclass_fields__}
    patched = {k: v for k, v in override.items() if k in allowed}
    return Rule(**{**asdict(base), **patched, "interpret": base.interpret})


def list_rules() -> list[dict]:
    return [r.to_dict() for r in REGISTRY.values()]
