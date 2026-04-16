"""Pluggable IHC (immunohistochemistry) scoring engine.

Rule-based path: deterministic, no training. Uses color deconvolution
(H/E/DAB separation via `skimage.color.rgb2hed`) plus optional cellpose
nuclear / membrane segmentation to compute clinical IHC scores per slide.

Learned path (via `patch_labels`): turns rule-based slide-level or
patch-level labels into a training set so the same score can be learned
by a per-patch regressor/classifier, then aggregated to slide level.

Supported presets (extend via `register_rule`):
    ki67_pi         — nuclear Ki-67 proliferation index (% DAB+ nuclei)
    er_allred       — ER Allred 0–8 (proportion + intensity, nuclear)
    pr_allred       — PR Allred 0–8 (proportion + intensity, nuclear)
    her2_membrane   — HER2 0/1+/2+/3+ (membrane DAB band + completeness)
    pdl1_tps        — PD-L1 Tumor Proportion Score (% membrane-stained tumor)

Custom rules: pass `rule_override={...}` through `score_slide` /
`score_ihc` tool; fields follow `Rule` dataclass schema.
"""
from __future__ import annotations

from .rules import Rule, REGISTRY, register_rule, get_rule, list_rules  # noqa: F401
from .score import score_slide, score_dataset  # noqa: F401
from .patch_labels import build_ihc_patch_labels  # noqa: F401
