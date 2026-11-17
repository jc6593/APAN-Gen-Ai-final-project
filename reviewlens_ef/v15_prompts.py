"""Versioned prompts for the isolated Evidence-first v1.5 recall repair."""

from __future__ import annotations

from typing import Mapping

from .prompts import OBSERVATION_PROMPT, build_normalization_prompt


OBSERVATION_PROMPT_V15 = OBSERVATION_PROMPT + """

V1.5 RECALL AND ATOMICITY ADDENDUM:
11. Inspect every ordered source record before answering. Emit each concrete,
    supported product finding even when Layer 1 omitted it; do not stop after a
    broad title or overall opinion.
12. Keep composite evidence atomic. In particular, split packaging damage from
    laptop damage; RAM amount from storage amount and upgradeability; battery
    life from portability and task performance; and screen quality from general
    specifications. If one exact span supports several aspects, emit separate
    observations with the shortest faithful substring for each.
13. A short exact quote may rely on only the immediately adjacent supplied
    source record to resolve a pronoun or omitted aspect. Cite the current
    sentence_id and its exact substring only; never concatenate two records or
    invent words from context. Python preserves the adjacent record separately.
14. A completed return/replacement interaction is service_experience. Forced
    Microsoft-account setup, removal of a local-account option, product-key
    problems, and preinstalled bloatware are OS setup findings, not S Mode unless
    the exact source actually discusses S Mode.
"""


def build_normalization_prompt_v15(
    taxonomy: Mapping[str, Mapping[str, str]],
) -> str:
    """Build a v1.5 prompt without changing the frozen v1.4 prompt bytes."""
    prompt = build_normalization_prompt(taxonomy)
    return prompt + """

V1.5 FROZEN-PAIR CLARIFICATIONS:
- Missing power cord/charger is battery/charging. A merely short cord is not.
- Microsoft/local-account setup, Windows product-key issues, and bloatware are
  software_os/os_setup. software_os/s_mode_restriction requires literal S Mode
  evidence or an immediately adjacent S Mode antecedent.
- A chassis/lid dent is build_quality/housing_damage. Damaged product packaging
  is purchase_service/delivery_damage. Do not force one label onto a quote that
  combines both packaging and laptop damage.
- Black screen plus an explicit failure to boot is system_stability/boot_failure;
  a display-only defect remains display/screen_quality.
- An internet/offline/disconnect failure with no wireless mechanism is
  connectivity_ports/other_connectivity; wifi_bluetooth needs a Wi-Fi, wireless,
  or Bluetooth cue.
- Positive fast/speed/responsiveness or explicit task success is
  performance/processing_capability, never slow_performance. Drive read/write
  speed is memory_storage/storage_performance. RAM amount or adequacy is
  memory_storage/ram_capacity.
- Laptop size or case fit is design_portability/dimensions. Weight, lightness,
  carrying, and travel portability are design_portability/weight. Good-looking
  appearance is design_portability/appearance.
- Nearby context is intentionally absent from this model input. Python may use
  one source-validated adjacent record after model generation, but only through
  an auditable deterministic rule. Never infer from another observation. If
  more than one frozen pair remains plausible, use unmapped.
"""


__all__ = ["OBSERVATION_PROMPT_V15", "build_normalization_prompt_v15"]
