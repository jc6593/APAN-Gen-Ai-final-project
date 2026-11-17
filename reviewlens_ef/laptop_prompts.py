"""Product-neutral prompts for the configurable laptop transfer pipeline.

The evaluated Aspire workflow remains frozen.  This module derives the same
v1.5 policy while removing the one-product wording, so a transfer run cannot be
mistaken for the frozen benchmark.
"""

from __future__ import annotations

import re
from typing import Mapping

from .v15_prompts import OBSERVATION_PROMPT_V15, build_normalization_prompt_v15


PROMPT_VERSION = "reviewlens_laptop_transfer_prompt_v1"


def build_laptop_observation_prompt() -> str:
    """Return the v1.5 extraction policy with product-neutral identity rules."""
    prompt = re.sub(
        r"from ONE customer review of\n[^\n]+\. The input includes",
        "from ONE customer review of the current laptop product named in the "
        "input. The input includes",
        OBSERVATION_PROMPT_V15,
        count=1,
    )
    prompt = re.sub(
        r"to the current [A-Z][A-Za-z0-9_-]*\.",
        "to the current reviewed laptop.",
        prompt,
        count=1,
    )
    prompt = re.sub(
        r"Use seller_platform for seller, Amazon, delivery, "
        r"[A-Z][A-Za-z0-9_-]*/manufacturer support,",
        "Use seller_platform for seller, retailer/platform, delivery, "
        "manufacturer support,",
        prompt,
        count=1,
    )
    prompt += """

CONFIGURABLE LAPTOP IDENTITY:
15. The input fields product_name and product_model define the one current
    reviewed laptop family for this run. Product aliases or child ASINs in the
    review do not change that identity. Do not import facts from another review
    or from product knowledge.
"""
    # Fail closed if an upstream prompt edit makes either product-specific token
    # survive the transformations above.
    if re.search(r"\bA[cC][eE][rR]\b|A5\d{2}", prompt):
        raise ValueError("frozen prompt shape changed; product-specific text remains")
    return prompt


def build_laptop_normalization_prompt(
    taxonomy: Mapping[str, Mapping[str, str]],
) -> str:
    """Use the frozen laptop taxonomy policy without product-specific claims."""
    return build_normalization_prompt_v15(taxonomy) + """

CONFIGURABLE LAPTOP SCOPE:
- Normalize only the validated evidence supplied for the current configured
  laptop family. Product identity never supplies a missing issue cue.
"""


__all__ = [
    "PROMPT_VERSION",
    "build_laptop_normalization_prompt",
    "build_laptop_observation_prompt",
]
