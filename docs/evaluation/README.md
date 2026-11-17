# Frozen internal validation evidence

This directory carries the small, immutable evidence needed to support the
evaluation claims in the project report without publishing raw reviews, model
responses, or local inference caches.

- `final_evaluation.json` contains the reported finding-precision, gold-recall,
  failure, and abstention diagnostics.
- `selection_manifest.json` freezes the 15-review internal validation cohort
  and the protocol established before inference.
- `run_manifest.json` records the frozen model and pipeline fingerprint.
- `completion_manifest.json` records completion and the hashes of the evaluated
  artifacts.

This is an internal locked validation cohort, not an external test set. The
recorded conclusion is a precision-recall trade-off, not a general accuracy or
transfer-performance claim.
