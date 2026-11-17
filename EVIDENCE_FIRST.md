# ReviewLens Evidence-first v1.4 frozen evaluation and holdout

Evidence-first v1.4 model behavior is now frozen. This phase keeps the existing
Layer 1 cache, Layer 2 v3 baseline, aggregation, Evidence Pack, original CLI
modes, prompts, schemas, taxonomy, and deterministic normalization unchanged.
It adds a fair human-evaluation layer and an isolated 25-review holdout under
`outputs/amazon/evidence_first/`.

## Data flow

1. **Observation extraction** — Qwen reads the review title, full review, and
   existing Layer 1 clues, then returns an exact quote plus one atomic
   observation. It labels each statement as actual experience, capability or
   limitation, intended use, expectation or assumption, feature description,
   service experience, overall opinion, or unclear. It does not assign a
   business category.
2. **Deterministic evidence gate** — Python verifies the cited title/body
   source, an exact quote, source offsets, and body offsets when applicable. A
   unique casing-only mismatch is replaced with the exact source slice and
   recorded; no fuzzy text repair is allowed. It rejects clearly bundled
   multi-aspect output and assigns the observation ID plus adjacent context.
3. **Constrained normalization** — Qwen maps each validated observation to the
   frozen v3 taxonomy or explicitly returns `unmapped`. Nearby sentences are
   deliberately withheld at this stage so the model cannot shift a category to
   the next observation. It cannot regenerate or change evidence, statement
   type, sentiment, product attribution, or the final claim text; Python uses
   the validated observation as the canonical claim and records any model
   paraphrase separately.
   Deterministic boundary rules prevent mappings such as crash → slow
   performance, gaming suitability → slow performance, SSD expansion → storage
   performance, intended use → proven compatibility, and recommendation → value
   for money without price/value evidence. If an issue group with an explicit
   deterministic evidence gate is placed under the wrong category, its unique
   frozen parent is restored and recorded before that gate runs. Power-port
   failures are routed from generic ports to `battery/charging`. Speaker,
   keyboard, touchpad, appearance, price, and physical-port mappings also have
   explicit evidence gates. Ungated groups are never auto-corrected. An invalid
   decision that survives the retry becomes an auditable `unmapped` fallback,
   so no validated observation disappears. A final high-precision recovery
   layer can map only one uniquely supported pair from the exact quote. It is
   limited to explicit deal/price judgments, power-port failure,
   waste-of-money judgments, and app blocking with a concrete
   download/install/run action. It excludes intentions, expectations, service
   anecdotes, unclear statements, generic Chrome advice, and ambiguous
   multi-pair evidence. Every recovery records its rule and prior source.
4. **Optional NLI** — a separate DeBERTa process checks claim, sentiment, and
   target support. Its uncalibrated result is advisory and never changes claims.
5. **Business finding merge** — atomic claims remain auditable, while claims
   with the same review/category/issue/sentiment/attribution tuple are merged
   into one business finding with all evidence spans.
6. **Human evaluation** — the primary blind sheet compares the same unit for
   both systems: v3 review-level findings versus EF merged review-level
   findings. Atomic evidence, unmapped abstentions, and independent gold
   coverage are separate diagnostics. System identity is kept in a hidden key.

## Run the deterministic tests

From the `ReviewLens` project root:

```bash
python3 -m unittest discover -s tests -v
python3 -m compileall reviewlens_ef noteboks/15_review_analysis.py
```

The tests do not load MLX, PyTorch, or any model.

## Run the fixed 10-review MLX pilot

Use the same local environment in which the existing MLX Layer 2 command runs:

```bash
python noteboks/15_review_analysis.py --mode ef-pilot --limit 10 --show 10
```

`--force-claims` is only for migration from an older claim schema. It preserves
the validated observation cache, backs up downstream pilot state, and reruns
normalization. A normal v1.4 run resumes from valid cache rows and should omit
both force flags. The pilot cannot expand beyond its fixed 10 reviews.

The old descriptive command remains available unchanged for compatibility:

```bash
python noteboks/15_review_analysis.py --mode ef-summary --show 10
```

Its `claim_precision_review.csv` and `extraction_coverage_review.csv` mix legacy
diagnostic units and should not be used for the new primary human comparison.
They are retained only so existing commands and consumers continue to work.

## Create the fair pilot evaluation sheets

```bash
python noteboks/15_review_analysis.py --mode ef-eval --show 10
```

This command does not load or call a model. Fill the human columns in:

- `business_finding_precision_blind.csv` — primary v3 versus EF finding precision
- `ef_atomic_evidence_audit.csv` — EF mapped atomic evidence diagnostic
- `ef_abstention_audit.csv` — EF unmapped decision diagnostic
- `gold_coverage_review.csv` — independent review-level gold findings

Keep `business_finding_precision_key.json` away from raters until labels are
frozen. Re-running `ef-eval` preserves labels by stable IDs, validates any row
marked `human_gold_complete=yes`, and writes `human_metrics.json`. A complete
gold JSON cell must be a list of objects with exactly `category`,
`issue_group`, `sentiment`, `attribution`, and a verbatim `evidence_quote`.

For human annotation, complete the candidate-free `gold_coverage_review.csv`
first, then the primary blind precision sheet, then the atomic and abstention
diagnostics. Do not open the blind key until primary labels are frozen. Use only
`yes` or `no`; leave genuinely uncertain fields blank and explain them in
`notes`.

## Freeze and run the 25-review holdout

Freeze the cohort before any holdout inference:

```bash
python noteboks/15_review_analysis.py --mode ef-holdout-prepare
```

Preparation calls no model. It deterministically selects 25 reviews, five per
star rating, excludes the ten pilot reviews, records source/config/code hashes,
and creates an immutable `holdout/selection_manifest.json`.

Start with a five-review MLX pilot that covers all five ratings, then resume the
same frozen cohort to 25:

```bash
python noteboks/15_review_analysis.py --mode ef-holdout --limit 5 --show 5
python noteboks/15_review_analysis.py --mode ef-holdout --show 10
python noteboks/15_review_analysis.py --mode ef-holdout-summary --show 10
```

The second inference command reuses the first five cache rows and processes
only the remaining twenty. It can never expand beyond the frozen 25. A source,
model, prompt, taxonomy, schema, or model-code fingerprint mismatch stops the
run instead of silently changing the cohort. `--force` on `ef-holdout` backs up
only holdout inference caches and the holdout run manifest; it never changes
the selection manifest or any pilot file.

The holdout summary reports cache completion separately from valid generation.
`human_metrics.json` keeps end-to-end coverage over all cache-complete reviews
and also reports both systems on the same conditional valid-generation subset.
A generation failure remains an empty EF prediction in the end-to-end metric;
it is never silently repaired or removed from the 25-review gold set.

## Optional NLI pass

Install the optional dependencies in the local ML environment:

```bash
python -m pip install -r requirements-nli.txt
```

Run NLI only after `ef-pilot` has finished, so Qwen and DeBERTa do not need to
remain in memory together:

```bash
python noteboks/15_review_analysis.py --mode ef-nli --nli-device auto --nli-batch-size 8 --show 10 --force
python noteboks/15_review_analysis.py --mode ef-summary --show 10
```

Use `--force` for the first NLI run after v1.4 claim regeneration; subsequent
resumptions should omit it. `auto` selects Apple MPS when
available and falls back to CPU if initialization or inference fails. The first
run downloads the configured Hugging Face model.

## Outputs

- `pilot_manifest.json` — fixed IDs plus source, prompt, model, and taxonomy hashes
- `observations_pilot.jsonl` — validated observations and quarantined model output
- `claims_pilot.jsonl` — atomic mapped/unmapped claims plus merged business findings
- `nli_pilot.jsonl` — optional advisory component probabilities
- `comparison_v3_vs_ef.json` — legacy-compatible descriptive pilot comparison
- `claim_precision_review.csv` — legacy mixed-unit diagnostic sheet
- `extraction_coverage_review.csv` — legacy diagnostic coverage sheet
- `business_finding_precision_blind.csv` — fair primary finding-level blind sheet
- `business_finding_precision_key.json` — hidden blind-ID key
- `ef_atomic_evidence_audit.csv` — mapped atomic diagnostic
- `ef_abstention_audit.csv` — unmapped diagnostic
- `gold_coverage_review.csv` — candidate-free human gold sheet
- `human_metrics.json` — human-label precision and common-gold coverage metrics
- `holdout/selection_manifest.json` — immutable 25-review selection
- `holdout/run_manifest.json` — inference fingerprint written before model load
- `holdout/observations_holdout.jsonl` — isolated holdout observation cache
- `holdout/claims_holdout.jsonl` — isolated holdout claim/finding cache
- `holdout/evaluation/` — holdout versions of the fair evaluation artifacts

All stage caches are append-and-resume files. A stale prompt, taxonomy, source,
model, or upstream hash is rejected rather than silently mixed with a new run.

## Configurable laptop product catalog

The evaluated Acer workflow above remains frozen. A separate laptop transfer
workflow lets the same Layer 1 and Evidence-first policy analyze preloaded
laptop product families without writing into any Acer evaluation directory.

To expand beyond the three legacy product configs, run the versioned data-only
collector. It applies a high-precision whole-device filter, scans all planned
parent ASINs in one pass, and freezes five new product families only after each
has at least 30 usable reviews. Together with the three preserved legacy configs,
this yields eight selectable catalog entries. It resolves and records the exact remote dataset
revision, checkpoints every 100,000 records, and never loads a model:

```bash
python noteboks/15_review_analysis.py \
  --mode laptop-catalog-expand \
  --catalog-config configs/catalogs/laptop_catalog_v1.json
```

The command is resumable with the same command. Frozen collection artifacts
are written under `data/processed/amazon/laptop_catalog_v1/`; qualified product
configs are added under `configs/products/`. Existing Acer, HP Chromebook, and
MacBook Pro configs and outputs are never overwritten. The resulting 30-review
cohorts are bounded convenience samples for the course demo, not market-share
or population estimates.

List the locally available products without loading a model:

```bash
python noteboks/15_review_analysis.py --mode laptop-catalog
```

Run the first ten reviews for one configured product. The normal command runs
the existing Qwen3-1.7B Layer 1 adapter first, releases it, and then loads the
Qwen3-4B Evidence-first Layer 2 model:

```bash
python noteboks/15_review_analysis.py \
  --mode laptop-pilot \
  --product-config configs/products/hp_chromebook_14.json \
  --limit 10 \
  --show 10
```

Read the completed cached result later without loading either model:

```bash
python noteboks/15_review_analysis.py \
  --mode laptop-summary \
  --product-config configs/products/hp_chromebook_14.json \
  --limit 10 \
  --show 10
```

All new state is isolated under `outputs/laptops/<product_id>/`. Non-Acer
results are labeled `transfer_demo_unvalidated`: they demonstrate that the
pipeline is product-configurable, but they do not inherit the Acer benchmark's
measured precision or recall. The current supported category is laptops; a new
product category would require its own taxonomy and evaluation.
