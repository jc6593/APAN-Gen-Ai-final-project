# ReviewLens

**Evidence-backed Voice-of-Customer intelligence for laptop product teams**

ReviewLens turns a frozen cohort of customer reviews into prioritized product
findings and keeps every finding connected to an exact, source-verified quote.
It is designed for product managers who need to move from scattered anecdotes
to an auditable product brief.

ReviewLens is **not** a shopping recommender, a market-wide product ranking, or
an unrestricted review chatbot. The demo answers a narrower and more defensible
question: *What product issues and strengths appear in this fixed review
cohort, and what evidence supports each conclusion?*

## Product experience

The v2 demo supports this workflow:

1. Choose one of twelve featured laptop product families across seven brands
   and three operating systems.
2. Review its sample, coverage, and evidence-backed executive brief.
3. Rank product findings by the number of distinct supporting reviews.
4. Open any finding to inspect the exact quotes, review IDs, and ratings.
5. Inspect verified signals that the system deliberately did not force into the
   business taxonomy.
6. Compare any pair on equal frozen 20-review cohorts. The comparison aligns
   the same issue dimensions, separates strengths from friction, summarizes
   only material two-review gaps, and links every signal back to evidence.

The twelve featured cases are:

- Acer Aspire 5 Slim Laptop — benchmark source product
- Apple MacBook Air M2 (2022) and MacBook Pro 16 (2021)
- Microsoft Surface Laptop 3 and Surface Laptop Studio
- HP 14 Ryzen 5
- ASUS E410 and ROG Strix SCAR 17
- Dell Inspiron 13 5000 2-in-1
- Acer Chromebook Spin 13 Convertible
- Lenovo IdeaPad G50 and B50 Business

The six established cases retain their frozen 30-review reports; six additional
cases use frozen 20-review reports. Every head-to-head comparison recomputes
support on the same first 20 selected reviews per product. The Acer Aspire 5
supplied the domain-development and internal-evaluation setting, but its display
cohort was not separately evaluated. The other eleven products are transfer
demonstrations; product-specific accuracy has not been established for them.

## Why this is more than an off-the-shelf prompt

ReviewLens adds a reproducible evidence and decision layer around local language
models:

- A fine-tuned Qwen3-1.7B Layer 1 supplies laptop aspect clues.
- Qwen3-4B extracts atomic observations before assigning business labels.
- Deterministic Python checks require an exact quote, valid source location,
  and consistent offsets.
- Constrained normalization maps only supported observations into a frozen
  laptop taxonomy; unsafe mappings remain explicitly `unmapped`.
- Findings are merged and ranked by distinct supporting reviews rather than by
  raw model mentions.
- Every run is fingerprinted and resumable, and stale or tampered caches fail
  closed.
- A small, hash-verified demo bundle separates expensive local inference from
  the portable API and interface.

```text
Amazon review
    ↓
Layer 1 aspect clues
    ↓
Atomic observation extraction
    ↓
Exact-quote and location validation
    ↓
Constrained taxonomy mapping or explicit abstention
    ↓
Review-level aggregation → evidence-backed report → FastAPI / web demo
```

## Architecture

The project deliberately separates two runtime profiles.

### Offline precomputation (Apple Silicon / local MLX)

The batch pipeline loads Layer 1 once, completes all twelve products, releases it,
then loads Layer 2 once. This keeps only one model family resident at a time and
is appropriate for the 24 GB M4 Pro development machine. It writes immutable,
fingerprinted product runs and then creates the portable `demo_bundle_v2/`.

Pinned models:

- Layer 1: `Qwen/Qwen3-1.7B-MLX-4bit`
- Layer 2: `mlx-community/Qwen3-4B-Instruct-2507-4bit`

The exact Hugging Face revisions are frozen in
`configs/demo/featured_laptops_v2.json`. Layer 1 also requires the existing
local adapter and checkpoint under `outputs/adapters/reviewlens_layers16/`.

### Portable serving (FastAPI / Docker)

The API serves only the precomputed, hash-bound bundle. It does not load MLX,
Qwen weights, adapters, raw reviews, or model response traces. This makes the
classroom deployment small enough to run on an ordinary instructor machine.

## Quick start

Run commands from the repository root.

### 1. Run the portable test suite

```bash
python -m pip install -r requirements-api.txt
python -m unittest \
  tests.test_api \
  tests.test_demo_bundle \
  tests.test_laptop_batch \
  tests.test_laptop_cli \
  tests.test_shipped_demo_bundle -v
python -m compileall reviewlens_ef app noteboks/15_review_analysis.py
```

These tests run from a lightweight repository clone and do not load an ML
model. Developers who also have the ignored local datasets and historical
outputs can run the complete test suite:

```bash
python -m unittest discover -s tests -v
```

### 2. Build the featured demo bundle

Use the existing `reviewlens` environment on the Apple Silicon development
machine. The two pinned models must already be available in the local Hugging
Face cache.

```bash
python noteboks/15_review_analysis.py \
  --mode laptop-demo-precompute \
  --demo-config configs/demo/featured_laptops_v2.json \
  --show 0
```

The command is append-and-resume: rerunning it reuses valid ordered caches and
processes only missing work. Do not pass `--limit` or `--force`. With the
existing six products cached, only the six new 20-review products require model
work. The resulting `demo_bundle_v2/manifest.json` binds every product report
to a SHA-256 hash.

The verified v2 bundle is the API and Docker default. To serve a different
bundle during development, set `REVIEWLENS_BUNDLE_DIR` explicitly:

```bash
REVIEWLENS_BUNDLE_DIR=demo_bundle_v2 \
  uvicorn app.main:app --host 0.0.0.0 --port 8000
```

### 3. Run the API locally

After `demo_bundle_v2/` exists:

```bash
python -m unittest discover -s tests -p 'test_api.py' -v
uvicorn app.main:app --host 0.0.0.0 --port 8000
```

Open:

- Dashboard: <http://localhost:8000/>
- Interactive API documentation: <http://localhost:8000/docs>
- Health check: <http://localhost:8000/health>

To serve a bundle from another location, set `REVIEWLENS_BUNDLE_DIR` to that
directory before starting Uvicorn.

### 4. Run with Docker

The Docker image packages only the small, hash-verified v2 serving bundle. Build
and run it with:

```bash
docker build -t reviewlens .
docker run --rm -p 8000:8000 reviewlens
```

In another terminal, smoke-test the running container:

```bash
curl http://localhost:8000/health
curl http://localhost:8000/api/products
curl -X POST http://localhost:8000/api/compare \
  -H 'Content-Type: application/json' \
  -d '{
    "left_product_id": "acer_aspire5",
    "right_product_id": "dell_inspiron_b077xx31vl"
  }'
```

The health response should report `featured_laptops_v2` and
`"product_count":12`; the comparison should report
`"cohort_size_per_product":20`.

The container is the serving layer only; model inference remains an offline
precomputation step. The image should be built and smoke-tested on the target
Docker installation before submission.

## API

| Method | Endpoint | Purpose |
|---|---|---|
| `GET` | `/health` | Service and bundle integrity metadata |
| `GET` | `/api/products` | Featured product catalog |
| `GET` | `/api/products/{product_id}/report` | Product sample, coverage, and brief |
| `GET` | `/api/products/{product_id}/findings` | Ranked evidence-backed findings |
| `GET` | `/api/products/{product_id}/findings/{finding_id}/evidence` | Exact supporting quotes |
| `GET` | `/api/products/{product_id}/signals` | Verified but intentionally unmapped signals |
| `POST` | `/api/compare` | Equal-sample directional comparison |
| `GET` | `/api/methodology` | Method and reporting limitations |

Example comparison request:

```bash
curl -X POST http://localhost:8000/api/compare \
  -H 'Content-Type: application/json' \
  -d '{
    "left_product_id": "acer_aspire5",
    "right_product_id": "apple_macbook_air_b0b4f9h4df"
  }'
```

## Data

- **Layer 1 training/evaluation source:** SemEval 2014 Task 4 laptop aspect
  sentiment data.
- **Product-review source:** McAuley Lab Amazon Reviews 2023 Electronics data.
- **Existing Acer source:** 189 cleaned Acer Aspire 5 reviews remain preserved.
- **Catalog discovery:** 18,227,164 Electronics records were screened once to
  locate suitable product families; this is not a claim that all records were
  analyzed by the language models.
- **Qualified local pool:** 549 reviews across the twelve featured families.
- **Evidence-first display cohorts:** 300 deterministic, rating-aware reviews
  (six products at 30 reviews and six at 20).
- **Fair comparisons:** all 66 product pairs use equal frozen 20-review
  prefixes.

The transfer-product families were selected from bounded product catalogs with
brand and operating-system diversity constraints. The data funnel is
intentional: broad records support discovery, the qualified pool supports
auditable product selection, and the smaller frozen cohorts keep local MLX
inference reproducible and comparison-balanced. These samples are useful for a
course demonstration, not for estimating market prevalence or customer
population shares.

## Evidence-first method

1. **Prepare.** Product configs define the allowed ASIN family, source file,
   selection seed, taxonomy, and isolated output directory.
2. **Extract clues.** The existing fine-tuned Layer 1 analyzes body sentences.
3. **Observe.** Layer 2 reads the title, full review, and Layer 1 clues and emits
   atomic statements without assigning a business category.
4. **Verify.** Deterministic checks locate the cited text exactly and reject
   unsupported, ambiguous, or bundled observations.
5. **Normalize.** Layer 2 chooses a frozen category/issue pair or abstains.
   Deterministic boundary rules prevent common category shifts and preserve the
   verified observation as the canonical claim.
6. **Aggregate.** Mapped claims are merged at review level, then aggregated by
   distinct supporting reviews. Unmapped claims stay visible as signals.
7. **Freeze.** The demo builder exports only concise evidence and business
   outputs. The API verifies schemas and file hashes before serving anything.

Optional NLI is available as an advisory diagnostic through
`requirements-nli.txt`; it never changes accepted findings.

## Evaluation status

The Evidence-first policy was frozen before a locked 15-review Acer internal
validation cohort was run. On that small cohort, v1.5 improved strict
finding-level precision over the v3 baseline (`23/30` versus `29/52`) but had
lower end-to-end full-tuple recall (`25/60` versus `32/60`). The recorded
decision is therefore a **precision-recall trade-off**, not an overall model
win. Raw numerators and denominators are reported because the cohort is too
small for significance or population-level claims.

That evaluation does not establish accuracy for the eleven transfer products,
or for the separate Acer display cohort. The interface labels these
cases accordingly and presents comparisons as directional only.

The immutable evaluation summary and its selection, run, and completion
manifests are committed under [`docs/evaluation/`](docs/evaluation/). Raw
reviews, model responses, weights, and local inference caches are deliberately
excluded from the portable repository.

## Reporting limitations

- Findings describe fixed convenience samples, not all owners or the laptop
  market.
- Counts are numbers of distinct reviews, not causal estimates or defect rates.
- Missing findings can reflect conservative abstention as well as absent issues.
- Exact evidence proves that a review supports a statement; it does not prove
  that the reviewer is representative or factually correct.
- Product families can contain allowed child ASIN variants.
- No transfer-product accuracy claim should be made without product-specific
  annotation.
- The demo is decision support. Product teams should inspect evidence before
  taking action.

## Repository map

```text
app/                    FastAPI service and static product interface
configs/catalogs/       Versioned catalog-collection contract
configs/demo/           Frozen v1 and twelve-product v2 demo contracts
configs/products/       Product-family definitions
configs/taxonomies/     Frozen laptop business taxonomy
data/                   Local-only SemEval and prepared Amazon review data
docs/evaluation/        Frozen internal-validation summary and manifests
noteboks/               Preserved historical analysis entry points
reviewlens_ef/           Evidence-first, catalog, batch, and bundle modules
tests/                  Deterministic unit and integration tests
demo_bundle*/           Generated portable API data (after precomputation)
```

The directory name `noteboks/` is intentionally preserved for compatibility
with the existing project and commands.

## Course-project deliverables

The repository is structured for the required submission:

- organized Generative AI implementation with a reproducible evidence layer;
- FastAPI endpoints and a user-facing product workflow;
- a Docker serving image that does not require local model weights;
- evaluation artifacts that expose the precision-recall trade-off; and
- material for a nontechnical presentation centered on one defensible story:
  **select a laptop, see the prioritized Voice-of-Customer findings, and trace
  every conclusion back to customer evidence.**

For the detailed historical Evidence-first experiments and evaluation commands,
see [`EVIDENCE_FIRST.md`](EVIDENCE_FIRST.md).
