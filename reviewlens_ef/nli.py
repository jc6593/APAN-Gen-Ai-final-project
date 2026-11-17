"""Optional, advisory NLI checks for Evidence-first claims.

This module intentionally imports PyTorch and Transformers only when NLI is
requested.  Its output is a separate cache and never changes accepted claims.
"""

from __future__ import annotations

import hashlib
import json
import math
import shutil
from collections import Counter, defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable


DEFAULT_NLI_MODEL = "MoritzLaurer/DeBERTa-v3-base-mnli-fever-anli"
NLI_SCHEMA_VERSION = "reviewlens_ef_nli_advisory_v1_1"
NLI_METHOD_VERSION = "quote_plus_title_or_adjacent_context_v1_1"
COMPONENTS = ("claim", "sentiment", "target")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        raise FileNotFoundError(path)
    rows = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, 1):
            if not line.strip():
                continue
            value = json.loads(line)
            if not isinstance(value, dict):
                raise ValueError(f"Expected an object at {path}:{line_number}")
            rows.append(value)
    return rows


def _stable_hash(value: Any) -> str:
    encoded = json.dumps(
        value, ensure_ascii=False, sort_keys=True, separators=(",", ":")
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _backup(path: Path) -> Path | None:
    if not path.exists():
        return None
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    destination = path.with_name(f"{path.stem}.backup_{stamp}{path.suffix}")
    shutil.move(path, destination)
    return destination


def _normalise_label(label: str) -> str:
    compact = label.strip().casefold().replace("_", " ").replace("-", " ")
    if "entail" in compact:
        return "entailment"
    if "neutral" in compact:
        return "neutral"
    if "contrad" in compact:
        return "contradiction"
    raise ValueError(
        f"Unsupported NLI label {label!r}. The model config must name "
        "entailment, neutral, and contradiction explicitly."
    )


class TransformersNliScorer:
    """Small batched wrapper around a Hugging Face sequence classifier."""

    def __init__(
        self,
        model_id: str = DEFAULT_NLI_MODEL,
        device: str = "auto",
        batch_size: int = 8,
    ) -> None:
        try:
            import torch
            from transformers import AutoModelForSequenceClassification, AutoTokenizer
        except ImportError as exc:  # pragma: no cover - depends on optional env
            raise RuntimeError(
                "NLI is optional. Install requirements-nli.txt in the ML environment "
                "before running --mode ef-nli."
            ) from exc

        if batch_size < 1:
            raise ValueError("batch_size must be at least 1")
        if device not in {"auto", "mps", "cpu"}:
            raise ValueError("device must be auto, mps, or cpu")
        chosen = device
        if chosen == "auto":
            chosen = "mps" if torch.backends.mps.is_available() else "cpu"
        if chosen == "mps" and not torch.backends.mps.is_available():
            raise RuntimeError("MPS was requested but is unavailable")

        self._torch = torch
        self.model_id = model_id
        self.device = chosen
        self.batch_size = batch_size
        self.tokenizer = AutoTokenizer.from_pretrained(model_id)
        self.model = AutoModelForSequenceClassification.from_pretrained(model_id)
        self.model.to(chosen)
        self.model.eval()
        self.model_revision = (
            getattr(self.model.config, "_commit_hash", None) or "unknown"
        )

        id2label = getattr(self.model.config, "id2label", None) or {}
        self._labels: dict[int, str] = {}
        for raw_index, raw_label in id2label.items():
            self._labels[int(raw_index)] = _normalise_label(str(raw_label))
        if set(self._labels.values()) != {
            "entailment", "neutral", "contradiction"
        }:
            raise ValueError(
                "NLI model id2label must contain exactly entailment, neutral, "
                f"and contradiction; got {id2label!r}"
            )

    def score_pairs(
        self, pairs: list[tuple[str, str]]
    ) -> list[dict[str, Any]]:
        """Return raw logits, probabilities, and the highest-probability label."""
        results: list[dict[str, Any]] = []
        torch = self._torch
        for start in range(0, len(pairs), self.batch_size):
            batch = pairs[start:start + self.batch_size]
            premises = [item[0] for item in batch]
            hypotheses = [item[1] for item in batch]
            encoded = self.tokenizer(
                premises,
                hypotheses,
                padding=True,
                truncation=True,
                max_length=512,
                return_tensors="pt",
            )
            encoded = {key: value.to(self.device) for key, value in encoded.items()}
            with torch.inference_mode():
                logits_tensor = self.model(**encoded).logits
                probabilities_tensor = torch.softmax(logits_tensor, dim=-1)
            logits_rows = logits_tensor.detach().cpu().tolist()
            probability_rows = probabilities_tensor.detach().cpu().tolist()
            for logits, probabilities in zip(logits_rows, probability_rows):
                probability_map = {
                    self._labels[index]: float(probability)
                    for index, probability in enumerate(probabilities)
                }
                best = max(probability_map, key=probability_map.get)
                results.append({
                    "label": best,
                    "logits": [float(value) for value in logits],
                    "probabilities": probability_map,
                })
        return results


def _sentence_neighbours(
    sentences: list[dict[str, Any]],
) -> tuple[dict[str, dict[str, Any]], dict[str, list[dict[str, Any]]]]:
    by_id = {row["sentence_id"]: row for row in sentences}
    by_review: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in sentences:
        by_review[row["review_id"]].append(row)
    for rows in by_review.values():
        rows.sort(key=lambda item: (
            0 if item.get("source_field") == "title" else 1,
            item.get("start_char", 0),
            item["sentence_id"],
        ))
    return by_id, by_review


def _with_title_sources(
    sentences: list[dict[str, Any]],
    reviews: list[dict[str, Any]],
) -> list[dict[str, Any]]:
    """Expose review titles to NLI with the same synthetic IDs as extraction."""
    sources = [dict(row, source_field=row.get("source_field", "clean_text"))
               for row in sentences]
    existing_ids = {row.get("sentence_id") for row in sources}
    for review in reviews:
        review_id = review.get("review_id")
        title = review.get("title")
        if not isinstance(review_id, str) or not isinstance(title, str) or not title.strip():
            continue
        sentence_id = f"{review_id}_TITLE"
        if sentence_id in existing_ids:
            raise ValueError(f"Synthetic title ID collides with source sentence: {sentence_id}")
        existing_ids.add(sentence_id)
        sources.append({
            "sentence_id": sentence_id,
            "review_id": review_id,
            "sentence_text": title,
            "source_field": "title",
            "start_char": 0,
            "end_char": len(title),
        })
    return sources


def _evidence_quote(evidence: dict[str, Any]) -> str:
    for key in ("evidence_text", "quote", "text"):
        value = evidence.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return ""


def _build_premise(
    review_id: str,
    claim: dict[str, Any],
    sentence_by_id: dict[str, dict[str, Any]],
    sentences_by_review: dict[str, list[dict[str, Any]]],
) -> str:
    """Use exact quotes plus at most one adjacent sentence on either side."""
    quotes = []
    cited_ids = []
    for evidence in claim.get("evidence", []) or []:
        if not isinstance(evidence, dict):
            continue
        quote = _evidence_quote(evidence)
        if quote:
            quotes.append(quote)
        sentence_id = evidence.get("sentence_id")
        if isinstance(sentence_id, str):
            sentence = sentence_by_id.get(sentence_id)
            if sentence and sentence.get("review_id") == review_id:
                cited_ids.append(sentence_id)

    context: list[str] = []
    review_sentences = sentences_by_review.get(review_id, [])
    positions = {row["sentence_id"]: index for index, row in enumerate(review_sentences)}
    wanted_positions: set[int] = set()
    for sentence_id in cited_ids:
        position = positions.get(sentence_id)
        if position is None:
            continue
        wanted_positions.update(
            index for index in (position - 1, position, position + 1)
            if 0 <= index < len(review_sentences)
        )
    for index in sorted(wanted_positions):
        text = review_sentences[index].get("sentence_text")
        if isinstance(text, str) and text.strip():
            context.append(text.strip())

    if not quotes:
        raise ValueError(
            f"Claim {claim.get('claim_id')} in {review_id} has no usable exact evidence"
        )
    premise = "Evidence quote: " + " | ".join(dict.fromkeys(quotes))
    if context:
        premise += "\nNearby review context: " + " ".join(dict.fromkeys(context))
    return premise[:2400]


def _sentiment_hypothesis(sentiment: str) -> str:
    templates = {
        "positive": "The reviewer expresses a positive opinion about the described target.",
        "negative": "The reviewer expresses a negative opinion about the described target.",
        "neutral": "The reviewer describes the target without expressing a positive or negative opinion.",
        "mixed": "The reviewer expresses both positive and negative opinions about the described target.",
    }
    if sentiment not in templates:
        raise ValueError(f"Unsupported sentiment for NLI: {sentiment!r}")
    return templates[sentiment]


def _target_hypothesis(attribution: str) -> str:
    templates = {
        "current_product": "The reviewer is describing the Acer Aspire 5 currently being reviewed.",
        "previous_product": "The reviewer is describing a product they owned previously.",
        "other_product": "The reviewer is describing a different product, not the current Acer Aspire 5.",
        "seller_platform": "The reviewer is describing the seller, shopping platform, delivery, or support service.",
        "unclear": "The text does not make clear which product or service is being described.",
    }
    if attribution not in templates:
        raise ValueError(f"Unsupported attribution for NLI: {attribution!r}")
    return templates[attribution]


def _build_nli_inputs(
    review_id: str,
    claim: dict[str, Any],
    sentence_by_id: dict[str, dict[str, Any]],
    sentences_by_review: dict[str, list[dict[str, Any]]],
) -> tuple[str, dict[str, str]]:
    premise = _build_premise(
        review_id, claim, sentence_by_id, sentences_by_review
    )
    claim_text = claim.get("claim")
    if not isinstance(claim_text, str) or not claim_text.strip():
        raise ValueError(f"Claim {claim.get('claim_id')} has no claim text")
    hypotheses = {
        "claim": claim_text.strip(),
        "sentiment": _sentiment_hypothesis(str(claim.get("sentiment"))),
        "target": _target_hypothesis(str(claim.get("attribution"))),
    }
    return premise, hypotheses


def _advisory_status(components: dict[str, dict[str, Any]]) -> str:
    """Conservative, explicitly uncalibrated routing; never edits claims."""
    contradiction = max(
        float(components[name]["probabilities"].get("contradiction", 0.0))
        for name in COMPONENTS
    )
    entailments = [
        float(components[name]["probabilities"].get("entailment", 0.0))
        for name in COMPONENTS
    ]
    if contradiction >= 0.75:
        return "contradicted"
    if min(entailments) >= 0.70:
        return "supported"
    return "unclear"


def _iter_claims(rows: Iterable[dict[str, Any]]) -> Iterable[tuple[str, dict[str, Any]]]:
    for row in rows:
        review_id = row.get("review_id")
        if not isinstance(review_id, str):
            raise ValueError("Claim cache row is missing review_id")
        for index, claim in enumerate(row.get("accepted_claims", []) or [], 1):
            if not isinstance(claim, dict):
                raise ValueError(f"Non-object claim in {review_id}")
            enriched = dict(claim)
            enriched.setdefault("claim_id", f"{review_id}_C{index:03d}")
            yield review_id, enriched


def run_nli(
    project_root: str | Path,
    model_id: str = DEFAULT_NLI_MODEL,
    device: str = "auto",
    batch_size: int = 8,
    force: bool = False,
    show: int = 5,
) -> dict[str, Any]:
    """Run optional NLI after EF generation and cache advisory results.

    The Qwen cache is read-only.  This function loads no MLX model and writes
    only ``outputs/amazon/evidence_first/nli_pilot.jsonl``.
    """
    root = Path(project_root).resolve()
    ef_dir = root / "outputs/amazon/evidence_first"
    claims_path = ef_dir / "claims_pilot.jsonl"
    sentences_path = root / "data/processed/amazon/acer_aspire5_sentences.jsonl"
    reviews_path = root / "data/processed/amazon/acer_aspire5_reviews_clean.jsonl"
    output_path = ef_dir / "nli_pilot.jsonl"
    claims_rows = _read_jsonl(claims_path)
    sentences = _read_jsonl(sentences_path)
    reviews = _read_jsonl(reviews_path)
    sources = _with_title_sources(sentences, reviews)
    sentence_by_id, sentences_by_review = _sentence_neighbours(sources)

    if force:
        backed_up = _backup(output_path)
        if backed_up:
            print("Previous NLI cache backed up:", backed_up)
    saved: dict[tuple[str, str], dict[str, Any]] = {}
    if output_path.exists():
        for row in _read_jsonl(output_path):
            key = (row.get("review_id"), row.get("claim_id"))
            if not all(isinstance(value, str) for value in key):
                raise ValueError("Invalid record in NLI cache")
            if key in saved:
                raise ValueError(f"Duplicate NLI cache key: {key}")
            saved[key] = row

    pending: list[tuple[str, dict[str, Any], str, dict[str, str], str]] = []
    for review_id, claim in _iter_claims(claims_rows):
        premise, hypotheses = _build_nli_inputs(
            review_id, claim, sentence_by_id, sentences_by_review
        )
        fingerprint = _stable_hash({
            "schema": NLI_SCHEMA_VERSION,
            "method": NLI_METHOD_VERSION,
            "model_id": model_id,
            "premise": premise,
            "hypotheses": hypotheses,
        })
        key = (review_id, claim["claim_id"])
        existing = saved.get(key)
        if existing:
            if (
                existing.get("schema_version") != NLI_SCHEMA_VERSION
                or existing.get("model_id") != model_id
                or existing.get("input_hash") != fingerprint
            ):
                raise ValueError(
                    f"Stale NLI cache for {review_id}/{claim['claim_id']}; "
                    "rerun ef-nli with --force to back up this NLI cache only"
                )
            continue
        pending.append((review_id, claim, premise, hypotheses, fingerprint))

    print(
        f"Evidence-first NLI: claims={len(saved) + len(pending)}, "
        f"cached={len(saved)}, pending={len(pending)}"
    )
    scorer: TransformersNliScorer | None = None
    if pending:
        try:
            scorer = TransformersNliScorer(model_id, device, batch_size)
        except RuntimeError:
            if device != "auto":
                raise
            print("MPS/auto NLI initialization failed; retrying on CPU.")
            scorer = TransformersNliScorer(model_id, "cpu", batch_size)

        ef_dir.mkdir(parents=True, exist_ok=True)
        with output_path.open("a", encoding="utf-8") as handle:
            for start in range(0, len(pending), batch_size):
                batch = pending[start:start + batch_size]
                pairs = [
                    (premise, hypotheses[component])
                    for _, _, premise, hypotheses, _ in batch
                    for component in COMPONENTS
                ]
                try:
                    scored = scorer.score_pairs(pairs)
                except RuntimeError:
                    if device != "auto" or scorer.device != "mps":
                        raise
                    print("MPS NLI inference failed; continuing this run on CPU.")
                    scorer = TransformersNliScorer(model_id, "cpu", batch_size)
                    scored = scorer.score_pairs(pairs)
                if len(scored) != len(pairs):
                    raise ValueError("NLI scorer returned the wrong result count")
                cursor = 0
                for review_id, claim, premise, hypotheses, fingerprint in batch:
                    components = {}
                    for component in COMPONENTS:
                        components[component] = scored[cursor]
                        cursor += 1
                    record = {
                        "schema_version": NLI_SCHEMA_VERSION,
                        "method_version": NLI_METHOD_VERSION,
                        "review_id": review_id,
                        "claim_id": claim["claim_id"],
                        "model_id": model_id,
                        "model_revision": scorer.model_revision,
                        "device": scorer.device,
                        "input_hash": fingerprint,
                        "premise": premise,
                        "hypotheses": hypotheses,
                        "components": components,
                        "advisory_status": _advisory_status(components),
                        "calibration_status": "thresholds_not_human_calibrated",
                    }
                    handle.write(json.dumps(record, ensure_ascii=False) + "\n")
                    handle.flush()
                    saved[(review_id, claim["claim_id"])] = record

    status_counts = Counter(
        row.get("advisory_status", "unknown") for row in saved.values()
    )
    print("NLI advisory statuses:", dict(status_counts))
    for row in list(saved.values())[:max(0, show)]:
        print(
            f"  {row['review_id']} {row['claim_id']}: "
            f"{row['advisory_status']}"
        )
    print("Advisory NLI does not overwrite or verify EF claims.")
    return {
        "output_path": str(output_path),
        "records": len(saved),
        "pending_run": len(pending),
        "status_counts": dict(status_counts),
        "model_id": model_id,
        "device": scorer.device if scorer else (
            next(iter(saved.values())).get("device") if saved else None
        ),
    }


__all__ = [
    "DEFAULT_NLI_MODEL",
    "TransformersNliScorer",
    "run_nli",
]
