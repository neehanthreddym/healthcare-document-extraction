"""Frozen-rule evaluation; gold labels are used only to score OCR/extraction."""
import argparse
import hashlib
import json
import platform
import re
import sys
from concurrent.futures import ThreadPoolExecutor
from importlib.metadata import version
from pathlib import Path

import pytesseract
from sklearn.metrics import precision_recall_fscore_support

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intake.pipeline import extract, load_pages, normalize, ocr_pages
from intake.schema import FIELDS

ROOT = Path(__file__).resolve().parents[1]


def edit_distance(a, b):
    previous = list(range(len(b) + 1))
    for i, left in enumerate(a, 1):
        current = [i]
        for j, right in enumerate(b, 1):
            current.append(min(current[-1] + 1, previous[j] + 1, previous[j - 1] + (left != right)))
        previous = current
    return previous[-1]


def canonical(value):
    return re.sub(r"\s+", " ", value).strip().casefold()


def scores(truth, predictions):
    union = sorted(truth | predictions)
    if not union:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "support": 0, "predicted": 0}
    precision, recall, f1, _ = precision_recall_fscore_support(
        [int(x in truth) for x in union], [int(x in predictions) for x in union],
        average="binary", zero_division=0,
    )
    return {"precision": precision, "recall": recall, "f1": f1, "support": len(truth), "predicted": len(predictions)}


def evaluate(data_dir, out, split="evaluation", workers=4):
    manifest_path = data_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    documents = [d for d in manifest["documents"] if d["split"] == split]
    if not documents:
        raise ValueError("No documents in requested split")
    # OCR is parallel; the shared spaCy pipeline is applied sequentially below.
    def scan(d):
        image = data_dir / d["image"]
        if hashlib.sha256(image.read_bytes()).hexdigest() != d["sha256"]:
            raise ValueError(f"Dataset checksum mismatch: {d['id']}")
        return ocr_pages(load_pages(image.read_bytes(), image.name))

    truth, predictions, details = set(), set(), []
    edits = words = missing = selected_missing = traceable = predicted_count = 0
    with ThreadPoolExecutor(max_workers=workers) as pool:
        for index, (document, ocr) in enumerate(zip(documents, pool.map(scan, documents)), 1):
            result = extract(ocr["raw_text"])
            # WER: whitespace tokens after NFKC/whitespace cleanup; case and punctuation retained.
            gold_tokens = normalize(document["text"]).split()
            actual_tokens = result["normalized_text"].split()
            errors = edit_distance(gold_tokens, actual_tokens)
            edits += errors
            words += len(gold_tokens)
            doc_truth = {(document["id"], e["field"], canonical(e["value"])) for e in document["entities"]}
            doc_pred = {(document["id"], f, canonical(c["value"])) for f in FIELDS for c in result["candidates"][f]}
            truth |= doc_truth
            predictions |= doc_pred
            for field in FIELDS:
                missing += not result["candidates"][field]
                selected_missing += result["fields"][field] is None
                for candidate in result["candidates"][field]:
                    predicted_count += 1
                    traceable += result["normalized_text"][candidate["start"]:candidate["end"]] == candidate["value"] and candidate["value"] in candidate["evidence"]
            details.append({"id": document["id"], "wer": errors / len(gold_tokens),
                            "false_positive": sorted(doc_pred - doc_truth), "false_negative": sorted(doc_truth - doc_pred),
                            "raw_ocr_text": ocr["raw_text"], "warnings": result["warnings"]})
            print(f"{index}/{len(documents)} {document['id']} WER={errors / len(gold_tokens):.3f}", flush=True)
    summary = {
        "synthetic_only": True, "sample_count": len(documents), "split": split,
        "layouts": sorted({d["layout"] for d in documents}), "split_method": manifest["split_method"],
        "seed": manifest["seed"], "manifest_sha256": hashlib.sha256(manifest_path.read_bytes()).hexdigest(),
        "ocr_word_error_rate": edits / words, "word_edits": edits, "reference_words": words,
        "entity_overall": scores(truth, predictions),
        "entity_by_field": {f: scores({x for x in truth if x[1] == f}, {x for x in predictions if x[1] == f}) for f in FIELDS},
        "missing_field_rate": missing / (len(documents) * len(FIELDS)),
        "unselected_field_rate_including_conflicts": selected_missing / (len(documents) * len(FIELDS)),
        "ground_truth_missing_field_rate": sum(not any(e["field"] == f for e in d["entities"]) for d in documents for f in FIELDS) / (len(documents) * len(FIELDS)),
        "evidence_coverage": traceable / predicted_count if predicted_count else 0,
        "predicted_values": predicted_count, "traceable_values": traceable,
        "software": {p: version(p) for p in ["spacy", "pytesseract", "PyMuPDF", "Pillow", "scikit-learn"]},
        "python": platform.python_version(), "tesseract": str(pytesseract.get_tesseract_version()),
        "pipeline_sha256": hashlib.sha256((ROOT / "intake/pipeline.py").read_bytes()).hexdigest(),
    }
    out.mkdir(parents=True, exist_ok=True)
    (out / f"{split}-summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    (out / f"{split}-details.json").write_text(json.dumps(details, indent=2) + "\n")
    print(json.dumps(summary, indent=2))
    return summary


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--data", type=Path, default=ROOT / "data")
    parser.add_argument("--out", type=Path, default=ROOT / "outputs")
    parser.add_argument("--split", choices=["development", "evaluation"], default="evaluation")
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    evaluate(args.data, args.out, args.split, args.workers)
