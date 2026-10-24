import io

import pymupdf as fitz
import pytest
from PIL import Image

from intake.export import csv_bytes
from intake.pipeline import extract, load_pages, reviewed_record, validate_value
from scripts.evaluate import edit_distance, scores


def test_conflicts_missing_and_evidence():
    result = extract("Member ID: SYN01000001\nDecision status: Approved\nDecision status: Denied\nRequested service: [blank]")
    assert result["fields"]["decision_status"] is None
    assert result["candidates"]["requested_service"] == []
    assert result["review_needed"]
    for options in result["candidates"].values():
        for c in options:
            assert result["normalized_text"][c["start"]:c["end"]] == c["value"]
    original = result["fields"].copy()
    record = reviewed_record({**result, "raw_text": "raw"}, {"decision_status": "Pending"}, "synthetic.png")
    assert result["fields"] == original
    assert record["reviewed_fields"]["decision_status"] == "Pending"
    assert record["original_extracted_fields"]["decision_status"] is None
    assert record["review_needed"]  # Other fields remain missing.


def test_unknown_provider_is_preserved_and_invalid_date_flagged():
    result = extract("Provider name: Dr. Synthetic Unknown\nService date: 2026-02-30")
    assert result["fields"]["provider_name"] == "Dr. Synthetic Unknown"
    assert any("invalid ISO date" in w for w in result["warnings"])
    assert validate_value("procedure_code", "ABC")


def test_evidence_page_survives_normalization():
    result = extract("Member ID: SYN01000001\n\f\nProvider name: Dr. Casey Park")
    c = result["candidates"]["provider_name"][0]
    assert c["page"] == 2
    assert result["normalized_text"][c["start"]:c["end"]] == "Dr. Casey Park"


def test_pdf_and_image_loading_limits():
    buffer = io.BytesIO()
    Image.new("RGB", (200, 200), "white").save(buffer, format="PNG")
    assert len(load_pages(buffer.getvalue(), "test.png")) == 1
    with fitz.open() as doc:
        doc.new_page()
        doc.new_page()
        assert len(load_pages(doc.tobytes(), "test.pdf")) == 2
        for _ in range(4):
            doc.new_page()
        with pytest.raises(ValueError, match="1–5"):
            load_pages(doc.tobytes(), "test.pdf")
    with pytest.raises(Exception):
        load_pages(b"not an image", "broken.png")


def test_metrics_count_false_positives_and_deletions():
    assert edit_distance("a b c".split(), "a x".split()) == 2
    assert scores({("d", "f", "a")}, {("d", "f", "b")})["f1"] == 0
    assert scores({("d", "f", "a")}, {("d", "f", "a")})["f1"] == 1


def test_csv_formula_escape():
    result = {**extract("Member ID: SYN01000001"), "raw_text": "raw"}
    record = reviewed_record(result, {"provider_name": "=1+1"}, "=evil.png")
    exported = csv_bytes(record).decode()
    assert "'=1+1" in exported and "'=evil.png" in exported
