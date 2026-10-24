"""Load -> OCR -> normalize/tokenize -> local NER/rules -> validate/evidence.

No gold labels are read here. EntityRuler is deterministic NER, not a trained
clinical model. Values must be verbatim substrings of the normalized OCR.
"""
from __future__ import annotations

import io
import re
import unicodedata
from datetime import date
from functools import lru_cache

import pymupdf as fitz
import pytesseract
import spacy
from PIL import Image, ImageOps

from intake.schema import FIELDS, PROVIDERS, SERVICES, STATUSES

MAX_BYTES = 20 * 1024 * 1024
MAX_PAGES = 5
MAX_PIXELS = 20_000_000


def load_pages(data: bytes, filename: str) -> list[Image.Image]:
    if len(data) > MAX_BYTES:
        raise ValueError("Document exceeds the 20 MB local demo limit.")
    if filename.lower().endswith(".pdf"):
        with fitz.open(stream=data, filetype="pdf") as pdf:
            if pdf.needs_pass:
                raise ValueError("Password-protected PDFs are not supported.")
            if not 1 <= len(pdf) <= MAX_PAGES:
                raise ValueError("Please use a PDF containing 1–5 pages.")
            pages = []
            for page in pdf:
                scale = 180 / 72
                if page.rect.width * page.rect.height * scale**2 > MAX_PIXELS:
                    raise ValueError("PDF page exceeds the 20 megapixel demo limit.")
                pix = page.get_pixmap(matrix=fitz.Matrix(scale, scale), alpha=False)
                pages.append(Image.frombytes("RGB", (pix.width, pix.height), pix.samples))
            return pages
    with Image.open(io.BytesIO(data)) as image:
        if image.width * image.height > MAX_PIXELS:
            raise ValueError("Image exceeds the 20 megapixel demo limit.")
        if getattr(image, "n_frames", 1) != 1:
            raise ValueError("Multi-frame images are unsupported; upload a PDF instead.")
        return [ImageOps.exif_transpose(image).convert("RGB")]


def normalize(text: str) -> str:
    text = unicodedata.normalize("NFKC", text).replace("\r\n", "\n").replace("\r", "\n")
    # Preserve OCR page separators so evidence page numbers survive cleanup.
    return "\n\f\n".join(
        "\n".join(re.sub(r"[^\S\n]+", " ", line).strip() for line in page.split("\n")).strip()
        for page in text.split("\f")
    )


def ocr_pages(pages: list[Image.Image]) -> dict:
    texts = []
    for page in pages:
        gray = ImageOps.autocontrast(ImageOps.grayscale(page))
        texts.append(pytesseract.image_to_string(gray, lang="eng", config="--psm 6", timeout=45))
    return {"raw_text": "\n\f\n".join(texts), "page_texts": texts}


@lru_cache(maxsize=1)
def nlp_model():
    nlp = spacy.blank("en")
    ruler = nlp.add_pipe("entity_ruler", config={"phrase_matcher_attr": "LOWER"})
    patterns = []
    for field, values in [("provider_name", PROVIDERS), ("requested_service", SERVICES), ("decision_status", STATUSES)]:
        patterns.extend({"label": field, "pattern": value} for value in values)
    patterns.extend([
        {"label": "member_id", "pattern": [{"TEXT": {"REGEX": r"^SYN\d{8}$"}}]},
        {"label": "procedure_code", "pattern": [{"TEXT": {"REGEX": r"^\d{5}$"}}]},
        {"label": "diagnosis_code", "pattern": [{"TEXT": {"REGEX": r"^[A-Z]\d{2}(?:\.\d{1,4})?$"}}]},
    ])
    ruler.add_patterns(patterns)
    return nlp


LABELS = {
    "member_id": r"member\s*(?:id|number)|subscriber\s*(?:id|number)",
    "provider_name": r"provider(?:\s*name)?|ordering\s*clinician|referring\s*provider",
    "service_date": r"service\s*date|date\s*of\s*service|appointment\s*date|requested\s*date",
    "procedure_code": r"procedure\s*code|cpt(?:\s*code)?",
    "diagnosis_code": r"diagnosis\s*code|icd-?10(?:\s*code)?",
    "requested_service": r"requested\s*service|service\s*requested|treatment\s*requested",
    "decision_status": r"decision\s*status|authorization\s*status|decision",
}


def extract(text: str) -> dict:
    clean = normalize(text)
    doc = nlp_model()(clean)
    candidates = {field: [] for field in FIELDS}

    def add(field, start, end, method):
        value = clean[start:end].strip()
        start += len(clean[start:end]) - len(clean[start:end].lstrip())
        end = start + len(value)
        if not value or value.lower() in {"[blank]", "blank", "n/a", "-"}:
            return
        if any(x["value"].casefold() == value.casefold() for x in candidates[field]):
            return
        line_start = clean.rfind("\n", 0, start) + 1
        line_end = clean.find("\n", end)
        candidates[field].append({
            "value": value, "start": start, "end": end,
            "evidence": clean[line_start:line_end if line_end >= 0 else len(clean)],
            "page": clean[:start].count("\f") + 1, "method": method,
        })

    # Label rules preserve unknown providers/services without inventing an entity.
    for field, labels in LABELS.items():
        for match in re.finditer(rf"(?im)^(?:{labels})\s*[:=|]\s*([^\n\f]+)", clean):
            add(field, match.start(1), match.end(1), "label rule")
    for ent in doc.ents:
        # Label extraction takes priority; NER adds different values so conflicts surface.
        add(ent.label_, ent.start_char, ent.end_char, "spaCy EntityRuler")
    for match in re.finditer(r"\b\d{4}-\d{2}-\d{2}\b", clean):
        add("service_date", match.start(), match.end(), "date rule")

    fields, warnings = {}, []
    for field in FIELDS:
        options = candidates[field]
        fields[field] = options[0]["value"] if len(options) == 1 else None
        if not options:
            warnings.append(f"{field}: missing")
        elif len(options) > 1:
            warnings.append(f"{field}: conflicting values; reviewer must select a value")
        for option in options:
            issue = validate_value(field, option["value"])
            if issue:
                warnings.append(f"{field}: {issue}")
    return {
        "normalized_text": clean, "tokens": [t.text for t in doc if not t.is_space],
        "entities": [{"field": e.label_, "text": e.text, "start": e.start_char, "end": e.end_char} for e in doc.ents],
        "fields": fields, "candidates": candidates, "warnings": warnings,
        "review_needed": bool(warnings),
    }


def validate_value(field: str, value: str) -> str | None:
    if not value.strip():
        return "missing"
    formats = {"member_id": r"SYN\d{8}", "procedure_code": r"\d{5}", "diagnosis_code": r"[A-Z]\d{2}(?:\.\d{1,4})?"}
    if field in formats and not re.fullmatch(formats[field], value):
        return "unexpected format"
    if field == "service_date":
        try:
            date.fromisoformat(value)
        except ValueError:
            return "invalid ISO date (YYYY-MM-DD)"
    if field == "decision_status" and value.casefold() not in {s.casefold() for s in STATUSES}:
        return "unrecognized status"
    return None


def run_pipeline(data: bytes, filename: str) -> dict:
    pages = load_pages(data, filename)
    ocr = ocr_pages(pages)
    return {**ocr, **extract(ocr["raw_text"]), "page_count": len(pages)}


def reviewed_record(result: dict, corrections: dict[str, str], document_name: str) -> dict:
    """Explicit correction overlay, with original extraction and evidence retained."""
    reviewed = {f: corrections.get(f, result["fields"][f]) or None for f in FIELDS}
    warnings = [f"{f}: {issue}" for f, v in reviewed.items() if (issue := validate_value(f, v or ""))]
    unresolved = [f for f in FIELDS if len(result["candidates"][f]) > 1 and f not in corrections]
    warnings.extend(f"{f}: unresolved conflict" for f in unresolved)
    return {
        "synthetic_demo": True, "document": document_name,
        "original_extracted_fields": result["fields"], "reviewer_corrections": corrections,
        "reviewed_fields": reviewed, "evidence": result["candidates"],
        "raw_ocr_text": result["raw_text"], "normalized_ocr_text": result["normalized_text"],
        "validation_warnings": warnings, "review_needed": bool(warnings),
    }
