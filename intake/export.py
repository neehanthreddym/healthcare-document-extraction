"""Portable reviewed exports, preserving provenance and spreadsheet safety."""
import json
import pandas as pd
from intake.schema import FIELDS


def safe_cell(value):
    if isinstance(value, str) and value.lstrip().startswith(("=", "+", "-", "@", "\t", "\r")):
        return "'" + value
    return value


def csv_bytes(record):
    row = {"document": record["document"], "synthetic_demo": True, "review_needed": record["review_needed"]}
    for field in FIELDS:
        row[field] = record["reviewed_fields"][field]
        row[f"original_{field}"] = record["original_extracted_fields"][field]
        row[f"correction_{field}"] = record["reviewer_corrections"].get(field)
    row["evidence_json"] = json.dumps(record["evidence"])
    row["warnings_json"] = json.dumps(record["validation_warnings"])
    return pd.DataFrame([{k: safe_cell(v) for k, v in row.items()}]).to_csv(index=False).encode("utf-8")
