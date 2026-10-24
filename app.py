"""Run with: streamlit run app.py"""
import hashlib
import json
from pathlib import Path

import pandas as pd
import pytesseract
import streamlit as st
from PIL import Image, UnidentifiedImageError

from intake.export import csv_bytes
from intake.pipeline import load_pages, reviewed_record, run_pipeline
from intake.schema import FIELDS

ROOT = Path(__file__).resolve().parent
st.set_page_config(page_title="Healthcare Document Extraction", page_icon="📄", layout="wide")
st.markdown("""<style>
.block-container {padding-top: 2rem; max-width: 1500px;}
div[data-testid="stMetric"] {background:#eff6f6; border-radius:10px; padding:14px;}
</style>""", unsafe_allow_html=True)
st.caption("SYNTHETIC DATA LAB · LOCAL OCR · HUMAN REVIEW")
st.title("Healthcare Document Extraction and Intake Analytics")
st.write("Turn a scanned intake or authorization form into an evidence-linked record.")
st.info("Portfolio demonstration • Synthetic forms only • Not a clinical or coverage decision tool. Do not upload real patient information.")


@st.cache_data(show_spinner=False)
def analyze(data, filename):
    return run_pipeline(data, filename)


manifest_path = ROOT / "data/manifest.json"
if not manifest_path.exists():
    st.error("Dataset missing. Run: python scripts/generate_data.py")
    st.stop()
manifest = json.loads(manifest_path.read_text())
documents = manifest["documents"]
with st.expander("Synthetic dataset and held-out evaluation"):
    counts = pd.DataFrame(documents).groupby(["layout", "split"]).size().reset_index(name="documents")
    st.write(f"{len(documents)} labeled synthetic scans · seed {manifest['seed']} · split by template")
    st.dataframe(counts, hide_index=True)
    summary_path = ROOT / "data/evaluation-summary.json"
    if summary_path.exists():
        summary = json.loads(summary_path.read_text())
        a, b, c = st.columns(3)
        a.metric("Held-out OCR WER", f"{summary['ocr_word_error_rate']:.2%}")
        b.metric("Entity micro F1", f"{summary['entity_overall']['f1']:.3f}")
        c.metric("Traceable candidates", f"{summary['evidence_coverage']:.1%}")
        st.caption(f"Measured on {summary['sample_count']} forms from the held-out review layout. Synthetic results do not establish real-world healthcare accuracy.")
        st.dataframe(pd.DataFrame(summary["entity_by_field"]).T, width="stretch")
with st.sidebar:
    st.header("1 · Choose a document")
    source = st.radio("Document source", ["Bundled sample", "Upload synthetic document"])
    if source == "Bundled sample":
        layout = st.selectbox("Layout", sorted({d["layout"] for d in documents}))
        options = [d for d in documents if d["layout"] == layout]
        chosen = st.selectbox("Sample", options, index=1, format_func=lambda d: d["id"] + (" · missing field" if d["missing_field"] else "") + (" · conflicting status" if d["conflict_field"] else ""))
        filename = Path(chosen["image"]).name
        data = (ROOT / "data" / chosen["image"]).read_bytes()
        st.caption(f"{chosen['split'].title()} split · All data is synthetic")
    else:
        confirmed = st.checkbox("I confirm this document contains synthetic data only.")
        upload = st.file_uploader("PDF or image · 20 MB / 5 pages maximum", type=["pdf", "png", "jpg", "jpeg", "tif", "tiff"])
        if not confirmed or upload is None:
            st.caption("Confirm synthetic content and choose a file to begin.")
            st.stop()
        filename, data = upload.name, upload.getvalue()
    st.divider()
    st.caption("Runs on this computer. No external OCR or language-model service is called. Uploaded files are kept in memory; do not use sensitive data.")
    st.download_button("Download synthetic upload fixture", (ROOT / "data/synthetic-demo.pdf").read_bytes(), "synthetic-demo.pdf", "application/pdf")

try:
    with st.spinner("Reading pages, running Tesseract and extracting fields…"):
        result = analyze(data, filename)
        pages = load_pages(data, filename)
except (ValueError, RuntimeError, OSError, UnidentifiedImageError, Image.DecompressionBombError, pytesseract.TesseractError, pytesseract.TesseractNotFoundError) as exc:
    st.error(f"Could not process this document: {exc}")
    st.stop()

doc_key = hashlib.sha256(data).hexdigest()
if st.session_state.get("active_document") != doc_key:
    for key in list(st.session_state):
        if key.startswith("review_"):
            del st.session_state[key]
    st.session_state.active_document = doc_key

metrics = st.columns(4)
metrics[0].metric("Pages", result["page_count"])
metrics[1].metric("Fields found", f"{sum(bool(v) for v in result['candidates'].values())} / {len(FIELDS)}")
metrics[2].metric("OCR tokens", len(result["tokens"]))
metrics[3].metric("Extraction status", "Review needed" if result["review_needed"] else "Ready for review")

left, right = st.columns([1, 1.15], gap="large")
with left:
    st.subheader("2 · Inspect the scan")
    page_index = st.selectbox("Page", range(len(pages)), format_func=lambda i: f"Page {i + 1}")
    st.image(pages[page_index], caption=f"{filename} · page {page_index + 1}", width="stretch")
    with st.expander("OCR text and processing stages"):
        raw_tab, clean_tab, tokens_tab = st.tabs(["Raw Tesseract output", "Normalized text", "Tokens & NER"])
        raw_tab.code(result["raw_text"], language=None)
        clean_tab.code(result["normalized_text"], language=None)
        tokens_tab.write(result["tokens"])
        tokens_tab.json(result["entities"])

with right:
    st.subheader("3 · Review fields and evidence")
    if result["warnings"]:
        for warning in result["warnings"]:
            st.warning(warning)
    st.caption("Evidence quotes refer to normalized OCR text; offsets are character positions in that text. Conflicts have no selected value.")
    rows = [{"Field": f, "Candidate": c["value"], "Supporting text": c["evidence"], "Page": c["page"], "Start": c["start"], "End": c["end"], "Method": c["method"]} for f in FIELDS for c in result["candidates"][f]]
    st.dataframe(pd.DataFrame(rows), hide_index=True, width="stretch")
    st.caption("Enter corrections after checking the scan. Blank fields stay missing. Saving preserves the original extraction and evidence.")
    with st.form("review_form"):
        entered = {field: st.text_input(field.replace("_", " ").title(), value=result["fields"][field] or "", key=f"review_{field}") for field in FIELDS}
        submitted = st.form_submit_button("Save reviewed record", type="primary")
    if submitted:
        corrections = {f: entered[f].strip() for f in FIELDS if entered[f].strip() != (result["fields"][f] or "")}
        st.session_state.review_saved = reviewed_record(result, corrections, filename)
    if "review_saved" in st.session_state:
        record = st.session_state.review_saved
        st.subheader("4 · Export saved review")
        if record["review_needed"]:
            st.warning("Saved with unresolved issues. Exports retain the review-needed status.")
            st.write(record["validation_warnings"])
        else:
            st.success("Review saved. Format checks passed; this is not clinical validation.")
        st.caption("Downloads contain the last saved review. Save again after making further changes.")
        a, b = st.columns(2)
        a.download_button("Download reviewed JSON", json.dumps(record, indent=2), "reviewed-record.json", "application/json")
        b.download_button("Download reviewed CSV", csv_bytes(record), "reviewed-record.csv", "text/csv")
        with st.expander("Saved record and correction audit"):
            st.json(record)
