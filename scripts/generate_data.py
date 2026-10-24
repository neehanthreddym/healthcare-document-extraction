"""Reproducible synthetic scans and exact pre-render transcripts/labels."""
import argparse
import hashlib
import json
import random
import sys
from datetime import date, timedelta
from pathlib import Path

import pymupdf as fitz
from PIL import Image, ImageDraw, ImageFilter, ImageFont

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from intake.schema import FIELDS, HELD_OUT, LAYOUTS, PROVIDERS, SEED, SERVICES, STATUSES

ROOT = Path(__file__).resolve().parents[1]
LABELS = {
    "intake": ["Member ID", "Provider name", "Service date", "Procedure code", "Diagnosis code", "Requested service", "Decision status"],
    "authorization": ["Member number", "Provider", "Date of service", "CPT code", "ICD-10 code", "Service requested", "Authorization status"],
    "referral": ["Subscriber ID", "Referring provider", "Appointment date", "CPT", "Diagnosis code", "Treatment requested", "Decision"],
    "review": ["Subscriber number", "Ordering clinician", "Requested date", "Procedure code", "ICD10", "Requested service", "Decision status"],
}


def generate(out: Path, count: int, seed: int, font_path: str | None = None):
    if count < 50:
        raise ValueError("Use at least 50 documents per layout (200 total).")
    fonts = [font_path] if font_path else [
        "/System/Library/Fonts/Supplemental/Arial.ttf",
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "C:/Windows/Fonts/arial.ttf",
    ]
    font_file = next((Path(p) for p in fonts if p and Path(p).exists()), None)
    if font_file is None:
        raise ValueError("Pass --font with a local TrueType font path.")
    font = ImageFont.truetype(str(font_file), 25)
    title_font = ImageFont.truetype(str(font_file), 30)
    out.mkdir(parents=True, exist_ok=True)
    (out / "images").mkdir(exist_ok=True)
    rng = random.Random(seed)
    records = []
    for layout_index, layout in enumerate(LAYOUTS):
        for i in range(count):
            identifier = f"{layout}-{i:03d}"
            values = dict(zip(FIELDS, [
                f"SYN{layout_index + 1:02d}{i:06d}", rng.choice(PROVIDERS),
                (date(2026, 1, 1) + timedelta(days=rng.randrange(365))).isoformat(),
                rng.choice(["97110", "72148", "99213", "73560"]),
                rng.choice(["M54.5", "M25.561", "Z00.00", "R51.9"]),
                rng.choice(SERVICES), rng.choice(STATUSES),
            ]))
            # Independently sampled codes/services are format-validation examples,
            # not clinically coherent treatment or coverage recommendations.
            missing = FIELDS[(i // 13) % len(FIELDS)] if i % 13 == 0 else None
            conflict = "decision_status" if i % 17 == 0 else None
            order = list(range(len(FIELDS)))
            if layout == "authorization":
                order = [0, 1, 5, 3, 4, 2, 6]
            elif layout == "referral":
                order = [1, 0, 2, 5, 4, 3, 6]
            elif layout == "review":
                order = [6, 0, 2, 1, 4, 5, 3]
            lines = ["SYNTHETIC - NOT A REAL PATIENT", f"{layout.upper()} FORM", f"Document: {identifier}"]
            entities = []
            for index in order:
                field = FIELDS[index]
                value = "[blank]" if field == missing else values[field]
                prefix = LABELS[layout][index] + ": "
                start = len("\n".join(lines)) + 1 + len(prefix)
                lines.append(prefix + value)
                if field != missing:
                    entities.append({"field": field, "value": value, "start": start, "end": start + len(value)})
            if conflict:
                alternative = next(s for s in STATUSES if s != values[conflict])
                prefix = "Decision status: "
                start = len("\n".join(lines)) + 1 + len(prefix)
                lines.append(prefix + alternative)
                entities.append({"field": conflict, "value": alternative, "start": start, "end": start + len(alternative)})
            lines.append("SYNTHETIC DEMO - FOR SOFTWARE TESTING ONLY")
            transcript = "\n".join(lines)
            image = Image.new("RGB", (1200, 1050), "white")
            draw = ImageDraw.Draw(image)
            x, y = [(65, 70), (105, 55), (70, 90), (90, 65)][layout_index]
            gap = [66, 68, 64, 67][layout_index]
            if layout in {"authorization", "review"}:
                draw.rectangle((35, 30, 1165, 1015), outline="#999999", width=2)
            for j, line in enumerate(lines):
                if layout == "referral" and j >= 3:
                    draw.line((60, y + 42, 1130, y + 42), fill="#dddddd", width=1)
                draw.text((x, y), line, font=title_font if j == 1 else font, fill="#171717")
                y += gap
            # Fixed mild scan degradation; no OCR text layer is embedded.
            angle = rng.uniform(-0.55, 0.55)
            blur = rng.uniform(0.15, 0.65)
            image = image.rotate(angle, resample=Image.Resampling.BICUBIC, fillcolor="white").filter(ImageFilter.GaussianBlur(blur))
            image_path = out / "images" / f"{identifier}.png"
            image.save(image_path, optimize=True)
            records.append({
                "id": identifier, "synthetic": True, "layout": layout,
                "split": "evaluation" if layout == HELD_OUT else "development",
                "image": f"images/{identifier}.png", "text": transcript, "entities": entities,
                "missing_field": missing, "conflict_field": conflict,
                "scan": {"rotation_degrees": angle, "blur_radius": blur},
                "sha256": hashlib.sha256(image_path.read_bytes()).hexdigest(),
            })
    (out / "manifest.json").write_text(json.dumps({
        "synthetic": True, "seed": seed, "documents_per_layout": count,
        "held_out_layout": HELD_OUT, "split_method": "template holdout; no document-level random split",
        "font_name": font_file.name, "font_sha256": hashlib.sha256(font_file.read_bytes()).hexdigest(),
        "documents": records,
    }, indent=2) + "\n")
    # A scanned PDF exercises the identical raster OCR path used by uploads.
    with fitz.open() as pdf:
        page = pdf.new_page(width=600, height=525)
        page.insert_image(page.rect, filename=str(out / records[1]["image"]))
        pdf.save(out / "synthetic-demo.pdf")
    print(f"Generated {len(records)} synthetic documents at {out}; held out {count} {HELD_OUT} forms.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--out", type=Path, default=ROOT / "data")
    parser.add_argument("--per-layout", type=int, default=60)
    parser.add_argument("--seed", type=int, default=SEED)
    parser.add_argument("--font")
    args = parser.parse_args()
    generate(args.out, args.per_layout, args.seed, args.font)
