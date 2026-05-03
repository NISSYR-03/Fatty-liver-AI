"""
backend/utils/ocr_extractor.py
Extract blood test values from PDF/image using Tesseract OCR.
"""

import re, os
from PIL import Image

try:
    import pytesseract
    OCR_OK = True
except ImportError:
    OCR_OK = False

try:
    import fitz          # PyMuPDF
    PDF_OK = True
except ImportError:
    PDF_OK = False

# Field → regex patterns (case-insensitive on uppercased text)
PATTERNS = {
    "alt":           [r"(?:ALT|SGPT|ALANINE[^0-9]*(?:AMINO)?TRANSFERASE)[^\d]{0,20}(\d+\.?\d*)"],
    "ast":           [r"(?:AST|SGOT|ASPARTATE[^0-9]*(?:AMINO)?TRANSFERASE)[^\d]{0,20}(\d+\.?\d*)"],
    "bilirubin":     [r"(?:TOTAL\s*BILIRUBIN|T\.?\s*BILI(?:RUBIN)?)[^\d]{0,20}(\d+\.?\d*)"],
    "albumin":       [r"(?:ALBUMIN|ALB)[^\d]{0,15}(\d+\.?\d*)"],
    "triglycerides": [r"(?:TRIGLYCERIDES?|TRIGS?|TG)[^\d]{0,15}(\d+\.?\d*)"],
    "glucose":       [r"(?:GLUCOSE|BLOOD\s*SUGAR|FBS|FASTING\s*BLOOD)[^\d]{0,20}(\d+\.?\d*)"],
}


def _pdf_to_text(path):
    if not PDF_OK:
        raise ImportError("Install PyMuPDF: pip install PyMuPDF")
    doc  = fitz.open(path)
    text = "".join(p.get_text() for p in doc)
    doc.close()
    return text


def _image_to_text(path):
    if not OCR_OK:
        raise ImportError("Install pytesseract + tesseract-ocr")
    img  = Image.open(path)
    return pytesseract.image_to_string(img, config="--psm 6 --oem 3")


def _gemini_ocr(path, client, model_name):
    """Use Gemini to extract raw text from a blood report image/PDF."""
    try:
        from PIL import Image
        img = Image.open(path)
        prompt = "Extract all text from this blood test report exactly as it appears. Output raw text only."
        response = client.models.generate_content(model=model_name, contents=[prompt, img])
        return response.text
    except Exception as e:
        print(f"⚠️ Gemini OCR fallback failed: {e}")
        return ""


def extract_blood_values(file_path: str, client=None, model_name="gemini-2.5-flash") -> dict:
    ext  = os.path.splitext(file_path)[1].lower()
    text = ""
    
    if ext == ".pdf":
        try:
            text = _pdf_to_text(file_path)
        except Exception as e:
            print(f"⚠️ PDF extraction failed: {e}")
            if client: text = _gemini_ocr(file_path, client, model_name)
    else:
        # For images, try local Tesseract first, then Gemini
        try:
            text = _image_to_text(file_path)
        except Exception as e:
            print(f"⚠️ Local OCR failed: {e}")
            if client: 
                print("🔄 Falling back to Gemini OCR for image...")
                text = _gemini_ocr(file_path, client, model_name)
    
    tu   = text.upper()

    found = {}
    for field, pats in PATTERNS.items():
        for pat in pats:
            m = re.search(pat, tu)
            if m:
                try:
                    found[field] = float(m.group(1))
                    break
                except (ValueError, IndexError):
                    continue

    return {
        "values":         found,
        "raw_text":       text[:3000],
        "fields_found":   list(found.keys()),
        "fields_missing": [f for f in PATTERNS if f not in found],
    }
