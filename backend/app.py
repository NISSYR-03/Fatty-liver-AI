"""
backend/app.py  —  HepatoAI Flask Server
Run: python app.py
Serves all API routes + frontend on http://localhost:5000
"""

import os, sys, traceback
from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

# Allow importing from project root
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from backend.models.predictor      import Predictor
from backend.utils.ocr_extractor   import extract_blood_values
from backend.utils.image_classifier import classify_ultrasound

FRONTEND_DIR = os.path.join(os.path.dirname(__file__), "..", "frontend")

app = Flask(__name__, static_folder=FRONTEND_DIR)
CORS(app)

UPLOAD_DIR = os.path.join(os.path.dirname(__file__), "uploads")
os.makedirs(UPLOAD_DIR, exist_ok=True)
app.config["MAX_CONTENT_LENGTH"] = 20 * 1024 * 1024   # 20 MB

predictor = Predictor()


# ─── SERVE FRONTEND ─────────────────────────────────────────────────
@app.route("/")
def serve_index():
    return send_from_directory(FRONTEND_DIR, "index.html")

@app.route("/<path:filename>")
def serve_frontend(filename):
    """Serve frontend HTML/CSS/JS files (only if not an /api route)"""
    filepath = os.path.join(FRONTEND_DIR, filename)
    if os.path.isfile(filepath):
        return send_from_directory(FRONTEND_DIR, filename)
    return jsonify({"error": "Not found"}), 404


# ─── HEALTH ─────────────────────────────────────────────────────────
@app.route("/api/health")
def health():
    return jsonify({"status":"ok","model_ready": predictor.is_ready()})


# ─── PREDICT (manual inputs) ────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json(force=True)
        result = predictor.predict(data)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── OCR ────────────────────────────────────────────────────────────
@app.route("/api/ocr", methods=["POST"])
def ocr():
    try:
        if "file" not in request.files:
            return jsonify({"error":"No file"}), 400
        f    = request.files["file"]
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        res  = extract_blood_values(path)
        os.remove(path)
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── IMAGE CNN ──────────────────────────────────────────────────────
@app.route("/api/image-predict", methods=["POST"])
def image_predict():
    try:
        if "file" not in request.files:
            return jsonify({"error":"No file"}), 400
        f    = request.files["file"]
        path = os.path.join(UPLOAD_DIR, f.filename)
        f.save(path)
        res  = classify_ultrasound(path)
        os.remove(path)
        return jsonify(res)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ─── CHATBOT ────────────────────────────────────────────────────────
@app.route("/api/chat", methods=["POST"])
def chat():
    try:
        body    = request.get_json(force=True)
        msg     = body.get("message","").strip()
        session = body.get("session_data", {})
        reply, session, prediction = process_chat(msg.lower(), session)
        return jsonify({"reply":reply,"session_data":session,
                        "prediction":prediction})
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500


# ────────────────────────────────────────────────────────────────────
#  CHATBOT LOGIC
# ────────────────────────────────────────────────────────────────────
FLOW = [
    ("age",           "What is your **age** in years?"),
    ("gender",        "What is your **gender**? (type: male or female)"),
    ("bmi",           "What is your **BMI**? (weight kg ÷ height m²)"),
    ("alt",           "What is your **ALT / SGPT** level? (U/L)"),
    ("ast",           "What is your **AST / SGOT** level? (U/L)"),
    ("bilirubin",     "What is your **Total Bilirubin**? (mg/dL)"),
    ("albumin",       "What is your **Albumin** level? (g/dL)"),
    ("triglycerides", "What is your **Triglycerides**? (mg/dL)"),
    ("glucose",       "What is your **Fasting Glucose**? (mg/dL)"),
]

GREETINGS = {"hi","hello","hey","start","begin","helo","hii","yo","namaste"}

EXPLAIN = {
    "alt":       "ALT (SGPT) is a liver enzyme. Normal: 7–56 U/L. High ALT means liver inflammation.",
    "ast":       "AST (SGOT) is a liver enzyme. Normal: 10–40 U/L. High AST suggests liver stress.",
    "bilirubin": "Bilirubin is produced when RBCs break down. Normal total: 0.2–1.2 mg/dL. High = jaundice risk.",
    "albumin":   "Albumin is a protein made by your liver. Normal: 3.5–5.0 g/dL. Low = liver dysfunction.",
    "bmi":       "BMI = weight(kg) / height(m)². Normal 18.5–24.9. ≥30 = obese. High BMI = fatty liver risk.",
    "triglycerides":"Triglycerides are blood fats. Normal <150 mg/dL. High = metabolic syndrome + liver risk.",
    "glucose":   "Fasting blood glucose. Normal 70–100 mg/dL. High glucose = insulin resistance = fatty liver.",
    "fatty liver":"Fatty liver (hepatic steatosis) = fat accumulation in liver. Can progress to cirrhosis if untreated.",
    "sgpt":      "SGPT = ALT. Liver enzyme. Normal: 7–56 U/L.",
    "sgot":      "SGOT = AST. Liver enzyme. Normal: 10–40 U/L.",
}


def process_chat(msg, session):
    prediction = None

    # Greeting / first message (session has no 'started' flag yet)
    if msg in GREETINGS or "started" not in session:
        session = {"started": True}
        return (
            "👋 Hello! I'm your **Liver Health AI Assistant**.\n\n"
            "I'll assess your fatty liver risk by asking 9 quick questions.\n\n"
            "Let's start — **what is your age in years?**",
            session, None
        )

    # Restart
    if any(w in msg for w in ["restart","reset","new","again","start over"]):
        return ("Sure! Starting fresh. 🔄\n\nWhat is your **age** in years?",
                {"started": True}, None)

    # Explain keywords
    for kw, exp in EXPLAIN.items():
        if kw in msg:
            # Don't interrupt flow — append explanation then re-ask
            missing = [f for f,_ in FLOW if f not in session]
            if missing:
                nf = missing[0]
                nq = next(q for f,q in FLOW if f==nf)
                return (f"📖 {exp}\n\n{nq}", session, None)
            return (f"📖 {exp}", session, None)

    # Collect values
    missing = [f for f,_ in FLOW if f not in session]
    if missing:
        cur_field = missing[0]
        try:
            if cur_field == "gender":
                if "male" in msg or msg == "m":
                    session["gender"] = 1
                elif "female" in msg or msg == "f":
                    session["gender"] = 0
                else:
                    return ("Please reply **male** or **female**.", session, None)
            else:
                # Extract first number from message
                nums = [float(t) for t in msg.replace(",",".").split()
                        if t.replace(".","").isdigit()]
                if not nums:
                    # Try stripping non-numeric chars
                    import re
                    m = re.search(r"(\d+\.?\d*)", msg)
                    if not m:
                        raise ValueError
                    nums = [float(m.group(1))]
                session[cur_field] = nums[0]
        except (ValueError, IndexError):
            q = next(q for f,q in FLOW if f==cur_field)
            return (f"I didn't catch that. {q}", session, None)

        # Check remaining
        remaining = [f for f,_ in FLOW if f not in session]
        if remaining:
            nq = next(q for f,q in FLOW if f==remaining[0])
            return (f"✅ Got it!\n\n{nq}", session, None)
        else:
            # All collected — run prediction
            pred_data = {k:v for k,v in session.items() if k != "started"}
            result    = predictor.predict(pred_data)
            risk      = result["risk_label"]
            probas    = result["probabilities"]
            shap_d    = result.get("shap_contributions", {})
            rec       = result["recommendation"]
            prediction = result

            emoji = {"Low":"🟢","Medium":"🟡","High":"🔴"}.get(risk, "⚪")
            conf  = round(max(probas)*100, 1)

            if shap_d:
                top3 = sorted(shap_d.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
                facts = "\n".join(
                    f"  • **{k.upper()}**: {'↑ raises' if v>0 else '↓ lowers'} risk (score: {abs(v):.3f})"
                    for k, v in top3
                )
                reply = (
                    f"{emoji} **Risk Level: {risk}**  |  Confidence: {conf}%\n\n"
                    f"🔍 **Key contributors:**\n{facts}\n\n"
                    f"💡 **Recommendation:**\n{rec}\n\n"
                    "---\nType **hi** to start a new assessment or ask me about any lab value."
                )
            else:
                reply = (
                    f"{emoji} **Risk Level: {risk}**  |  Confidence: {conf}%\n\n"
                    f"💡 **Recommendation:**\n{rec}\n\n"
                    "Note: SHAP feature explanations are unavailable in this environment.\n"
                    "Type **hi** to start a new assessment or ask me about any lab value."
                )
            session = {"started": True}
            return (reply, session, prediction)

    return (
        "I'm here to help with liver health! Type **hi** to start an assessment, "
        "or ask me about: ALT, AST, BMI, Bilirubin, Albumin, Fatty Liver, etc.",
        session, None
    )


if __name__ == "__main__":
    print("\n🚀 HepatoAI backend → http://localhost:5000\n")
    app.run(debug=True, host="0.0.0.0", port=5000)
