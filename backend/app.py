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
    return jsonify({
        "status": "ok",
        "model_ready": predictor.is_ready(),
        "model_accuracy": predictor.meta.get("accuracy", "N/A"),
        "model_auc": predictor.meta.get("auc", "N/A")
    })


# ─── PREDICT (manual inputs) ────────────────────────────────────────
@app.route("/api/predict", methods=["POST"])
def predict():
    try:
        data   = request.get_json(force=True)
        result = predictor.predict(data)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": "Backend Error", "details": str(e)}), 500


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
        return jsonify({"error": "Backend Error", "details": str(e)}), 500


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
        return jsonify({"error": "Backend Error", "details": str(e)}), 500


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
import os
import json
import re
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()
API_KEY = os.environ.get("GEMINI_API_KEY", "")
if API_KEY:
    genai.configure(api_key=API_KEY)
else:
    print("WARNING: GEMINI_API_KEY not found in environment.")

MODEL_NAME = "gemini-2.5-flash"
SYSTEM_PROMPT = """
You are HepatoAI, an expert Liver Health AI Assistant. Your goal is to provide deep, accurate, and comprehensive knowledge about liver health, fatty liver disease, medical tests (like ALT, AST, Bilirubin, Albumin), and general wellness.

When the user asks a question, provide a detailed, educational, and highly informative answer. Use markdown for readability (bullet points, bold text, etc.).
DO NOT repeatedly ask the user for their 9 parameters (age, gender, BMI, lab values). Just answer their questions directly and provide deep knowledge.

Your response MUST be a valid JSON object matching this schema exactly:
{
  "reply": "your conversational response formatted in markdown",
  "extracted": {
    "age": number (if found),
    "gender": "male" or "female" (if found),
    "bmi": number (if found),
    "alt": number (if found),
    "ast": number (if found),
    "bilirubin": number (if found),
    "albumin": number (if found),
    "triglycerides": number (if found),
    "glucose": number (if found)
  }
}
Do NOT wrap the JSON in markdown code blocks like ```json ... ```. Output raw JSON only.
If the user happens to provide any values, parse them into 'extracted', but DO NOT prompt them for missing ones.
"""

def process_chat(msg, session):
    prediction = None

    if "started" not in session:
        session = {"started": True}
    
    # Check if we should restart
    if msg.lower() in ["restart", "reset", "new", "again", "start over"]:
        session = {"started": True}
        msg = "I want to start a new assessment."

    # Identify missing values
    fields = ["age", "gender", "bmi", "alt", "ast", "bilirubin", "albumin", "triglycerides", "glucose"]
    missing = [f for f in fields if f not in session]
    
    current_state = {k: v for k, v in session.items() if k in fields}
    
    # Prompt Gemini
    prompt = (
        f"{SYSTEM_PROMPT}\n\n"
        f"Currently known values from user (if any): {json.dumps(current_state)}\n\n"
        f"User message: {msg}"
    )

    reply_text = "I'm having trouble connecting to my AI brain right now."
    try:
        if not API_KEY:
            raise ValueError("GEMINI_API_KEY is not set in your environment. Please add it to your .env file.")
            
        model = genai.GenerativeModel(MODEL_NAME)
        response = model.generate_content(prompt)
        
        # Clean up response text in case Gemini wraps it in markdown code blocks
        raw_text = response.text.strip()
        if raw_text.startswith("```json"):
            raw_text = raw_text[7:]
        if raw_text.endswith("```"):
            raw_text = raw_text[:-3]
        raw_text = raw_text.strip()
            
        # Extract JSON using regex to handle extra conversational text
        json_match = re.search(r'\{.*\}', raw_text, re.DOTALL)
        if not json_match:
            raise ValueError("No valid JSON found in AI response.")
        
        data = json.loads(json_match.group(0))
        reply_text = data.get("reply", "I didn't quite get that.")
        extracted = data.get("extracted", {})
        
        # Safely update session with newly extracted variables
        for k, v in extracted.items():
            if k in fields and v is not None:
                if k == "gender":
                    # Convert to our model's format: 1 for male, 0 for female
                    val = str(v).lower()
                    if "male" in val and "female" not in val:
                        session[k] = 1
                    elif "female" in val:
                        session[k] = 0
                else:
                    try:
                        session[k] = float(v)
                    except ValueError:
                        pass
                        
    except Exception as e:
        import traceback
        traceback.print_exc()
        return (f"⚠️ AI Error: {str(e)}", session, None)

    # Check if all fields are collected now
    remaining = [f for f in fields if f not in session]
    if not remaining:
        # All collected — run prediction
        pred_data = {k:v for k,v in session.items() if k in fields}
        result    = predictor.predict(pred_data)
        risk      = result["risk_label"]
        probas    = result["probabilities"]
        shap_d    = result.get("shap_contributions", {})
        rec       = result["recommendation"]
        prediction = result

        emoji = {"Low":"🟢","Medium":"🟡","High":"🔴"}.get(risk, "⚪")
        conf  = round(max(probas)*100, 1)

        if self.explainer is None:
            try:
                # Use a smaller background dataset for faster SHAP
                self.explainer = shap.TreeExplainer(self.model)
            except:
                self.explainer = None

        if shap_d:
            top3 = sorted(shap_d.items(), key=lambda x: abs(x[1]), reverse=True)[:3]
            facts = "\n".join(
                f"  • **{k.upper()}**: {'↑ raises' if v>0 else '↓ lowers'} risk (score: {abs(v):.3f})"
                for k, v in top3
            )
            reply_text += (
                f"\n\n---\n"
                f"{emoji} **Risk Level: {risk}**  |  Confidence: {conf}%\n\n"
                f"🔍 **Key contributors:**\n{facts}\n\n"
                f"💡 **Recommendation:**\n{rec}\n\n"
                "Type **hi** or **restart** to start a new assessment."
            )
        else:
            reply_text += (
                f"\n\n---\n"
                f"{emoji} **Risk Level: {risk}**  |  Confidence: {conf}%\n\n"
                f"💡 **Recommendation:**\n{rec}\n\n"
                "Type **hi** or **restart** to start a new assessment."
            )
        session = {"started": True} # reset session for the next assessment

    return (reply_text, session, prediction)


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"\n🚀 HepatoAI backend starting on port {port} → http://0.0.0.0:{port}\n")
    app.run(debug=True, host="0.0.0.0", port=port)
