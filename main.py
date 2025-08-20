import os
import json
import traceback
import re
import whisperx
from deep_translator import GoogleTranslator
from sentence_transformers import SentenceTransformer, util
from flask import Flask, request, jsonify, send_file

# Import from utils.py
from utils import (
    detect_phishing_sentences, hash_audio_file, calculate_rating,
    split_sentences, phrases as PHRASES_FLAT
)

# Import from database.py
from database import (
    get_existing_rating, save_audio_file, save_rating,
    get_full_record, get_audio_file
)

# ==== SETTINGS ====
DEVICE = "cpu"  # change to "cuda" if you have GPU
BATCH_SIZE = 4
COMPUTE_TYPE = "int8"  # float32 for better accuracy but slower on CPU
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# ==== FLASK APP ====
app = Flask(__name__)

# ==========================================
# MODELS
# ==========================================
print("Loading WhisperX model...")
whisper_model = whisperx.load_model("large-v3", device=DEVICE, compute_type=COMPUTE_TYPE)

print("Loading embedding model...")
embedder = SentenceTransformer('all-MiniLM-L6-v2')

# Embed the flat phrase list once for semantic detection
PHRASE_EMBEDDINGS = embedder.encode(PHRASES_FLAT, convert_to_tensor=True)

# ==========================================
# LOADERS
# ==========================================
def load_phishing_words_for_detection():
    """
    Existing detector data (kept separate). This continues to use
    phishing_words.json if you rely on it elsewhere. We do not modify this file.
    """
    path = os.path.join(os.path.dirname(__file__), "phishing_words.json")
    if not os.path.exists(path):
        return {"generic": []}
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        if isinstance(data, dict):
            return data
        return {"generic": data}

def load_reason_categories():
    """
    Loads categories and reasons from categories.json.
    Expected schema:
    {
      "bank": {
        "keywords": ["bank account", "routing number", ...],
        "reason": "mentions sensitive banking details"
      },
      "otp": {
        "keywords": ["otp", "verification code", ...],
        "reason": "requests OTP or verification codes"
      },
      ...
    }
    """
    path = os.path.join(os.path.dirname(__file__), "categories.json")
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
        # minimal validation
        if not isinstance(data, dict):
            raise ValueError("categories.json must be a JSON object mapping category -> {keywords, reason}")
        for cat, cfg in data.items():
            if not isinstance(cfg, dict) or "keywords" not in cfg or "reason" not in cfg:
                raise ValueError(f"Category '{cat}' must have 'keywords' (list) and 'reason' (string).")
            if not isinstance(cfg["keywords"], list) or not isinstance(cfg["reason"], str):
                raise ValueError(f"Category '{cat}' has invalid types for 'keywords' or 'reason'.")
        return data

# Keep detector words (not used for reason; separate concern)
PHISHING_DETECTOR_DATA = load_phishing_words_for_detection()
# Reason categories (this powers the explanation)
REASON_CATS = load_reason_categories()

# Precompute a lowercased keyword index for faster matching
# { category: {"reason": str, "keywords": [original...], "_kw_lower": [lower...]} }
for cat, cfg in REASON_CATS.items():
    cfg["_kw_lower"] = [k.lower() for k in cfg["keywords"]]

# ==========================================
# HELPERS
# ==========================================
def normalize_number(number: str) -> str:
    digits = ''.join(filter(str.isdigit, number or ""))
    if not digits.startswith("91"):
        digits = "91" + digits
    return f"+{digits}"

def translate_texts(texts):
    out = []
    for t in texts:
        t = t.strip()
        if not t:
            continue
        try:
            out.append(GoogleTranslator(source='auto', target='en').translate(t))
        except Exception:
            out.append(t)
    return out

def semantic_detect(sentences, threshold=0.6):
    """
    Return (matched_sentences, percent) using sentence->phrase max similarity.
    Uses the PHRASE_EMBEDDINGS (from utils.phrases).
    """
    matches = []
    for s in sentences:
        s_clean = s.strip()
        if not s_clean:
            continue
        s_emb = embedder.encode(s_clean, convert_to_tensor=True)
        scores = util.cos_sim(s_emb, PHRASE_EMBEDDINGS)[0]
        if float(scores.max()) >= threshold:
            matches.append(s_clean)
    pct = (len(matches) / max(len(sentences), 1)) * 100.0
    return matches, pct

def match_reason_categories(sentences):
    """
    Scan sentences against categories.json keywords.
    Returns:
      matched_map: { category: [ { "sentence": str, "keyword": str } ... ] }
    """
    matched_map = {cat: [] for cat in REASON_CATS.keys()}
    for s in sentences:
        s_low = s.lower()
        if not s_low:
            continue
        for cat, cfg in REASON_CATS.items():
            for kw_low, kw_orig in zip(cfg["_kw_lower"], cfg["keywords"]):
                if kw_low in s_low:
                    matched_map[cat].append({"sentence": s, "keyword": kw_orig})
    return matched_map

def build_reason_sentence(matched_map):
    """
    Build a single-sentence human-friendly reason from matched_map (category -> hits).
    Returns a single sentence, e.g.:
      "Possible phishing attempt: requests OTP or verification codes and uses urgency to pressure action."
    """
    # Collect unique reason strings for categories that had hits
    reasons = []
    seen = set()
    for cat, hits in matched_map.items():
        if not hits:
            continue
        # reason text from categories.json (strip trailing period)
        reason_text = REASON_CATS[cat]["reason"].strip().rstrip(".")
        if reason_text not in seen:
            seen.add(reason_text)
            reasons.append(reason_text)

    if not reasons:
        return "No phishing indicators detected."

    # Join reasons into a single grammatical sentence:
    # - 1 item: "A."
    # - 2 items: "A and B."
    # - 3+ items: "A, B, and C."
    if len(reasons) == 1:
        joined = reasons[0]
    elif len(reasons) == 2:
        joined = f"{reasons[0]} and {reasons[1]}"
    else:
        joined = ", ".join(reasons[:-1]) + ", and " + reasons[-1]

    return "Possible phishing attempt: " + joined + "."

# ==========================================
# CORE
# ==========================================
def process_audio(audio_path):
    audio = whisperx.load_audio(audio_path)
    result = whisper_model.transcribe(audio, batch_size=BATCH_SIZE)
    language = result.get("language")

    segments_text = [seg.get("text", "").strip()
                     for seg in result.get("segments", [])
                     if seg.get("text")]
    translated_segments = translate_texts(segments_text)
    translated_text = ". ".join(
        t.strip().rstrip(".") for t in translated_segments if t.strip()
    ) + "."
    sentences = split_sentences(translated_text)

    # Detection (semantic)
    sem_matches, sem_pct = semantic_detect(sentences, threshold=0.6)

    # Detection (weighted from utils)
    weighted_matches, total_weight = detect_phishing_sentences(sentences, threshold=0.6)
    weighted_pct = min((total_weight / max(len(sentences), 1)) * 10.0, 100.0)

    # Combine detection signals
    final_pct = max(sem_pct, weighted_pct)
    phishing_all = list({*sem_matches, *(m[0] for m in weighted_matches)})


    # Also provide a compact per-category list as part of the API response
    # Also provide a compact per-category list as part of the API response
    matched_map = match_reason_categories(phishing_all)

    category_details = {
        cat: [{"keyword": h["keyword"], "sentence": h["sentence"]} for h in hits]
        for cat, hits in matched_map.items() if hits
    }

    # Reasoning via categories.json
    reason = build_reason_sentence(matched_map)


    return {
        "Detected Language": language,
        "Transcribed and Translated Sentences": translated_text,
        "Total Sentences": len(sentences),
        "Phishing Count": len(phishing_all),
        "Phishing Percentage": final_pct,
        "Phishing Sentences": phishing_all,
        "Reason": reason
    }


# ==========================================
# API
# ==========================================
@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"}), 200

@app.route("/upload_call", methods=["POST"])
def upload_call():
    """
    Upload audio + phone number, process, update rating, save in DB
    """
    try:
        number = normalize_number((request.form.get("number") or "").strip())
        audio_file = request.files.get("audio")

        if not audio_file:
            return jsonify({"error": "No audio file uploaded"}), 400

        # Save to disk
        file_path = os.path.join(UPLOAD_FOLDER, audio_file.filename)
        audio_file.save(file_path)

        # Save in GridFS (with duplicate protection)
        audio_id, upload_count = save_audio_file(file_path, audio_file.filename)

        # Run analysis
        result = process_audio(file_path)

        # Update rating
        old_rating = get_existing_rating(number)
        new_rating = calculate_rating(old_rating, result["Phishing Percentage"])

        # Save transcript + rating in DB
        save_rating(
            number=number,
            transcript=result["Transcribed and Translated Sentences"],
            new_rating=new_rating,
            audio_id=audio_id,
            phishing_percent=result["Phishing Percentage"]
        )

        return jsonify({
            "number": number,
            "phishing_percent": result["Phishing Percentage"],
            "new_rating": new_rating,
            "analysis": result,
            "audio_id": str(audio_id)
        }), 200

    except Exception as e:
        traceback.print_exc()
        return jsonify({"error": str(e)}), 500

@app.route("/record/<number>", methods=["GET"])
def get_record(number: str):
    try:
        number = normalize_number(number)
        rec = get_full_record(number)
        if not rec:
            return jsonify({"error": "not found"}), 404

        rating = None
        if "audio_records" in rec and isinstance(rec["audio_records"], list):
            for ar in rec["audio_records"]:
                if isinstance(ar, dict) and "rating" in ar:
                    rating = ar["rating"]
                    break
        if rating is None and "rating" in rec:
            rating = rec["rating"]
        if rating is None:
            return jsonify({"error": "no rating found"}), 404

        return jsonify(rating), 200

    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/audio/<audio_id>", methods=["GET"])
def fetch_audio(audio_id: str):
    """Download audio file by GridFS id"""
    try:
        data = get_audio_file(audio_id)
        tmp = os.path.join(UPLOAD_FOLDER, f"{audio_id}.wav")
        with open(tmp, "wb") as f:
            f.write(data)
        return send_file(tmp, as_attachment=True, download_name=f"{audio_id}.wav")
    except Exception as e:
        return jsonify({"error": str(e)}), 404

# ==========================================
# RUN
# ==========================================
if __name__ == "__main__":
    import sys
    if len(sys.argv) > 1 and sys.argv[1] == "cli":
        audio_file = input("Enter path to audio file: ").strip()
        if not os.path.exists(audio_file):
            print("Error: File not found!")
        else:
            result = process_audio(audio_file)
            print("\n=== Analysis Report ===")
            print(json.dumps(result, indent=2))
    else:
        app.run(host="0.0.0.0", port=5000, debug=True)

