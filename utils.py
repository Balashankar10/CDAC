import re
import json
import os
import hashlib
from sentence_transformers import SentenceTransformer, util

def _load_raw_phishing_words():
    file_path = os.path.join(os.path.dirname(__file__), "phishing_words.json")
    with open(file_path, "r", encoding="utf-8") as f:
        return json.load(f)

def load_phrases_and_categories():
    """
    Returns:
      phrases: flat list of phrases (for embeddings)
      cat_map: dict phrase(lower) -> category
    Accepts both a dict of categories->list and a plain list fallback.
    """
    data = _load_raw_phishing_words()
    phrases, cat_map = [], {}
    if isinstance(data, dict):
        for cat, kws in data.items():
            for kw in kws:
                phrases.append(kw)
                cat_map[kw.lower()] = cat
    elif isinstance(data, list):
        phrases = data
        # best-effort: map everything to "generic"
        cat_map = {kw.lower(): "generic" for kw in data}
    return phrases, cat_map

phrases, cat_map = load_phrases_and_categories()
model = SentenceTransformer('all-MiniLM-L6-v2')
phishing_embeddings = model.encode(phrases, convert_to_tensor=True)

def split_sentences(text: str):
    return [s.strip() for s in re.split(r'(?<=[.!?])\s+', text) if s.strip()]

def detect_phishing_sentences(sentences, threshold=0.6):
    """
    Returns:
      phishing: list of tuples (sentence, score)
      total_weight: sum of scores for matched sentences
    """
    phishing = []
    total_weight = 0.0
    for sentence in sentences:
        if not sentence.strip():
            continue
        sent_embedding = model.encode(sentence, convert_to_tensor=True)
        scores = util.cos_sim(sent_embedding, phishing_embeddings)[0]
        max_score = float(scores.max())
        if max_score >= threshold:
            phishing.append((sentence, max_score))
            total_weight += max_score
    return phishing, total_weight

def calculate_rating(old_rating, phishing_percent):
    alpha = 0.3
    correcting_factor = 2.0
    if old_rating is None:
        old_rating = 8.0
    corr_phishing = (phishing_percent / 10.0)
    new_rating = alpha * (10.0 - (corr_phishing * correcting_factor)) + (1 - alpha) * old_rating
    return round(max(0.0, min(10.0, new_rating)), 2)

def hash_audio_file(file_path):
    sha256 = hashlib.sha256()
    with open(file_path, "rb") as f:
        for chunk in iter(lambda: f.read(4096), b""):
            sha256.update(chunk)
    return sha256.hexdigest()

