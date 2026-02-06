import os
import re
import json
import logging
import threading
from dataclasses import dataclass
from typing import List, Dict, Any, Tuple

import numpy as np
import requests
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from pypdf import PdfReader

import faiss
from sklearn.feature_extraction.text import TfidfVectorizer

# ==========================================================
# ENV + LOGGING
# ==========================================================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

BASE_DIR = os.path.dirname(os.path.abspath(__file__))

DATA_DIR = os.getenv("DATA_DIR", os.path.join(BASE_DIR, "..", "data", "lectures"))
DATA_DIR = os.path.abspath(DATA_DIR)

INDEX_DIR = os.getenv("INDEX_DIR", os.path.join(BASE_DIR, "index_store"))
INDEX_DIR = os.path.abspath(INDEX_DIR)
os.makedirs(INDEX_DIR, exist_ok=True)

GROQ_API_KEY = os.getenv("GROQ_API_KEY")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.1-8b-instant")
GROQ_BASE_URL = os.getenv("GROQ_BASE_URL", "https://api.groq.com/openai/v1")

TOP_K = int(os.getenv("TOP_K", "5"))
MAX_FEATURES = int(os.getenv("MAX_FEATURES", "4096"))

# ==========================================================
# FLASK
# ==========================================================
app = Flask(__name__)
CORS(app)

chat_history: List[Dict[str, Any]] = []
_index_lock = threading.Lock()

# ==========================================================
# TEXT + PDF
# ==========================================================
def normalize_text(t: str) -> str:
    t = t.replace("\x00", " ")
    t = re.sub(r"[ \t]+", " ", t)
    t = re.sub(r"\n{3,}", "\n\n", t)
    return t.strip()

@dataclass
class Chunk:
    text: str
    file: str
    page: int

def load_pdfs(pdf_dir: str) -> List[Chunk]:
    chunks = []
    for fname in os.listdir(pdf_dir):
        if not fname.lower().endswith(".pdf"):
            continue
        reader = PdfReader(os.path.join(pdf_dir, fname))
        for i, page in enumerate(reader.pages, start=1):
            text = normalize_text(page.extract_text() or "")
            if text:
                chunks.append(Chunk(text, fname, i))
    logging.info(f"Loaded {len(chunks)} chunks")
    return chunks

# ==========================================================
# VECTOR STORE (TF-IDF + FAISS)
# ==========================================================
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")
VECT_PATH = os.path.join(INDEX_DIR, "vectorizer.json")

_vectorizer = None
_index = None
_meta = []

def build_index():
    global _vectorizer, _index, _meta

    chunks = load_pdfs(DATA_DIR)
    texts = [c.text for c in chunks]

    _vectorizer = TfidfVectorizer(
        max_features=MAX_FEATURES,
        stop_words=None
    )

    X = _vectorizer.fit_transform(texts).astype(np.float32)
    X = X.toarray()

    dim = X.shape[1]
    _index = faiss.IndexFlatIP(dim)
    faiss.normalize_L2(X)
    _index.add(X)

    _meta = [
        {"file": c.file, "page": c.page, "text": c.text}
        for c in chunks
    ]

    faiss.write_index(_index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(_meta, f, ensure_ascii=False)

    with open(VECT_PATH, "w", encoding="utf-8") as f:
        json.dump(_vectorizer.vocabulary_, f)

def ensure_index():
    global _vectorizer, _index, _meta

    if _index is not None:
        return

    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH) and os.path.exists(VECT_PATH):
        _index = faiss.read_index(INDEX_PATH)
        with open(META_PATH, "r", encoding="utf-8") as f:
            _meta = json.load(f)

        with open(VECT_PATH, "r", encoding="utf-8") as f:
            vocab = json.load(f)

        _vectorizer = TfidfVectorizer(vocabulary=vocab)
    else:
        build_index()

def retrieve(question: str):
    ensure_index()
    qv = _vectorizer.transform([question]).toarray().astype(np.float32)
    faiss.normalize_L2(qv)

    scores, ids = _index.search(qv, TOP_K)
    results = []

    for s, i in zip(scores[0], ids[0]):
        if i >= 0:
            m = _meta[i]
            results.append({
                "score": float(s),
                "file": m["file"],
                "page": m["page"],
                "snippet": m["text"][:400]
            })
    return results

# ==========================================================
# GROQ
# ==========================================================
def groq(system, user):
    r = requests.post(
        f"{GROQ_BASE_URL}/chat/completions",
        headers={
            "Authorization": f"Bearer {GROQ_API_KEY}",
            "Content-Type": "application/json"
        },
        json={
            "model": GROQ_MODEL,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user}
            ],
            "temperature": 0.2
        },
        timeout=60
    )
    return r.json()["choices"][0]["message"]["content"]

# ==========================================================
# ROUTES
# ==========================================================
@app.route("/health")
def health():
    ensure_index()
    return jsonify({"status": "ok", "chunks": len(_meta)})

@app.route("/reindex", methods=["POST"])
def reindex():
    with _index_lock:
        build_index()
    return jsonify({"ok": True, "chunks": len(_meta)})

@app.route("/ask", methods=["POST"])
def ask():
    data = request.get_json() or {}
    question = data.get("question", "").strip()
    if not question:
        return jsonify({"error": "question required"}), 400

    sources = retrieve(question)
    ctx = "\n".join([f"{s['file']} p.{s['page']}: {s['snippet']}" for s in sources])

    system = "You are an academic SFSD tutor. Use PSEUDOCODE."
    answer = groq(system, question + "\n\n" + ctx)

    return jsonify({
        "answer": answer,
        "sources": sources
    })

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    ensure_index()
    app.run(host="0.0.0.0", port=5000)
