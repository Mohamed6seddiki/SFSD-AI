import os
import re
import json
import time
import logging
import threading
import pickle
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
from sklearn.preprocessing import normalize

# ==========================================================
# ENV + LOGGING
# ==========================================================
load_dotenv()

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s",
)

APP_TITLE = "SFSD AI (Hybrid RAG • FR/EN • Pseudocode)"

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
SIM_THRESHOLD = float(os.getenv("SIM_THRESHOLD", "0.65"))  # خفضناه لـ TF-IDF
MAX_CONTEXT_CHARS = int(os.getenv("MAX_CONTEXT_CHARS", "12000"))

CHUNK_SIZE = int(os.getenv("CHUNK_SIZE", "900"))
CHUNK_OVERLAP = int(os.getenv("CHUNK_OVERLAP", "150"))

AUTO_REINDEX = os.getenv("AUTO_REINDEX", "true").lower() in ("1", "true", "yes", "y")

# ==========================================================
# FLASK
# ==========================================================
app = Flask(__name__)
# Allow all origins for CORS (adjust in production if needed)
CORS(app, resources={r"/*": {"origins": "*"}})

chat_history: List[Dict[str, Any]] = []
_index_lock = threading.Lock()

# ==========================================================
# LANGUAGE + EXERCISE DETECTION
# ==========================================================
def wants_bilingual(question: str) -> bool:
    q = question.lower()
    return any(x in q for x in [
        "français et anglais", "francais et anglais",
        "french and english", "fr+en", "fr/en"
    ])

def detect_lang(question: str) -> str:
    q = question.lower()
    en = ["what", "how", "explain", "write", "algorithm", "stack", "queue", "file system"]
    fr = ["explique", "donne", "algorithme", "pile", "file", "système"]
    return "en" if sum(w in q for w in en) > sum(w in q for w in fr) else "fr"

def is_exercise(question: str) -> bool:
    q = question.lower()
    return any(w in q for w in [
        "exercice", "td", "tp", "corrig", "solution",
        "écrire", "ecrire", "pseudocode", "algorithm"
    ])

# ==========================================================
# TEXT + PDF HANDLING
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

def chunk_text(text: str) -> List[str]:
    text = normalize_text(text)
    if not text:
        return []
    chunks, i = [], 0
    while i < len(text):
        j = min(len(text), i + CHUNK_SIZE)
        chunks.append(text[i:j])
        if j == len(text):
            break
        i = max(0, j - CHUNK_OVERLAP)
    return chunks

def load_pdfs_from_dir(pdf_dir: str) -> List[Chunk]:
    chunks: List[Chunk] = []
    if not os.path.exists(pdf_dir):
        logging.warning(f"PDF directory not found: {pdf_dir}")
        return chunks
    
    pdfs = [f for f in os.listdir(pdf_dir) if f.lower().endswith(".pdf")]
    logging.info(f"📚 Found {len(pdfs)} PDFs in {pdf_dir}")

    for fname in pdfs:
        try:
            reader = PdfReader(os.path.join(pdf_dir, fname))
            for i, page in enumerate(reader.pages, start=1):
                text = normalize_text(page.extract_text() or "")
                for c in chunk_text(text):
                    chunks.append(Chunk(c, fname, i))
        except Exception as e:
            logging.error(f"Error reading {fname}: {e}")

    logging.info(f"✅ Loaded {len(chunks)} chunks")
    return chunks

# ==========================================================
# VECTOR STORE (FAISS) - Using TF-IDF instead of transformers
# ==========================================================
INDEX_PATH = os.path.join(INDEX_DIR, "faiss.index")
META_PATH = os.path.join(INDEX_DIR, "meta.json")
VECTORIZER_PATH = os.path.join(INDEX_DIR, "vectorizer.pkl")

_vectorizer = None
_index = None
_meta = []

def vectorizer():
    """Initialize TF-IDF vectorizer (lightweight alternative to transformers)"""
    global _vectorizer
    if _vectorizer is None:
        # Load existing vectorizer if available
        if os.path.exists(VECTORIZER_PATH):
            logging.info("🔧 Loading existing TF-IDF vectorizer")
            with open(VECTORIZER_PATH, 'rb') as f:
                _vectorizer = pickle.load(f)
        else:
            logging.info("🔧 Creating new TF-IDF vectorizer")
            _vectorizer = TfidfVectorizer(
                max_features=512,  # Dimension size
                ngram_range=(1, 3),  # Unigrams to trigrams
                min_df=1,
                max_df=0.95,
                sublinear_tf=True,  # Better for longer documents
                strip_accents='unicode',
                lowercase=True,
                analyzer='word',
                token_pattern=r'\b\w+\b'
            )
    return _vectorizer

def embed(texts: List[str], is_query: bool = False) -> np.ndarray:
    """
    Convert texts to vectors using TF-IDF
    
    Args:
        texts: List of text strings
        is_query: True if embedding a query, False if embedding documents
    """
    vec = vectorizer()
    
    try:
        if is_query and hasattr(vec, 'vocabulary_'):
            # Transform query using existing vocabulary
            vectors = vec.transform(texts).toarray()
        elif not hasattr(vec, 'vocabulary_'):
            # First time: fit and transform
            vectors = vec.fit_transform(texts).toarray()
            # Save the vectorizer
            with open(VECTORIZER_PATH, 'wb') as f:
                pickle.dump(vec, f)
            logging.info(f"✅ Vectorizer trained with {len(vec.vocabulary_)} features")
        else:
            # Transform documents using existing vocabulary
            vectors = vec.transform(texts).toarray()
        
        # Normalize vectors for cosine similarity
        vectors = normalize(vectors, norm='l2', axis=1)
        return vectors.astype(np.float32)
    
    except Exception as e:
        logging.error(f"Error in embedding: {e}")
        # Fallback: return zero vectors
        return np.zeros((len(texts), 512), dtype=np.float32)

def build_index():
    """Build FAISS index from PDFs"""
    global _index, _meta, _vectorizer
    
    # Reset vectorizer to retrain
    _vectorizer = None
    
    chunks = load_pdfs_from_dir(DATA_DIR)
    if not chunks:
        # Create empty index
        _index = faiss.IndexFlatIP(512)  # Default dimension
        _meta = []
        logging.warning("No PDFs found, created empty index")
        
        # Save empty index
        faiss.write_index(_index, INDEX_PATH)
        with open(META_PATH, "w", encoding="utf-8") as f:
            json.dump(_meta, f, ensure_ascii=False)
        return

    # Extract texts and embed
    texts = [c.text for c in chunks]
    logging.info(f"🔄 Embedding {len(texts)} chunks with TF-IDF...")
    
    vecs = embed(texts, is_query=False)
    
    # Build FAISS index
    dim = vecs.shape[1]
    _index = faiss.IndexFlatIP(dim)  # Inner Product (cosine similarity after normalization)
    _index.add(vecs)
    
    _meta = [{"file": c.file, "page": c.page, "text": c.text} for c in chunks]

    # Save to disk
    faiss.write_index(_index, INDEX_PATH)
    with open(META_PATH, "w", encoding="utf-8") as f:
        json.dump(_meta, f, ensure_ascii=False)
    
    logging.info(f"✅ Index built successfully with {len(_meta)} chunks, dimension={dim}")

def ensure_index():
    """Load or build index"""
    global _index, _meta, _vectorizer
    
    if _index is not None:
        return
    
    if os.path.exists(INDEX_PATH) and os.path.exists(META_PATH) and os.path.exists(VECTORIZER_PATH):
        try:
            _index = faiss.read_index(INDEX_PATH)
            with open(META_PATH, "r", encoding="utf-8") as f:
                _meta = json.load(f)
            # Vectorizer will be loaded when needed
            logging.info(f"✅ Loaded existing index with {len(_meta)} chunks")
        except Exception as e:
            logging.error(f"Error loading index: {e}")
            build_index()
    else:
        build_index()

def retrieve(question: str) -> Tuple[List[Dict], float]:
    """Retrieve relevant chunks for a question"""
    ensure_index()
    
    if not _meta:
        return [], 0.0
    
    # Embed query
    qv = embed([question], is_query=True)
    
    # Search
    scores, ids = _index.search(qv, min(TOP_K, len(_meta)))
    
    results, best = [], 0.0
    for s, i in zip(scores[0], ids[0]):
        if i >= 0 and i < len(_meta):
            m = _meta[int(i)]
            score = float(s)
            if score > best:
                best = score
            results.append({
                "score": score,
                "file": m["file"],
                "page": m["page"],
                "snippet": m["text"][:400]
            })
    
    return results, best

# ==========================================================
# LLM CALL (GROQ)
# ==========================================================
def groq(system, user):
    """Call Groq API"""
    try:
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
        r.raise_for_status()
        return r.json()["choices"][0]["message"]["content"]
    except Exception as e:
        logging.error(f"Groq API error: {e}")
        raise

# ==========================================================
# PROMPTS
# ==========================================================
SYSTEM_GROUNDED = """
You are an academic SFSD tutor.
Answer in the SAME language as the question (French or English).
Use SFSD PSEUDOCODE ONLY.

Use:
DEBUT / FIN
SI / SINON / FSI
TQ / FTQ
← for assignment
// comments

If EXERCISE_MODE=yes:
- Give idea
- Steps
- PSEUDOCODE
- Complexity

Put code inside ```pseudo```.

Add Sources at the end.
"""

SYSTEM_GENERAL = """
⚠️ General answer (not based on PDFs).
Same rules as above.
"""

# ==========================================================
# ROUTES
# ==========================================================
@app.route("/")
def home():
    return jsonify({
        "status": "ok",
        "message": "SFSD AI Backend is running (TF-IDF Mode)",
        "endpoints": ["/health", "/ask", "/reindex"]
    })

@app.route("/health")
def health():
    try:
        ensure_index()
        return jsonify({
            "status": "healthy",
            "chunks_indexed": len(_meta),
            "groq_configured": bool(GROQ_API_KEY),
            "data_dir": DATA_DIR,
            "index_dir": INDEX_DIR,
            "vectorizer_trained": _vectorizer is not None and hasattr(_vectorizer, 'vocabulary_'),
            "embedding_method": "TF-IDF (lightweight)",
            "errors": []
        })
    except Exception as e:
        return jsonify({
            "status": "unhealthy",
            "error": str(e)
        }), 500

@app.route("/reindex", methods=["POST"])
def reindex():
    try:
        with _index_lock:
            build_index()
        return jsonify({
            "ok": True, 
            "chunks_indexed": len(_meta),
            "message": "Index rebuilt successfully"
        })
    except Exception as e:
        logging.error(f"Reindex error: {e}")
        return jsonify({"error": str(e)}), 500

@app.route("/ask", methods=["POST"])
def ask():
    try:
        data = request.get_json() or {}
        question = data.get("question", "").strip()
        
        if not question:
            return jsonify({"error": "question required"}), 400

        # Detect language and exercise mode
        lang = detect_lang(question)
        bilingual = wants_bilingual(question)
        exercise = is_exercise(question)

        # Retrieve relevant chunks
        sources, best = retrieve(question)
        grounded = best >= SIM_THRESHOLD and len(sources) > 0

        # Build context metadata
        meta = f"LANG={lang}\nBILINGUAL={bilingual}\nEXERCISE_MODE={'yes' if exercise else 'no'}\n"

        # Generate answer
        if grounded:
            ctx = "\n".join([f"{s['file']} p.{s['page']}: {s['snippet']}" for s in sources])
            answer = groq(SYSTEM_GROUNDED, meta + question + "\n\n" + ctx)
        else:
            answer = groq(SYSTEM_GENERAL, meta + question)

        # Store in history
        chat_history.append({"q": question, "a": answer})

        return jsonify({
            "answer": answer,
            "grounded": grounded,
            "sources": sources,
            "history_count": len(chat_history),
            "best_score": best
        })
        
    except Exception as e:
        logging.error(f"Error in /ask: {e}")
        return jsonify({"error": str(e)}), 500

# ==========================================================
# MAIN
# ==========================================================
if __name__ == "__main__":
    port = int(os.getenv("PORT", 5000))
    
    logging.info("="*60)
    logging.info("🚀 Starting SFSD AI (TF-IDF Mode - Lightweight)")
    logging.info("="*60)
    
    ensure_index()
    
    app.run(host="0.0.0.0", port=port, debug=False)
