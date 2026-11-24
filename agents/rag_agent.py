"""
CogniOps AI - RAG Agent (Retrieval-Augmented Generation).
Indexes and retrieves operational enterprise policies and past decision cases.
Uses sentence-transformers + FAISS if available; otherwise falls back gracefully to
scikit-learn TF-IDF Vectorizer with Cosine Similarity.
Provides matching text snippets and active method reporting.
"""
import os
import json
import re
from pathlib import Path
from typing import List, Dict, Any
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from core.config import POLICIES_DIR, HISTORY_DIR

class RAGAgent:
    def __init__(self):
        self.name = "RAG Agent"
        self.documents = []  # List of {id, title, text, type, source_file}
        self.active_backend = "TF-IDF + Cosine Similarity (Offline Fallback)"
        self.backend_code = "tfidf"
        self.vectorizer = None
        self.tfidf_matrix = None
        self.st_model = None
        self.faiss_index = None
        self.loaded = False

    def _init_embeddings(self):
        """Attempts to load sentence-transformers and FAISS."""
        try:
            from sentence_transformers import SentenceTransformer
            import faiss
            self.st_model = SentenceTransformer("all-MiniLM-L6-v2")
            self.backend_code = "sentence_transformers_faiss"
            self.active_backend = "sentence-transformers (all-MiniLM-L6-v2) + FAISS Index"
        except Exception:
            self.st_model = None
            self.faiss_index = None
            self.backend_code = "tfidf"
            self.active_backend = "TF-IDF + Cosine Similarity (Deterministic Offline)"

    def index_documents(self):
        """Indexes all policy text files and historical JSON decision records."""
        self._init_embeddings()
        docs = []

        # 1. Policies
        if POLICIES_DIR.exists():
            for pfile in sorted(POLICIES_DIR.glob("*.txt")):
                with open(pfile, "r", encoding="utf-8") as f:
                    content = f.read()
                title_line = content.splitlines()[1] if len(content.splitlines()) > 1 else pfile.stem
                title = title_line.replace("TITLE: ", "").strip()
                docs.append({
                    "id": pfile.stem,
                    "title": title,
                    "text": content,
                    "doc_type": "policy",
                    "source_file": pfile.name
                })

        # 2. Historical Decisions
        if HISTORY_DIR.exists():
            for hfile in sorted(HISTORY_DIR.glob("*.json")):
                with open(hfile, "r", encoding="utf-8") as f:
                    data = json.load(f)
                text = (
                    f"Situation: {data.get('situation_title')}. "
                    f"Applied Strategy: {data.get('applied_strategy')}. "
                    f"Outcome: {data.get('outcome_status')}. Delta Utility: {data.get('delta_utility')}. "
                    f"Summary: {data.get('summary')}"
                )
                docs.append({
                    "id": data.get("record_id", hfile.stem),
                    "title": f"Past Decision: {data.get('applied_strategy')}",
                    "text": text,
                    "doc_type": "past_decision",
                    "source_file": hfile.name,
                    "meta": data
                })

        self.documents = docs

        # Try building FAISS index if sentence-transformers is available
        if self.backend_code == "sentence_transformers_faiss" and self.st_model is not None:
            try:
                import faiss
                corpus_texts = [d["text"] for d in self.documents]
                embeddings = self.st_model.encode(corpus_texts, convert_to_numpy=True, normalize_embeddings=True)
                dim = embeddings.shape[1]
                index = faiss.IndexFlatIP(dim)
                index.add(embeddings)
                self.faiss_index = index
            except Exception:
                self.backend_code = "tfidf"
                self.active_backend = "TF-IDF + Cosine Similarity (Deterministic Offline)"

        # Always prepare TF-IDF as rock-solid guaranteed fallback
        corpus = [d["text"] for d in self.documents]
        self.vectorizer = TfidfVectorizer(stop_words="english", ngram_range=(1, 2))
        self.tfidf_matrix = self.vectorizer.fit_transform(corpus)
        self.loaded = True

    def _extract_matching_highlight(self, query: str, full_text: str) -> str:
        """Finds sentence with highest term overlap with query to highlight."""
        query_words = set(re.findall(r"\w+", query.lower()))
        sentences = re.split(r"(?<=[.!?\n])\s+", full_text)
        best_sentence = ""
        best_overlap = -1
        for s in sentences:
            s_clean = s.strip()
            if not s_clean:
                continue
            s_words = set(re.findall(r"\w+", s_clean.lower()))
            overlap = len(query_words.intersection(s_words))
            if overlap > best_overlap:
                best_overlap = overlap
                best_sentence = s_clean
        return best_sentence if best_sentence else full_text[:180]

    def retrieve(self, query: str, top_k: int = 3, doc_type: str = None) -> List[Dict[str, Any]]:
        """
        Retrieves top-k relevant policy passages or historical decisions for a query.
        Returns ranked list with similarity scores, document references, and highlighted snippet.
        """
        if not self.loaded:
            self.index_documents()

        results = []

        if self.backend_code == "sentence_transformers_faiss" and self.faiss_index is not None:
            try:
                query_emb = self.st_model.encode([query], convert_to_numpy=True, normalize_embeddings=True)
                D, I = self.faiss_index.search(query_emb, len(self.documents))
                for score, idx in zip(D[0], I[0]):
                    doc = self.documents[idx]
                    if doc_type and doc["doc_type"] != doc_type:
                        continue
                    highlight = self._extract_matching_highlight(query, doc["text"])
                    results.append({
                        "doc_id": doc["id"],
                        "title": doc["title"],
                        "snippet": highlight,
                        "score": round(float(score), 3),
                        "doc_type": doc["doc_type"],
                        "source_file": doc["source_file"],
                        "full_text": doc["text"]
                    })
                    if len(results) >= top_k:
                        return results
            except Exception:
                pass

        # TF-IDF Cosine Similarity Fallback
        query_vec = self.vectorizer.transform([query])
        sim_scores = cosine_similarity(query_vec, self.tfidf_matrix)[0]
        ranked_indices = sim_scores.argsort()[::-1]

        for idx in ranked_indices:
            score = float(sim_scores[idx])
            doc = self.documents[idx]
            if doc_type and doc["doc_type"] != doc_type:
                continue

            highlight = self._extract_matching_highlight(query, doc["text"])

            results.append({
                "doc_id": doc["id"],
                "title": doc["title"],
                "snippet": highlight,
                "score": round(score, 3),
                "doc_type": doc["doc_type"],
                "source_file": doc["source_file"],
                "full_text": doc["text"]
            })

            if len(results) >= top_k:
                break

        return results

    def retrieve_context_for_situation(self, situation: Dict[str, Any]) -> Dict[str, Any]:
        """Convenience method to retrieve top policies and past decisions with active backend metadata."""
        query_text = f"{situation.get('title')} {situation.get('description')} {situation.get('category')}"
        policies = self.retrieve(query_text, top_k=3, doc_type="policy")
        history = self.retrieve(query_text, top_k=3, doc_type="past_decision")
        return {
            "policies": policies,
            "past_decisions": history,
            "active_backend": self.active_backend,
            "backend_code": self.backend_code
        }
