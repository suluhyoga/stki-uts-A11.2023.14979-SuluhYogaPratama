# ===============================
#  VECTOR SPACE MODEL (TF-IDF)
# ===============================
# Menghitung bobot TF-IDF dan ranking menggunakan cosine similarity

import math
from collections import Counter
from src.preprocess import preprocess_text

def compute_tf(doc_tokens):
    """Hitung Term Frequency"""
    tf = Counter(doc_tokens)
    total = len(doc_tokens)
    return {t: f / total for t, f in tf.items()}

def compute_idf(all_docs):
    """Hitung Inverse Document Frequency"""
    N = len(all_docs)
    df = {}
    for tokens in all_docs.values():
        for term in set(tokens):
            df[term] = df.get(term, 0) + 1
    return {term: math.log((N + 1) / (df_val + 1)) + 1 for term, df_val in df.items()}

def compute_tfidf(all_docs):
    """Hitung TF-IDF seluruh dokumen"""
    idf = compute_idf(all_docs)
    tfidf_docs = {}
    for name, tokens in all_docs.items():
        tf = compute_tf(tokens)
        tfidf_docs[name] = {term: tf[term] * idf[term] for term in tf}
    return tfidf_docs, idf

def cosine_similarity(vec1, vec2):
    """Cosine similarity antar dua vektor"""
    common = set(vec1.keys()) & set(vec2.keys())
    num = sum(vec1[t] * vec2[t] for t in common)
    denom1 = math.sqrt(sum(v * v for v in vec1.values()))
    denom2 = math.sqrt(sum(v * v for v in vec2.values()))
    return num / (denom1 * denom2) if denom1 and denom2 else 0.0

def get_snippet(doc_path, length=120):
    """Ambil potongan awal teks untuk preview hasil pencarian"""
    try:
        with open(doc_path, "r", encoding="utf-8") as f:
            text = f.read().strip()
        return text[:length] + "..." if len(text) > length else text
    except:
        return "-"

def search_vsm(query, tfidf_docs, idf, data_folder=None, top_k=10):
    """Pencarian berbasis Vector Space Model"""
    query_tokens = preprocess_text(query)
    if not query_tokens:
        return []

    tf_query = compute_tf(query_tokens)
    tfidf_query = {term: tf_query[term] * idf.get(term, 0) for term in tf_query}
    scores = {doc: cosine_similarity(tfidf_query, tfidf_docs[doc])
            for doc in tfidf_docs}

    ranked = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    ranked = [(doc, score) for doc, score in ranked if score > 0]

    if data_folder:
        ranked = [(doc, score, get_snippet(f"{data_folder}/{doc}"))
                for doc, score in ranked[:top_k]]
    return ranked