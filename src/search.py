# ============================
#  FUNGSI PENCARIAN TERPADU
# ============================
# Menggabungkan dua model pencarian:
#  - Boolean Model
#  - Vector Space Model (TF-IDF)

from src.boolean_ir import load_processed_docs, build_inverted_index, boolean_search
from src.vsm_ir import compute_tfidf, search_vsm

# Muat dokumen dan buat indeks sekali di awal
docs = load_processed_docs("data/processed")
index = build_inverted_index(docs)
all_docs = list(docs.keys())
tfidf_docs, idf = compute_tfidf(docs)

def search_boolean(query):
    """Pencarian model Boolean"""
    result = boolean_search(query.lower(), index, all_docs)
    return sorted(list(result))

def search_vector(query):
    """Pencarian model VSM (TF-IDF + Cosine Similarity)"""
    result = search_vsm(query.lower(), tfidf_docs, idf)
    return result