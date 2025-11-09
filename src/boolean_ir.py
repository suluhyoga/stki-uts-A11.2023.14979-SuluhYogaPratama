# =======================
#  BOOLEAN RETRIEVAL MODEL
# =======================
# Modul ini bertugas membangun struktur indeks boolean
# (inverted index + incidence matrix) dan melakukan pencarian
# berdasarkan operator logika: AND, OR, NOT, ( )

from collections import defaultdict
import os

def load_processed_docs(folder_path):
    """Memuat semua dokumen hasil preprocessing dari folder data/processed"""
    docs = {}
    for filename in os.listdir(folder_path):
        if filename.endswith(".txt"):
            with open(os.path.join(folder_path, filename), "r", encoding="utf-8") as f:
                text = f.read().strip()
                # setiap dokumen disimpan sebagai list token
                docs[filename] = text.split()
    return docs


def build_inverted_index(docs):
    """Membangun inverted index: term → daftar dokumen yang mengandung term tersebut"""
    index = {}
    for doc, tokens in docs.items():
        for term in set(tokens):  # set() supaya tidak duplikat
            index.setdefault(term, set()).add(doc)
    return index


def build_incidence_matrix(docs):
    """Membangun incidence matrix (term vs dokumen) berisi 1/0"""
    vocab = sorted({t for tokens in docs.values() for t in tokens})
    matrix = defaultdict(dict)
    for term in vocab:
        for doc, tokens in docs.items():
            matrix[term][doc] = 1 if term in tokens else 0
    return matrix


def boolean_search(query, index, all_docs):
    """
    Melakukan pencarian Boolean.
    Mendukung operator: AND, OR, NOT, dan tanda kurung ( ).
    """
    q = query.lower().replace("(", " ( ").replace(")", " ) ")
    tokens = q.split()

    def eval_expr(tokens):
        stack = []
        i = 0
        while i < len(tokens):
            t = tokens[i]
            if t == "(":
                # rekursif: cari pasangan kurung )
                j, depth = i + 1, 1
                while j < len(tokens):
                    if tokens[j] == "(":
                        depth += 1
                    elif tokens[j] == ")":
                        depth -= 1
                        if depth == 0:
                            break
                    j += 1
                stack.append(eval_expr(tokens[i + 1:j]))
                i = j
            elif t in ("and", "or"):
                stack.append(t)
            elif t == "not":
                if i + 1 < len(tokens):
                    term = tokens[i + 1]
                    stack.append(set(all_docs) - index.get(term, set()))
                    i += 1
            else:
                stack.append(index.get(t, set()))
            i += 1

        # hanya satu term
        if len(stack) == 1:
            return stack[0]

        # eksekusi operator kiri → kanan
        while len(stack) > 2:
            left, op, right = stack[0], stack[1], stack[2]
            new = left & right if op == "and" else left | right
            stack = [new] + stack[3:]
        return stack[0] if stack else set()

    try:
        result = eval_expr(tokens)
        return result if result else set()
    except Exception:
        return set()