# =========================
#  EVALUASI SISTEM PENCARIAN
# =========================
# Berisi metrik evaluasi untuk menilai performa sistem IR:
# Precision, Recall, P@K, Average Precision, nDCG

import math

def precision_recall(system_result, gold_relevant):
    """Hitung Precision dan Recall dasar"""
    system_result = set(system_result)
    gold_relevant = set(gold_relevant)

    tp = len(system_result & gold_relevant)  # true positive
    fp = len(system_result - gold_relevant)  # false positive
    fn = len(gold_relevant - system_result)  # false negative

    precision = tp / (tp + fp) if (tp + fp) else 0.0
    recall = tp / (tp + fn) if (tp + fn) else 0.0
    return precision, recall, tp, fp, fn


def precision_at_k(results, gold_set, k=5):
    """Precision pada top-k hasil pencarian"""
    results = results[:k]
    hits = sum(1 for r in results if r[0] in gold_set)
    return hits / k


def average_precision(results, gold_set, k=10):
    """Average Precision (AP) untuk top-k"""
    hits, sum_prec = 0, 0.0
    for i, (doc, _) in enumerate(results[:k], start=1):
        if doc in gold_set:
            hits += 1
            sum_prec += hits / i
    return sum_prec / max(1, len(gold_set))


def ndcg_at_k(results, gold_set, k=5):
    """Normalized Discounted Cumulative Gain (nDCG@k)"""
    dcg = 0.0
    for i, (doc, _) in enumerate(results[:k], start=1):
        rel = 1 if doc in gold_set else 0
        dcg += (2**rel - 1) / math.log2(i + 1)

    # ideal DCG (semua relevan di urutan atas)
    idcg = sum((2**1 - 1) / math.log2(i + 1)
                for i in range(1, min(len(gold_set), k) + 1))
    return dcg / idcg if idcg else 0.0