"""Evaluate the PGS repository's small, labeled ranking example.

Run: python3 TECH400_W7_evaluate.py
This checks rankings in a feature table, not a live web search index.
"""

import csv
import math
from collections import defaultdict
from pathlib import Path


DATA = Path("/home/zoroo/tekraj_sir/pgs-search-engine/search-engine/training_data")
RELEVANT_LABEL = 2


def read_groups(path):
    groups = defaultdict(list)
    with path.open(newline="", encoding="utf-8") as source:
        for row in csv.DictReader(source):
            groups[row["query"]].append(row)
    return groups


def ranked(group, score_field):
    # Stable tie break: document ID. No relevance labels enter ranking.
    return sorted(group, key=lambda row: (-float(row[score_field]), row["document_id"]))


def fused(group):
    # Match repository's reciprocal rank fusion constant: k = 60.
    scores = defaultdict(float)
    for field in ("bm25_score", "vector_score"):
        for rank, row in enumerate(ranked(group, field), start=1):
            scores[row["document_id"]] += 1 / (60 + rank)
    return sorted(group, key=lambda row: (-scores[row["document_id"]], row["document_id"]))


def measures(order, all_rows):
    # Labels 2 and 3 count as relevant for binary measures.
    relevant = {row["document_id"] for row in all_rows if int(row["label"]) >= RELEVANT_LABEL}
    top2 = order[:2]
    hits = sum(row["document_id"] in relevant for row in top2)
    precision = hits / 2
    recall = hits / len(relevant) if relevant else 0.0

    # Average precision uses precision at every relevant hit.
    seen = 0
    precision_sum = 0.0
    for rank, row in enumerate(order, start=1):
        if row["document_id"] in relevant:
            seen += 1
            precision_sum += seen / rank
    average_precision = precision_sum / len(relevant) if relevant else 0.0

    # Graded gain rewards labels 0–3; earlier results get more credit.
    def dcg(labels):
        return sum((2**label - 1) / math.log2(rank + 1)
                   for rank, label in enumerate(labels, start=1))

    labels = [int(row["label"]) for row in order]
    ideal = sorted((int(row["label"]) for row in all_rows), reverse=True)
    ndcg = dcg(labels) / dcg(ideal) if dcg(ideal) else 0.0
    return precision, recall, average_precision, ndcg


def main():
    groups = read_groups(DATA / "ranking_training_data.csv")
    print(f"Rows: {sum(map(len, groups.values()))}; queries: {len(groups)}; candidates/query: 4")
    print("Relevant: label >= 2; cutoff: 2; judgments: sample candidates only")
    print("Model       P@2  Recall@2  MAP@4  nDCG@4")
    for name, order_fn in (
        ("BM25", lambda g: ranked(g, "bm25_score")),
        ("LaBSE", lambda g: ranked(g, "vector_score")),
        ("RRF", fused),
    ):
        values = [measures(order_fn(group), group) for group in groups.values()]
        means = [sum(row[i] for row in values) / len(values) for i in range(4)]
        print(f"{name:<11} {means[0]:.2f}      {means[1]:.2f}    {means[2]:.2f}     {means[3]:.2f}")
    example = groups["kathmandu municipality"]
    print("Example, kathmandu municipality:")
    for name, order in (("BM25", ranked(example, "bm25_score")),
                        ("LaBSE", ranked(example, "vector_score")),
                        ("RRF", fused(example))):
        print(f"  {name}: " + ", ".join(f"{r['document_id']}({r['label']})" for r in order))
    held_out = read_groups(DATA / "reranked_results.csv")
    scores = {row["rerank_score"] for group in held_out.values() for row in group}
    print(f"Saved reranker output: {sum(map(len, held_out.values()))} rows, "
          f"{len(held_out)} queries; unique scores: {', '.join(sorted(scores))}")


if __name__ == "__main__":
    main()
