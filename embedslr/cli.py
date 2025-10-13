from __future__ import annotations
import argparse, json, sys, os
import pandas as pd

from .io import read_csv, autodetect_columns, combine_title_abstract
from .embeddings import get_embeddings
from .similarity import rank_by_cosine
from .bibliometrics import full_report
from .advanced_scoring import rank_with_advanced_scoring
from .config import ScoringConfig, ColumnMap


def _parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(prog="embedslr", description="SLR screening toolkit")
    sub = ap.add_subparsers(dest="cmd", required=True)

    # --- embed & rank by cosine ---
    emb = sub.add_parser("embed", help="Compute embeddings & cosine distances")
    emb.add_argument("-i", "--input", required=True, help="CSV file exported from Scopus")
    emb.add_argument("-q", "--query", required=True, help="Research problem / query string")
    emb.add_argument("-p", "--provider", default="sbert",
                     choices=["sbert", "openai", "cohere", "nomic", "jina"])
    emb.add_argument("-m", "--model", help="Override default model name")
    emb.add_argument("--api_key", help="Pass API key via CLI (otherwise use env var)")
    emb.add_argument("-o", "--out", default="ranking.csv")
    emb.add_argument("--json_embs", action="store_true",
                     help="Store embeddings JSON in the output CSV")

    # --- advanced scoring ---
    sc = sub.add_parser("score", help="Run advanced scoring (L-Scoring / Z-Scoring / L-Scoring+)")
    sc.add_argument("-i", "--input", required=True, help="Input CSV with at least Title/Abstract")
    sc.add_argument("--method", default="linear_plus",
                    choices=["linear", "zscore", "linear_plus"])
    sc.add_argument("--top_keywords", type=int, default=5)
    sc.add_argument("--top_references", type=int, default=15)
    sc.add_argument("--penalty_no_keywords", type=float, default=0.10)
    sc.add_argument("--penalty_no_references", type=float, default=0.10)
    sc.add_argument("--weights", default=None,
                    help='JSON mapping of weights, e.g. {"semantic":0.4,"keywords":0.3,"references":0.2,"citations":0.1}')
    sc.add_argument("--bonus_start_z", type=float, default=2.0)
    sc.add_argument("--bonus_full_z", type=float, default=4.0)
    sc.add_argument("--bonus_cap_points", type=float, default=None)
    sc.add_argument("--save_frequencies", action="store_true")
    sc.add_argument("--out_dir", default=".")
    sc.add_argument("-o", "--out", default="advanced_ranking.csv")
    # optional column mappings
    sc.add_argument("--col_keywords", default=None)
    sc.add_argument("--col_references", default=None)
    sc.add_argument("--col_citations", default=None)
    sc.add_argument("--col_semantic_similarity", default=None)
    sc.add_argument("--col_distance_cosine", default=None)

    # optional convenience: recompute semantic similarity if query is provided
    sc.add_argument("-q", "--query", default=None,
                    help="If provided: compute (1 - cosine distance) to query into 'semantic_similarity'")

    return ap


def cmd_embed(args: argparse.Namespace) -> None:
    df = read_csv(args.input)
    title_col, abs_col = autodetect_columns(df)
    text = combine_title_abstract(df, title_col, abs_col).tolist()
    provider = args.provider
    model = args.model
    api_key = args.api_key or os.getenv("OPENAI_API_KEY") or os.getenv("COHERE_API_KEY") or None

    print(f"[i] Embedding {len(text)} documents using {provider} {model or ''}".strip())
    embs = get_embeddings(text, provider=provider, model=model, api_key=api_key)
    ranked = rank_by_cosine(embs['query'], embs['docs'], df)

    if args.json_embs:
        ranked["combined_embeddings"] = [json.dumps(e) for e in embs["docs"]]

    ranked.to_csv(args.out, index=False)
    print(f"[✓] Ranking saved -> {args.out}")


def cmd_score(args: argparse.Namespace) -> None:
    df = read_csv(args.input)

    # optional: compute semantic similarity from query if asked
    cols = ColumnMap(
        keywords=args.col_keywords,
        references=args.col_references,
        citations=args.col_citations,
        semantic_similarity=args.col_semantic_similarity,
        distance_cosine=args.col_distance_cosine,
    )

    cfg = ScoringConfig(
        method=args.method,
        top_keywords=args.top_keywords,
        top_references=args.top_references,
        penalty_no_keywords=args.penalty_no_keywords,
        penalty_no_references=args.penalty_no_references,
        bonus_start_z=args.bonus_start_z,
        bonus_full_z=args.bonus_full_z,
        bonus_cap_points=args.bonus_cap_points,
        save_frequencies=args.save_frequencies,
        out_dir=args.out_dir,
        columns=cols,
    )

    # Recompute semantic_similarity if a query was provided
    if args.query:
        title_col, abs_col = autodetect_columns(df)
        text = combine_title_abstract(df, title_col, abs_col).tolist()
        embs = get_embeddings([args.query] + text, provider="sbert", model=None, api_key=None)
        query_vec = embs["docs"][0]
        doc_vecs = embs["docs"][1:]
        # cosine similarity
        import numpy as np
        from sklearn.metrics.pairwise import cosine_similarity
        q = np.asarray(query_vec).reshape(1, -1)
        d = np.asarray(doc_vecs)
        sim = cosine_similarity(q, d)[0]
        df["semantic_similarity"] = sim
        cfg.columns.semantic_similarity = "semantic_similarity"

    # parse weights from JSON string if provided
    if args.weights:
        cfg.weights = json.loads(args.weights)

    res = rank_with_advanced_scoring(df, cfg)
    res.df.to_csv(args.out, index=False)
    print(f"[✓] Advanced ranking saved -> {args.out}")
    if cfg.save_frequencies:
        print(f"[✓] Frequencies saved to {cfg.out_dir}")


def main(argv: list[str] | None = None) -> None:
    args = _parser().parse_args(argv)
    if args.cmd == "embed":
        cmd_embed(args)
    elif args.cmd == "score":
        cmd_score(args)
    else:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
