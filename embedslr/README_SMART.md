
# SMART-based MCDM module for EmbedSLR

This module adds a SMART (Simple Multi‑Attribute Rating Technique) ranking layer
on top of the existing **EmbedSLR** pipeline. It combines four criteria into a single
score and returns a ranked list of publications.

**Criteria**
1. Semantic similarity (cosine similarity to the query)
2. Topical similarity by Authors' Keywords
3. Overlap of intellectual linkages (shared references)
4. Mutual citations (bidirectional links within the core set)

The aggregation follows the additive SMART model. Default weights are derived
from **SMART importance ranks** via the exponential mapping
\( w_j \propto (\sqrt{2})^{h_j} \) normalized to sum to 1. You can also pass
explicit weights that sum to 1.

## Files

- `smart_mcdm.py` – the module (drop into your project or `embedslr/` package)
- `demo_smart.py` – minimal usage example

## Minimal usage

```python
import pandas as pd
from smart_mcdm import rank_with_smart, SMARTConfig

df = pd.read_csv("ranked.csv")  # produced by embedslr CLI, contains "distance_cosine"
cfg = SMARTConfig(
    importance_ranks={'semantic': 8, 'keywords': 7, 'references': 7, 'mutual': 6},
    scale_4to10=False,   # if True, utilities are mapped to 4..10 SMART scale before aggregation
    top_k_seed=20
)
res = rank_with_smart(df, config=cfg)

ranked = res.df
ranked.to_csv("ranked_smart.csv", index=False)

print("Weights:", res.weights)
print(ranked[["SMART_score"]].head())
```

## CLI integration sketch

If you want to extend the EmbedSLR CLI with a `--smart` flag:

```python
# inside embedslr/cli.py after ranking by cosine
from smart_mcdm import rank_with_smart, SMARTConfig

if args.smart:
    cfg = SMARTConfig(
        importance_ranks={'semantic': args.w_sem, 'keywords': args.w_kw,
                          'references': args.w_ref, 'mutual': args.w_mut},
        scale_4to10=args.smart_scale_4_10,
        top_k_seed=args.smart_top_k
    )
    smart_res = rank_with_smart(ranked, config=cfg)
    ranked = smart_res.df
```

Corresponding parser options:
```python
ap.add_argument("--smart", action="store_true", help="apply SMART MCDM re‑ranking")
ap.add_argument("--smart-top-k", type=int, default=20, help="core size used for criteria 2..4")
ap.add_argument("--smart-scale-4-10", action="store_true", help="map utilities to 4..10 before aggregation")
ap.add_argument("--w-sem", type=int, default=7, help="SMART rank (4..10) – semantic")
ap.add_argument("--w-kw",  type=int, default=7, help="SMART rank (4..10) – keywords")
ap.add_argument("--w-ref", type=int, default=7, help="SMART rank (4..10) – shared refs")
ap.add_argument("--w-mut", type=int, default=7, help="SMART rank (4..10) – mutual citations")
```

## Columns expected

- **distance_cosine** – produced by `embedslr.similarity.rank_by_cosine`
- **Author Keywords** – any of: `Author Keywords`, `Authors Keywords`, `DE`, `Index Keywords`
- **References** – any of: `References`, `Cited References`, `CR`, `REF`
- **DOI** – or a title column is needed to detect in‑dataset citation links

The module uses heuristics to parse keywords and references and falls back
gracefully if a column is missing (the corresponding utility becomes 0).

## Notes

- If you pass `query_vector` + `doc_vectors` instead of `distance_cosine`, the module
  will compute cosine similarity internally.
- By default, the "core set" used for criteria (2)–(4) is the top‑K articles by semantic similarity.
  You can override this with `seed_core_idxs=[...]` or pass a fixed `seed_keywords` set.
- The result object exposes per‑criterion utilities and contributions so you can audit the score.
