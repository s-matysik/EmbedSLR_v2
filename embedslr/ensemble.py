# embedslr/ensemble.py
from __future__ import annotations
from dataclasses import dataclass
from typing import Iterable, List, Dict, Tuple, Literal
import re, numpy as np, pandas as pd

from .embeddings import get_embeddings
from .similarity import rank_by_cosine

Agg = Literal["mean", "min", "median"]

@dataclass(frozen=True)
class ModelSpec:
    provider: str
    model: str
    @property
    def label(self) -> str:
        m = self.model.split("/")[-1]
        safe = re.sub(r"[^a-zA-Z0-9._-]+", "-", m)
        return f"{self.provider}_{safe}"

def parse_models(specs: Iterable[str]) -> List[ModelSpec]:
    out: List[ModelSpec] = []
    for s in specs:
        if not s.strip() or ":" not in s:
            raise ValueError(f"Invalid model spec '{s}'. Use PROVIDER:MODEL_ID.")
        prov, model = s.split(":", 1)
        out.append(ModelSpec(prov.strip(), model.strip()))
    if not out:
        raise ValueError("No models provided.")
    return out

def _rank_one(df: pd.DataFrame, texts: List[str], query: str, ms: ModelSpec) -> pd.DataFrame:
    doc_vecs = get_embeddings(texts, provider=ms.provider, model=ms.model)
    q_vec    = get_embeddings([query], provider=ms.provider, model=ms.model)[0]
    ranked   = rank_by_cosine(q_vec, doc_vecs, df).reset_index(drop=False).rename(columns={"index":"_idx"})
    ranked[f"rank_{ms.label}"]     = np.arange(1, len(ranked) + 1, dtype=int)
    ranked[f"distance_{ms.label}"] = ranked["distance_cosine"].astype(float)
    ranked = ranked[["_idx", f"rank_{ms.label}", f"distance_{ms.label}"]]
    out = pd.DataFrame(index=df.index)
    out = out.join(ranked.set_index("_idx"), how="left")
    return out

def run_ensemble(
    df: pd.DataFrame,
    combined_text_col: str,
    query: str,
    model_specs: List[ModelSpec],
    *,
    top_k_per_model: int = 50,
    aggregator: Agg = "mean",
) -> pd.DataFrame:
    texts = df[combined_text_col].astype(str).tolist()
    base = df.copy()
    per_model: Dict[str, pd.DataFrame] = {}
    for ms in model_specs:
        part = _rank_one(df, texts, query, ms)
        per_model[ms.label] = part
        base = base.join(part)

    rank_cols     = [c for c in base.columns if c.startswith("rank_")]
    distance_cols = [c for c in base.columns if c.startswith("distance_")]

    hit_mask = np.zeros((len(base), len(rank_cols)), dtype=bool)
    for j, c in enumerate(rank_cols):
        hit_mask[:, j] = (base[c].values <= top_k_per_model)

    base["hit_count"]  = hit_mask.sum(axis=1).astype(int)
    base["hit_models"] = [
        ";".join([rank_cols[j].replace("rank_", "") for j, ok in enumerate(row) if ok])
        for row in hit_mask
    ]

    dist_arr = np.column_stack([base[c].values for c in distance_cols])
    rank_arr = np.column_stack([base[c].values for c in rank_cols])
    dist_arr = np.where(hit_mask, dist_arr, np.nan)
    rank_arr = np.where(hit_mask, rank_arr, np.nan)

    if aggregator == "mean":
        agg_dist = np.nanmean(dist_arr, axis=1)
    elif aggregator == "min":
        agg_dist = np.nanmin(dist_arr, axis=1)
    else:
        agg_dist = np.nanmedian(dist_arr, axis=1)

    base["agg_distance"] = agg_dist
    base["mean_rank"]    = np.nanmean(rank_arr, axis=1)

    base = base.sort_values(
        by=["hit_count", "agg_distance", "mean_rank"],
        ascending=[False, True, True]
    ).reset_index(drop=True)
    return base

def per_group_bibliometrics(ranked_df: pd.DataFrame, groups=(5,4,3,2,1)) -> pd.DataFrame:
    try:
        from .bibliometrics import indicator_a, indicator_a_prime, indicator_b, indicator_b_prime
    except Exception:
        # repo bez bibliometrii – bezpieczny fallback
        import numpy as _np
        rows = [{"group": g, "n": int((ranked_df["hit_count"]==g).sum()),
                 "A": _np.nan, "A′": _np.nan, "B": _np.nan, "B′": _np.nan} for g in groups]
        return pd.DataFrame(rows)

    rows = []
    for k in groups:
        g = ranked_df[ranked_df["hit_count"] == k]
        if g.empty:
            rows.append({"group": k, "n": 0, "A": np.nan, "A′": np.nan, "B": np.nan, "B′": np.nan})
            continue
        A   = indicator_a(g); Ap = indicator_a_prime(g)
        B   = indicator_b(g); Bp = indicator_b_prime(g)
        rows.append({"group": k, "n": len(g), "A": A, "A′": Ap, "B": B, "B′": Bp})
    return pd.DataFrame(rows)
