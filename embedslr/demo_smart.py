
import pandas as pd
from smart_mcdm import rank_with_smart, SMARTConfig

# Expect a CSV with at least a "distance_cosine" column (e.g., output from embedslr CLI)
df = pd.read_csv("ranked.csv")

cfg = SMARTConfig(
    importance_ranks={'semantic': 8, 'keywords': 7, 'references': 7, 'mutual': 6},
    scale_4to10=False,
    top_k_seed=20
)

res = rank_with_smart(df, config=cfg)

# Full ranked dataframe with SMART score
ranked = res.df
ranked.to_csv("ranked_smart.csv", index=False)

# Inspect utilities and contributions
print("Weights:", res.weights)
print(res.utilities.head())
print(res.contributions.head())
