# %%
from typing import List
import pandas as pd
import plotly.express as px
from pathlib import Path
import numpy as np

# %%
df_inter = pd.read_csv(
    "../results/inter_user_scores.csv",
    names=[
        "user1_id",
        "user2_id",
        "img1_path",
        "img2_path",
        "verified",
        "distance",
        "threshold",
        "model",
        "detector_backend",
        "similarity_metric",
        "facial_areas",
        "time",
    ],
)

# %%
df_inter.info()
# %%
df_inter = df_inter.loc[df_inter["user1_id"] != 3468]


# %%
def get_users_latest_selfies(selfies_dir: Path | str):
    selfies_dir = Path(selfies_dir)
    latest_selfie_paths: List[Path] = [
        sorted(user.glob("*.jpg"))[-1] for user in selfies_dir.glob("*")
    ]
    return latest_selfie_paths


# %%
selfies = get_users_latest_selfies("../data/selfies")

# %%
# %%
df_inter

# %%
for user in selfies:
    if int(user.parent.name) in df_inter["user1_id"].unique():
        print(user.parent.name)
        break

# %%
