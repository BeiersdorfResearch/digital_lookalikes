# %%
from typing import List
import pandas as pd
import plotly.express as px
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import numpy as np

plt.style.use("seaborn-v0_8-whitegrid")

# %%
df_inter = pd.read_csv("../results/inter_user_scores.csv", header=0)
# %%
df_inter = pd.read_parquet("../results/inter_user_scores.parquet")
# %%
df_inter = df_inter.astype(
    {
        "user1_id": "int32",
        "user2_id": "int32",
        "distance": "float32",
        "threshold": "category",
        "model": "category",
        "detector_backend": "category",
        "similarity_metric": "category",
        "time": "float32",
    }
)

# %%
fig, ax = plt.subplots(figsize=(10, 10))

sns.histplot(data=df_inter, x="distance", binwidth=0.1, stat="percent", ax=ax, kde=True)
ax.set_title("Distribution of distances between users")
ax.vlines(x=0.29, ymin=0, ymax=17, color="red", label="Threshold")
fig.show()


# %%
