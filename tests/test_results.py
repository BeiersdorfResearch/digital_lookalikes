# %%
import pandas as pd
import plotly.express as px

# %%
df_scores = pd.read_csv("../results/scores.csv")
# %%
df_scores

# %%
df_grp = (
    df_scores.groupby(["model", "similarity_metric"])["distance"].mean().reset_index()
)
df_grp["std"] = (
    df_scores.groupby(["model", "similarity_metric"])["distance"]
    .std()
    .reset_index()["distance"]
)

# %%
df_grp

# %%
px.bar(
    df_grp,
    x="model",
    y="distance",
    color="similarity_metric",
    error_y="std",
    barmode="group",
)
# %%
px.box(
    df_scores,
    x="model",
    y="distance",
    color="similarity_metric",
)

# %%
df_scores.groupby(["model", "similarity_metric"])["time"].sum() / 3600
# %%
