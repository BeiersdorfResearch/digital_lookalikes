# %%
import pandas as pd
import plotly.express as px
from plotly.subplots import make_subplots

# %%
df_scores = pd.read_csv("../results/scores.csv")

df_grp = (
    df_scores.groupby(["model", "similarity_metric"])["distance"].mean().reset_index()
)
df_grp["std"] = (
    df_scores.groupby(["model", "similarity_metric"])["distance"]
    .std()
    .reset_index()["distance"]
)

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
df_scores_wbug = pd.read_csv("../results/scores_wbug.csv")

# %%
df_grp_wbug = (
    df_scores_wbug.groupby(["model", "similarity_metric"])["distance"]
    .mean()
    .reset_index()
)
df_grp_wbug["std"] = (
    df_scores_wbug.groupby(["model", "similarity_metric"])["distance"]
    .std()
    .reset_index()["distance"]
)


# %%
px.bar(
    df_grp_wbug,
    x="model",
    y="distance",
    color="similarity_metric",
    error_y="std",
    barmode="group",
)
# %%
px.box(
    df_scores_wbug,
    x="model",
    y="distance",
    color="similarity_metric",
)


# %%

# make a plotly figure with subplots
fig = make_subplots(rows=1, cols=2, subplot_titles=("With Bug", "Without Bug"))

# add traces

fig.add_trace(
    px.box(
        df_scores_wbug,
        x="model",
        y="distance",
        color="similarity_metric",
    ).data[0],
    row=1,
    col=1,
)

fig.add_trace(
    px.box(
        df_scores,
        x="model",
        y="distance",
        color="similarity_metric",
    ).data[0],
    row=1,
    col=2,
)

# %%
