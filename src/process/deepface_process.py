# %%
import concurrent.futures
from pathlib import Path
import deepface.DeepFace as dpf
import pandas as pd
import numpy as np
import plotly.express as px
from tqdm import tqdm
import time


def deepface_find_user(user_path):
    pic = sorted(user_path.glob("*.jpg"))[0]
    try:
        return dpf.find(
            img_path=pic.as_posix(),
            db_path=user_path.as_posix(),
            enforce_detection=False,
            detector_backend="mediapipe",
        )
    except AttributeError as e:
        raise AttributeError from e


def deepface_verify_user(img2_path):
    pic = sorted(target_user.glob("*.jpg"))[0]
    return dpf.verify(
        img1_path=pic.as_posix(),
        img2_path=img2_path.as_posix(),
        enforce_detection=False,
        detector_backend="mediapipe",
    )


def deepface_find_db(user_dirs):
    with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
        return list(executor.map(deepface_find_user, user_dirs))


# %%

user_dirs = sorted(Path("./data/selfies/").glob("*"))

target_user = user_dirs[1]

pics = sorted(target_user.glob("*.jpg"))
pic = pics[0]

# %%
all_users_df = deepface_find_db(user_dirs=user_dirs)

# %%
with concurrent.futures.ProcessPoolExecutor(max_workers=4) as executor:
    all_users_df = [
        executor.submit(deepface_find_user, user_path) for user_path in user_dirs
    ]
    for i in all_users_df:
        try:
            df = i.result()
        except AttributeError:
            continue


# %%
all_users_df[0].result()[0]

# %%
dfs = []
for df in all_users_df:
    try:
        dfs.append(df.result()[0])
    except AttributeError:
        continue

# %%
distance_means = np.array([df["VGG-Face_cosine"][1:].mean() for df in dfs])
# %%
np.mean(distance_means)

len(dfs)
# %%

df_users_distance = pd.concat(dfs)

# %%
df_users_distance["user_id"] = df_users_distance["identity"].str.split(
    "/", expand=True
)[2]

# %%
df_users_distance = df_users_distance[["user_id", "VGG-Face_cosine"]]


# %%
df_users_distance = pd.read_csv(
    "./preliminary_results_deepface/top100users_distance_metric_to_self.csv",
    index_col=0,
)
df_users_distance = df_users_distance.astype({"user_id": "category"})
# %%
fig = px.box(
    df_users_distance.loc[df_users_distance["VGG-Face_cosine"] > 1e-3],
    x="user_id",
    y="VGG-Face_cosine",
    height=1000,
    width=1500,
    points=False,
)

fig.update_xaxes(type="category")

# %%
deepface_find_user(target_user)

# %%

user_dirs = sorted(Path("./data/selfies/").glob("*"))

target_user = user_dirs[1]

pics = target_user.glob("*.jpg")

with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:
    results = list(tqdm(executor.map(deepface_verify_user, pics)))

# %%
