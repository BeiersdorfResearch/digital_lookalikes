# %%
import itertools as it
import random
from pathlib import Path

import deepface.DeepFace as dpf
import matplotlib.pyplot as plt
import mediapipe as mp
import pandas as pd
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from tqdm import tqdm

from src.visualization.visutil import (
    draw_landmarks_on_image,
    plot_face_blendshapes_bar_graph,
)

path_selfies = Path("./data/selfies")
model_path = Path("./test_data/face_landmarker.task")

base_options = python.BaseOptions(model_asset_path=model_path)
face_landmarker = vision.FaceLandmarker
face_landmarker_options = vision.FaceLandmarkerOptions

options = face_landmarker_options(
    base_options=base_options,
    running_mode=vision.RunningMode.IMAGE,
    num_faces=1,
    min_face_detection_confidence=0.5,
    min_face_presence_confidence=0.5,
    output_face_blendshapes=True,
    output_facial_transformation_matrixes=True,
)

landmarker = vision.FaceLandmarker.create_from_options(options)

# %%[markdown]
# # Same User Comparison

# %%
user_id = 1210
path_user_selfies = path_selfies / Path(f"{user_id}")
user_selfies = sorted(path_user_selfies.glob("*.jpg"))
selfie1_path = user_selfies[0].as_posix()
selfie2_path = user_selfies[100].as_posix()


def plot_selfies_w_landmarks(landmarker, selfie1_path, selfie2_path):
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

    selfie = mp.Image.create_from_file(selfie1_path)
    ax1.imshow(selfie.numpy_view())
    ax1.set_title("Raw Selfie")

    detection_result = landmarker.detect(selfie)
    annotated_selfie = draw_landmarks_on_image(selfie.numpy_view(), detection_result)
    ax2.imshow(annotated_selfie)
    ax2.set_title("Selfie w/ detected landmarks")

    selfie = mp.Image.create_from_file(selfie2_path)
    ax3.imshow(selfie.numpy_view())

    detection_result = landmarker.detect(selfie)
    annotated_selfie = draw_landmarks_on_image(selfie.numpy_view(), detection_result)
    ax4.imshow(annotated_selfie)
    return fig


# %%

fig = plot_selfies_w_landmarks(landmarker, selfie1_path, selfie2_path)

cosine_distance = dpf.verify(
    img1_path=selfie1_path,
    img2_path=selfie2_path,
    detector_backend="mediapipe",
    distance_metric="cosine",
)["distance"]

fig.suptitle(f"Selfies for {user_id = } | {cosine_distance = :.5f}")
fig.savefig(f"./presentation_plots/{user_id}_selfie_comp_w_distance.jpg")

# %%[markdown]
# # Different User Comparison
# %%
data_dict = dict(user1=[], user2=[], selfie1_path=[], selfie2_path=[], distance=[])

selfie_pool = []
for user_path in path_selfies.glob("*"):
    user_selfies = list(user_path.glob("*"))
    selfie = random.choice(user_selfies)
    selfie_pool.append(selfie)

users = [path.parts[-2] for path in selfie_pool]

for img2 in tqdm(selfie_pool[1:]):
    img1 = selfie_pool[0]
    user1 = selfie_pool[0].parts[-2]
    user2 = img2.parts[-2]
    try:
        cosine_distance = dpf.verify(
            img1_path=img1.as_posix(),
            img2_path=img2.as_posix(),
            detector_backend="mediapipe",
            distance_metric="cosine",
        )["distance"]
    except ValueError or AttributeError:
        continue
    data_dict["user1"].append(user1)
    data_dict["user2"].append(user2)
    data_dict["selfie1_path"].append(img1.as_posix())
    data_dict["selfie2_path"].append(img2.as_posix())
    data_dict["distance"].append(cosine_distance)

df_selfie_comp = pd.DataFrame(data_dict)
# %%
min_distance = df_selfie_comp.loc[
    df_selfie_comp["distance"] == df_selfie_comp["distance"].min()
]
max_distance = df_selfie_comp.loc[
    df_selfie_comp["distance"] == df_selfie_comp["distance"].max()
]

fig = plot_selfies_w_landmarks(
    landmarker=landmarker,
    selfie1_path=min_distance["selfie1_path"].values[0],
    selfie2_path=min_distance["selfie2_path"].values[0],
)

cosine_distance = dpf.verify(
    img1_path=min_distance["selfie1_path"].values[0],
    img2_path=min_distance["selfie2_path"].values[0],
    detector_backend="mediapipe",
    distance_metric="cosine",
)["distance"]


fig.suptitle(
    f"Selfies for {min_distance['user1'].values[0]}, {min_distance['user2'].values[0]} | {cosine_distance = :.5f}"
)
fig.savefig(
    f"./presentation_plots/{min_distance['user1'].values[0]}_{min_distance['user2'].values[0]}_comparison_cosine_distance.jpg"
)
fig.show()

# %%

fig = plot_selfies_w_landmarks(
    landmarker=landmarker,
    selfie1_path=max_distance["selfie1_path"].values[0],
    selfie2_path=max_distance["selfie2_path"].values[0],
)

cosine_distance = dpf.verify(
    img1_path=max_distance["selfie1_path"].values[0],
    img2_path=max_distance["selfie2_path"].values[0],
    detector_backend="mediapipe",
    distance_metric="cosine",
)["distance"]

fig.suptitle(
    f"Selfies for {max_distance['user1'].values[0]}, {max_distance['user2'].values[0]} | {cosine_distance = :.5f}"
)
fig.savefig(
    f"./presentation_plots/{max_distance['user1'].values[0]}_{max_distance['user2'].values[0]}_comparison_cosine_distance.jpg"
)
fig.show()
# %%
for img1, img2 in tqdm(it.combinations(selfie_pool, 2)):
    user1 = img1.parts[-2]
    user2 = img2.parts[-2]
    try:
        cosine_distance = dpf.verify(
            img1_path=img1.as_posix(),
            img2_path=img2.as_posix(),
            detector_backend="mediapipe",
            distance_metric="cosine",
        )["distance"]
    except ValueError:
        continue
    data_dict["user1"].append(user1)
    data_dict["user2"].append(user2)
    data_dict["selfie1_path"].append(img1.as_posix())
    data_dict["selfie2_path"].append(img2.as_posix())
    data_dict["distance"].append(cosine_distance)
