# %%
from pathlib import Path

import matplotlib.pyplot as plt
import mediapipe as mp
import numpy as np
from mediapipe.framework.formats import landmark_pb2
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from visutil import draw_landmarks_on_image, plot_face_blendshapes_bar_graph
import deepface.DeepFace as dpf

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

# %%
user_id = 1210
path_user_selfies = path_selfies / Path(f"{user_id}")
user_selfies = sorted(path_user_selfies.glob("*.jpg"))

# %%

cosine_distance = dpf.verify(
    user_selfies[0].as_posix(),
    user_selfies[100].as_posix(),
    detector_backend="mediapipe",
    distance_metric="cosine",
)["distance"]

fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(nrows=2, ncols=2)

selfie_1 = mp.Image.create_from_file(user_selfies[0].as_posix())
ax1.imshow(selfie_1.numpy_view())
ax1.set_title("Raw Selfie")

detection_result_1 = landmarker.detect(selfie_1)
annotated_selfie_1 = draw_landmarks_on_image(selfie_1.numpy_view(), detection_result_1)
ax2.imshow(annotated_selfie_1)
ax2.set_title("Selfie w/ detected landmarks")

selfie_2 = mp.Image.create_from_file(user_selfies[100].as_posix())
ax3.imshow(selfie_2.numpy_view())

detection_result_2 = landmarker.detect(selfie_2)
annotated_selfie_2 = draw_landmarks_on_image(selfie_2.numpy_view(), detection_result_2)
ax4.imshow(annotated_selfie_2)

fig.suptitle(f"Selfies for {user_id = } | {cosine_distance = :.5f}")
fig.savefig(f"./presentation_plots/{user_id}_selfie_comp_w_distance.jpg")

# %%
