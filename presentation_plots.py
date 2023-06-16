# %%
from pathlib import Path

import cv2
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
user_id = 2221
path_user_selfies = path_selfies / Path(f"{user_id}")
user_selfies = sorted(path_user_selfies.glob("*.jpg"))

# %%

fig, ax = plt.subplots(ncols=2)

ax1, ax2 = ax[0], ax[1]


ax1.imshow(cv2.imread(user_selfies[0].as_posix()))
ax1.set_title("Raw Selfie")

selfie_1 = mp.Image.create_from_file(user_selfies[0].as_posix())
detection_result_1 = landmarker.detect(selfie_1)
annotated_selfie_1 = draw_landmarks_on_image(selfie_1.numpy_view(), detection_result_1)
color_corrected_selfie_1 = cv2.cvtColor(annotated_selfie_1, cv2.COLOR_BGR2RGB)
ax2.imshow(color_corrected_selfie_1)
ax2.set_title("Selfie w/ detected landmarks")

fig.suptitle(f"Selfie for {user_id = }")
fig.savefig(f"./presentation_plots/{user_id}_selfie1.jpg")

# %%

fig, ax = plt.subplots(ncols=2)

ax1, ax2 = ax[0], ax[1]


ax1.imshow(cv2.imread(user_selfies[100].as_posix()))
ax1.set_title("Raw Selfie")

selfie_1 = mp.Image.create_from_file(user_selfies[100].as_posix())
detection_result_1 = landmarker.detect(selfie_1)
annotated_selfie_1 = draw_landmarks_on_image(selfie_1.numpy_view(), detection_result_1)
color_corrected_selfie_1 = cv2.cvtColor(annotated_selfie_1, cv2.COLOR_BGR2RGB)
ax2.imshow(color_corrected_selfie_1)
ax2.set_title("Selfie w/ detected landmarks")

fig.suptitle(f"Selfie for {user_id = }")
fig.savefig(f"./presentation_plots/{user_id}_selfie2.jpg")

# %%

dpf.verify(
    user_selfies[0].as_posix(),
    user_selfies[100].as_posix(),
    detector_backend="mediapipe",
)
# %%
