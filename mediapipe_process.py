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

model_path = Path(
    "/home/azureuser/localfiles/digital-twins/test_data/face_landmarker.task"
)

# %%
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

image_paths = sorted(Path("./test_data/twin_images").glob("*jpg"))
# %%
image_errors = []
for path in image_paths:
    image = mp.Image.create_from_file(path.as_posix())
    detection_result = landmarker.detect(image)
    try:
        annotated_image = draw_landmarks_on_image(image.numpy_view(), detection_result)
    except IndexError:
        image_errors.append(path)
    color_corrected_image = cv2.cvtColor(annotated_image, cv2.COLOR_BGR2RGB)
    plt.imshow(color_corrected_image)
    plt.show()

print(image_errors)
# %%
plot_face_blendshapes_bar_graph(detection_result.face_blendshapes[0])

# %%
twin1 = image_paths[0]
twin2 = image_paths[1]

twin1_image = mp.Image.create_from_file(twin1.as_posix())

twin2_image = mp.Image.create_from_file(twin2.as_posix())

detection1 = landmarker.detect(twin1_image)
detection2 = landmarker.detect(twin2_image)

annotated_image1 = draw_landmarks_on_image(
    np.zeros_like(twin1_image.numpy_view()), detection1
)
annotated_image2 = draw_landmarks_on_image(
    np.zeros_like(twin2_image.numpy_view()), detection2
)

# %%
fig, (ax1, ax2) = plt.subplots(ncols=2)

ax1.imshow(cv2.cvtColor(annotated_image1, cv2.COLOR_BGR2RGB))
ax2.imshow(cv2.cvtColor(annotated_image2, cv2.COLOR_BGR2RGB))


# %%
def get_landmark_coords(landmarker):
    x_coords = []
    y_coords = []
    z_coords = []
    for landmark in landmarker.face_landmarks[0]:
        x_coords.append(landmark.x)
        y_coords.append(landmark.y)
        z_coords.append(landmark.z)
    return x_coords, y_coords, z_coords


# %%
twin1_coords = get_landmark_coords(detection1)
twin2_coords = get_landmark_coords(detection2)

plt.scatter(twin1_coords[0], twin1_coords[1], s=twin1_coords[2])
plt.scatter(twin2_coords[0], twin2_coords[1], s=twin2_coords[2])

# %%
