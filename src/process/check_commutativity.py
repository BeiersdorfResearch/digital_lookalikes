# %%
from pathlib import Path
import deepface.DeepFace as dpf
from typing import List
import pandas as pd

# %%
list_of_userdirs = list(Path("../../data/selfies").glob("*"))

# %%
results: List[dict] = []
for user in list_of_userdirs[:5]:
    pics = list(user.glob("*.jpg"))
    results.extend(
        (
            dpf.verify(
                img1_path=pics[0].as_posix(),
                img2_path=pics[1].as_posix(),
                enforce_detection=False,
                detector_backend="mediapipe",
                distance_metric="cosine",
                model_name="OpenFace",
            ),
            dpf.verify(
                img1_path=pics[1].as_posix(),
                img2_path=pics[0].as_posix(),
                enforce_detection=False,
                detector_backend="mediapipe",
                distance_metric="cosine",
                model_name="OpenFace",
            ),
        )
    )

# %%
df_results = pd.DataFrame.from_dict(results)

# %%
