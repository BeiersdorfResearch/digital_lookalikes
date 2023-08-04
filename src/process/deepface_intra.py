import hydra
import time
import logging
from omegaconf import DictConfig
import concurrent.futures
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path
from random import sample
import more_itertools as mit
import deepface.DeepFace as dpf
import pandas as pd
import numpy as np
from tqdm import tqdm

log = logging.getLogger(__name__)


def intra_user_comps(target_user: int, selfies_dir: Path, metric: str, model: str):
    target_user_pics = list(Path(selfies_dir / str(target_user)).glob("*.jpg"))
    target_user_pics_shuffled = sample(list(target_user_pics), len(target_user_pics))
    list_df = []
    for pic in target_user_pics_shuffled[1:]:
        dpf_dict = dpf.verify(
            img1_path=target_user_pics_shuffled[0].as_posix(),
            img2_path=target_user_pics_shuffled[1].as_posix(),
            enforce_detection=False,
            detector_backend="mediapipe",
            distance_metric=metric,
            model_name=model,
        )
        dpf_dict["facial_areas"] = [dpf_dict["facial_areas"]]
        list_df.append(
            pd.DataFrame(
                {
                    **{
                        "user_id": target_user,
                        "img1_path": target_user_pics_shuffled[0].as_posix(),
                        "img2_path": pic.as_posix(),
                    },
                    **dpf_dict,
                }
            )
        )
    return list_df


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig):
    list_of_users = pd.read_csv(
        Path(cfg.selfie_data.save_dir).parent / "downloaded_selfies.csv"
    )["user_id"].unique()
    # chunks_of_users = np.array_split(list_of_users, 5)
    # chunk_to_process = chunks_of_users[cfg.user_chunk]
    with tqdm(desc="Performing Facial Recognition", total=len(list_of_users)) as pbar:
        with ProcessPoolExecutor(max_workers=8, max_tasks_per_child=3500) as executor:
            futures = [
                executor.submit(
                    intra_user_comps,
                    user,
                    Path(cfg.selfie_data.save_dir),
                    cfg.metric,
                    cfg.model,
                )
                for user in list_of_users
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    df_scores = pd.concat(future.result())
                    df_scores.to_csv(
                        "../../results/scores.csv",
                        index=False,
                        mode="a",
                        header=not Path("../../results/scores.csv").exists(),
                    )
                except Exception as e:
                    log.info(f"Metric: {cfg.metric}, and model: {cfg.model}")
                    log.info(f"Error: {e}")
                pbar.update(1)


if __name__ == "__main__":
    main()
    print("Done!")
