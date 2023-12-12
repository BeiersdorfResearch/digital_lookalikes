# %%
from concurrent.futures import ProcessPoolExecutor
import concurrent.futures
import pandas as pd
from pathlib import Path
import deepface.DeepFace as dpf
from typing import List
from omegaconf import DictConfig
import hydra
import logging

from tqdm import tqdm

log = logging.getLogger(__name__)


def get_users_latest_selfies(selfies_dir: Path | str):
    selfies_dir = Path(selfies_dir)
    latest_selfie_paths: List[Path] = [
        sorted(user.glob("*.jpg"))[-1] for user in selfies_dir.glob("*")
    ]
    return latest_selfie_paths


selfies = get_users_latest_selfies("../../data/selfies")


def inter_user_comps(pic1: Path, pic2: Path, metric: str, model: str):
    dpf_dict = dpf.verify(
        img1_path=pic1.as_posix(),
        img2_path=pic2.as_posix(),
        enforce_detection=False,
        detector_backend="mediapipe",
        distance_metric=metric,
        model_name=model,
    )
    dpf_dict["facial_areas"] = [dpf_dict["facial_areas"]]
    return pd.DataFrame(
        {
            **{
                "user1_id": int(pic1.parent.name),
                "user2_id": int(pic2.parent.name),
                "img1_path": pic1.as_posix(),
                "img2_path": pic2.as_posix(),
            },
            **dpf_dict,
        }
    )


@hydra.main(config_path="../../config", config_name="config_inter", version_base=None)
def main(cfg: DictConfig):
    # chunks_of_users = np.array_split(list_of_users, 5)
    # chunk_to_process = chunks_of_users[cfg.user_chunk]
    latest_selfie_paths = get_users_latest_selfies(cfg.selfie_data.save_dir)
    df_done = pd.read_csv("../../results/inter_user_scores.csv")
    finished_users = df_done["user1_id"].unique()
    with tqdm(
        desc="Outer loop iterating over user selfies", total=len(latest_selfie_paths)
    ) as pbar_outer:
        for i, pic1 in enumerate(latest_selfie_paths):
            if int(pic1.parent.name) in finished_users:
                pbar_outer.update(1)
                continue
            with tqdm(
                desc="Inner loop iterating over user selfies",
                total=len(latest_selfie_paths[i + 1 :]),
            ) as pbar_inner:
                with ProcessPoolExecutor(max_workers=32) as executor:
                    futures = [
                        executor.submit(
                            inter_user_comps,
                            pic1,
                            pic2,
                            cfg.metric,
                            cfg.model,
                        )
                        for pic2 in latest_selfie_paths[i + 1 :]
                    ]
                    for future in concurrent.futures.as_completed(futures):
                        try:
                            df_scores = future.result()
                            df_scores.to_csv(
                                "../../results/inter_user_scores.csv",
                                index=False,
                                mode="a",
                                header=not Path(
                                    "../../results/inter_user_scores.csv"
                                ).exists(),
                            )
                        except Exception as e:
                            log.info(
                                f"Metric: {cfg.metric}, model: {cfg.model}, user1: {df_scores['user1_id']}, user2: {df_scores['user2_id']}"
                            )
                            log.info(f"Error: {e}")
                        pbar_inner.update(1)
            pbar_outer.update(1)


if __name__ == "__main__":
    main()
    print("Done!")
