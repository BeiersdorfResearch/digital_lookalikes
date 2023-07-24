# %%
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
import logging
from pathlib import Path

import hydra
from PIL import Image
import numpy as np
import pandas as pd
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
from dl_conn import get_dl_conn
from dl_orm import DLorm
from omegaconf import DictConfig
from tqdm import tqdm

credential = ManagedIdentityCredential()


def get_user_selfie_data(cfg: DictConfig) -> pd.DataFrame:
    query = f"""
    SELECT pgsfe.user_id
            ,pgsfe.ts_date
            ,pgsfe.id
            ,pgsfe.full_path
            ,pgsfe.selfie_link_id
        FROM pg.selfie as pgsfe
        JOIN pg.users as u ON u.user_id = pgsfe.user_id

        WHERE pgsfe.error_code IS {cfg.dl_filters.error_code} 
        AND pgsfe.anonymization_date IS {cfg.dl_filters.anonymization_date}
        AND pgsfe.ts_date 
            BETWEEN '{cfg.dl_filters.earliest_ts_date}' AND '{cfg.dl_filters.latest_ts_date}'
        AND u.participant_type = '{cfg.dl_filters.participant_type}'
        AND pgsfe.selfie_link_id in (SELECT DISTINCT selfie_link_id FROM pg.measure_procedure)
    """
    con = get_dl_conn()
    datalake = DLorm(con)
    df_selfies = (
        datalake.get_query(query)
        .sort_values("user_id")
        .drop_duplicates(subset=["user_id", "selfie_link_id"])
    )
    filter_users = get_users_w_min_nrselfies(df_selfies, cfg.dl_filters.nr_selfies)
    df_selfies = df_selfies.loc[df_selfies["user_id"].isin(filter_users)]
    return df_selfies


def get_users_w_min_nrselfies(
    df_selfies: pd.DataFrame, min_nr_selfies: int
) -> np.ndarray:
    df_nr_selfies = (
        df_selfies.groupby("user_id")
        .nunique()
        .reset_index()[["user_id", "selfie_link_id"]]
        .rename(columns={"selfie_link_id": "nr_selfies"})
    )

    return np.array(
        df_nr_selfies.loc[df_nr_selfies["nr_selfies"] >= min_nr_selfies][
            "user_id"
        ].values
    )


def init_blob(
    user_id: int,
    date: str,
    filename: str,
    container_name: str = "selfies",
):
    logging.getLogger("azure").setLevel(logging.ERROR)

    return BlobClient(
        account_url="https://claire1kstorage.blob.core.windows.net",
        container_name=container_name,
        blob_name=f"{date}/{user_id}/{filename}",
        credential=credential,
    )


def check_blob_exists(row):
    path = Path(row["full_path"])
    date = path.parts[3]
    filename = path.parts[5]
    user_id = row["user_id"]

    blob = init_blob(user_id, date, filename)

    return np.nan if blob.exists() else row["selfie_link_id"]


def filter_bad_blobs(df_selfies: pd.DataFrame) -> pd.Series:
    non_existent_blobs = []
    with tqdm(total=len(df_selfies), ncols=100) as pbar:
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = [
                executor.submit(check_blob_exists, row)
                for row in df_selfies.to_dict("records")
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    non_existent_blobs.append(future.result())
                    pbar.update(1)
                except Exception as e:
                    print(f"{future} raised an exception {e}")
                    pbar.update(1)
                    continue

    return df_selfies.loc[~df_selfies["selfie_link_id"].isin(non_existent_blobs)]


def clean_selfie_blobs(df_selfies: pd.DataFrame) -> pd.DataFrame:
    df_filtered_latest = filter_bad_blobs(
        df_selfies.sort_values("ts_date").groupby("user_id").tail(2)
    ).drop_duplicates(subset=["user_id"])
    df_filtered_random = filter_bad_blobs(
        df_selfies.loc[
            ~df_selfies["selfie_link_id"].isin(df_filtered_latest["selfie_link_id"])
        ]
        .groupby("user_id")
        .sample(5)
    )
    df_count_selfies = df_filtered_random.groupby("user_id").nunique().reset_index()
    passing_users = df_count_selfies.loc[df_count_selfies["full_path"] >= 2][
        "user_id"
    ].values
    df_sampled_random = (
        df_filtered_random.loc[df_filtered_random["user_id"].isin(passing_users)]
        .groupby("user_id")
        .sample(2)
    )
    df_match_users = df_filtered_latest.loc[
        df_filtered_latest["user_id"].isin(df_filtered_random["user_id"])
    ]
    return pd.concat([df_match_users, df_sampled_random])


def download_blob(
    user_id: int,
    date: str,
    filename: str,
    save_path: Path | str,
):
    blob = init_blob(user_id, date, filename)

    save_path = Path(save_path)
    if save_path.exists():
        return

    try:
        return blob.download_blob()
    except ResourceNotFoundError as e:
        raise ResourceNotFoundError(f"Blob for {user_id}-{date} not found.") from e


def get_selfie(dataframe_row: dict, save_dir: Path | str):
    path = Path(dataframe_row["full_path"])
    date = path.parts[3]
    filename = path.parts[5]
    user_id = dataframe_row["user_id"]

    save_dir = Path(save_dir)
    save_dir = save_dir / Path(f"{user_id}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / Path(f"{date}_{dataframe_row['selfie_link_id']}.jpg")

    try:
        return (
            download_blob(
                user_id=user_id,
                date=date,
                filename=filename,
                save_path=save_path,
            ),
            save_path,
        )
    except ResourceNotFoundError as e:
        raise ResourceNotFoundError(f"Blob for {user_id}-{date} not found.") from e


def get_selfies(df_selfies: pd.DataFrame, save_dir: Path | str = "../../data/selfies"):
    l = len(df_selfies)
    with tqdm(total=l, ncols=100) as pbar:
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = [
                executor.submit(get_selfie, row, save_dir=save_dir)
                for row in df_selfies.to_dict("records")
            ]
            for future in concurrent.futures.as_completed(futures):
                try:
                    data, save_path = future.result()
                    with open(save_path, "wb") as f:
                        data.readinto(f)
                    pbar.update(1)
                except Exception as e:
                    print(f"{future} raised an exception {e}")
                    pbar.update(1)
                    continue


# %%
def main_dl(cfg: DictConfig) -> None:
    df_selfies = get_user_selfie_data(cfg)
    df_selfies.to_csv("../../data/all_selfies.csv", index=False)
    df_clean_selfie_blobs = clean_selfie_blobs(df_selfies)
    df_clean_selfie_blobs.to_csv("../../data/downloaded_selfies.csv", index=False)
    get_selfies(df_clean_selfie_blobs, save_dir="../../data/selfies")


def main_csv() -> None:
    # df_selfies = pd.read_csv("../../data/all_selfies.csv")
    # df_clean_selfie_blobs = clean_selfie_blobs(df_selfies)
    # df_clean_selfie_blobs.to_csv("../../data/downloaded_selfies.csv", index=False)
    df_clean_selfie_blobs = pd.read_csv("../../data/downloaded_selfies.csv")
    get_selfies(df_clean_selfie_blobs, save_dir="../../data/selfies")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.selfie_data.download:
        main_dl(cfg)
        return
    main_csv()


if __name__ == "__main__":
    main()
