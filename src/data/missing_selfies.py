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
import os

credential = ManagedIdentityCredential()

def list_files_with_extension(directory, extension) -> list:
    files = []
    for filename in os.listdir(directory):
        if filename.endswith(extension):
            files.append(os.path.join(directory, filename))
    return files


def get_users_missing_selfies(cfg: DictConfig) -> pd.DataFrame:
    selfies_dir = Path(cfg.selfie_data.save_dir)
    users_selfie_list = []
    for user in selfies_dir.glob('*'):
        selfies = [user] + list_files_with_extension(user, '.jpg')
        users_selfie_list.append(selfies)
    df_selfiepaths = pd.DataFrame(users_selfie_list, columns=['user_id', 'selfie1', 'selfie2', 'selfie3'])
    df_selfiepaths['user_id'] = df_selfiepaths['user_id'].apply(lambda x : os.path.basename(x)).astype(int)
    df_selfiepaths['selfie_link_id_1'] = df_selfiepaths['selfie1'].apply(lambda x : os.path.basename(x).split('_',1)[1].split('.')[0] if isinstance(x, str) else None)
    df_selfiepaths['selfie_link_id_2'] = df_selfiepaths['selfie2'].apply(lambda x : os.path.basename(x).split('_',1)[1].split('.')[0] if isinstance(x, str) else None)
    df_selfiepaths['selfie_link_id_3'] = df_selfiepaths['selfie3'].apply(lambda x : os.path.basename(x).split('_',1)[1].split('.')[0] if isinstance(x, str) else None)
    df_selfiepaths['missing_count'] = df_selfiepaths[['selfie1', 'selfie2', 'selfie3']].isnull().sum(axis=1)
    users_missing_selfies = df_selfiepaths[df_selfiepaths['selfie3'].isnull()][['user_id', 'selfie_link_id_1', 'selfie_link_id_2', 'selfie_link_id_3', 'missing_count']]
    return users_missing_selfies


def get_user_specific_selfie_data(cfg: DictConfig, users_missing: pd.DataFrame) -> pd.DataFrame:
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
    users_missing = users_missing
    df_selfies = df_selfies.merge(users_missing, on = 'user_id', how = 'inner')
    df_selfies['selfie_exists'] = ((df_selfies['selfie_link_id'] == df_selfies['selfie_link_id_1']) | (df_selfies['selfie_link_id'] == df_selfies['selfie_link_id_2']))
    return df_selfies


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


def clean_missing_selfie_blobs(df_selfies: pd.DataFrame) -> pd.DataFrame:
    df_filtered_latest = filter_bad_blobs(
        df_selfies.sort_values("ts_date").groupby("user_id").tail(2)
    ).drop_duplicates(subset=["user_id"])
    df_latest = df_filtered_latest[~df_filtered_latest['selfie_exists']]
    df_filtered_random = filter_bad_blobs(
        df_selfies.loc[
            ~df_selfies["selfie_link_id"].isin(df_latest["selfie_link_id"])  
        ]
        .groupby("user_id")
        .sample(5)
    )

    latest_selfie_users = df_latest.user_id.unique()
    df_filtered_random['missing_count'][df_filtered_random["user_id"].isin(latest_selfie_users)] -= 1

    df_one_sampled_random = (
        df_filtered_random[df_filtered_random['missing_count']==1 & ~df_filtered_random['selfie_exists']]
        .groupby('user_id')
        .sample(1)
    )
    df_two_sampled_random = (
        df_filtered_random[df_filtered_random['missing_count']==2 & ~df_filtered_random['selfie_exists']]
        .groupby('user_id')
        .sample(2)
    )
    return pd.concat([df_latest, df_one_sampled_random, df_two_sampled_random])


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


def get_selfies(df_selfies: pd.DataFrame, save_dir: Path | str = "./selfies"):
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
    users_missing_selfies = get_users_missing_selfies(cfg)
    users_missing_selfies.to_csv('./users_missing_selfies.csv', index=False)
    df_selfies = get_user_specific_selfie_data(cfg, users_missing_selfies)
    df_selfies.to_csv("./all_missing_selfies.csv", index=False)
    df_clean_selfie_blobs = clean_missing_selfie_blobs(df_selfies)
    df_clean_selfie_blobs.to_csv("./downloaded_missing_selfies.csv", index=False)
    get_selfies(df_clean_selfie_blobs, save_dir="./selfies")


def main_csv() -> None:
    df_clean_selfie_blobs = pd.read_csv("./downloaded_missing_selfies.csv")
    get_selfies(df_clean_selfie_blobs, save_dir="./selfies")


@hydra.main(config_path="../../config", config_name="config", version_base=None)
def main(cfg: DictConfig) -> None:
    if cfg.selfie_data.download:
        main_dl(cfg)
        return
    main_csv()


if __name__ == "__main__":
    main()
