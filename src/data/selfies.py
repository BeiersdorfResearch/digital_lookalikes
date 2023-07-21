# %%
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
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

con = get_dl_conn()
datalake = DLorm(con)


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
    df_selfies = datalake.get_query(query).sort_values("user_id")
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
    credential = ManagedIdentityCredential()
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


def sample_user_selfies(df_selfies: pd.DataFrame) -> pd.DataFrame:
    df_latest_selfie = df_selfies.sort_values("ts_date").groupby("user_id").tail(1)
    df_random_selfies = (
        df_selfies.loc[
            ~df_selfies["selfie_link_id"].isin(df_latest_selfie["selfie_link_id"])
        ]
        .groupby("user_id")
        .sample(2)
    )
    return pd.concat([df_latest_selfie, df_random_selfies]).sort_values("user_id")


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

    with open(save_path, "wb") as f:
        try:
            data = blob.download_blob()
            data.readinto(f)
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
    save_path = save_dir / Path(f"{date}{dataframe_row['selfie_link_id']}.jpg")

    try:
        download_blob(
            user_id=user_id,
            date=date,
            filename=filename,
            save_path=save_path,
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
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"{future} raised an exception {e}")
                    pbar.update(1)
                    continue


def validate_selfie(path: Path | str):
    path = Path(path)
    try:
        img = Image.open(path.as_posix())
        img.verify()
    except (IOError, SyntaxError) as e:
        raise


def validate_selfies(paths: list[Path | str]):
    l = len(paths)
    with tqdm(total=l, ncols=100) as pbar:
        with ThreadPoolExecutor(max_workers=40) as executor:
            futures = [executor.submit(validate_selfie, path) for path in paths]
            for future in concurrent.futures.as_completed(futures):
                try:
                    future.result()
                    pbar.update(1)
                except Exception as e:
                    print(f"{future} raised an exception {e}")
                    pbar.update(1)
                    continue


# %%
# with hydra.initialize(version_base=None, config_path="../../config"):
#     cfg = hydra.compose(config_name="config")
#     print(cfg.dl_filters)
# df_selfies = get_user_selfie_data(cfg)
# df_selfies.to_csv("../../data/selfies.csv", index=False)
df_selfies = pd.read_csv("../../data/selfies.csv")
#%%
df_filtered_latest = filter_bad_blobs(df_selfies.sort_values("ts_date").groupby("user_id").tail(2))
#%%
df_random_selfies = (
    df_selfies.loc[
        ~df_selfies["selfie_link_id"].isin(df_filtered_latest["selfie_link_id"])
    ]
    .groupby("user_id")
    .sample(5)
)
# %%
df_filtered_random = filter_bad_blobs(df_random_selfies)
# %%
df_nr_random_selfies = df_filtered_random.groupby("user_id").nunique()["selfie_link_id"].reset_index()
df_nr_random_selfies.loc[df_nr_random_selfies["selfie_link_id"] < 2].info()
# %%
df_nr_latest_selfies = df_filtered_latest.groupby("user_id").nunique()["selfie_link_id"].reset_index()
df_nr_latest_selfies.loc[df_nr_latest_selfies["selfie_link_id"] < 1]
# %%
# @hydra.main(config_path="../../config", config_name="config", version_base=None)
# def main(cfg: DictConfig):
#     df_selfies = get_user_selfie_data(cfg)
#     df_filtered_blobs = filter_bad_blobs(df_selfies)


# if __name__ == "__main__":
#     main()

# %%
