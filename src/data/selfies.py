# %%
import concurrent.futures
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import hydra
import mediapipe as mp
import pandas as pd
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
from dl_conn import get_dl_conn
from dl_orm import DLorm
from omegaconf import DictConfig

con = get_dl_conn()
datalake = DLorm(con)

credential = ManagedIdentityCredential()


def get_user_selfie_data(cfg: DictConfig) -> pd.DataFrame:
    query = f"""
    SELECT pgsfe.user_id
            ,pgsfe.ts_date
            ,pgsfe.id
            ,pgsfe.full_path
            ,pgsfe.selfie_link_id
            ,u.nr_selfies
        FROM pg.selfie as pgsfe
        JOIN pg.users as u ON u.user_id = pgsfe.user_id

        WHERE pgsfe.error_code IS {cfg.dl_filters.error_code} 
        AND pgsfe.anonymization_date IS {cfg.dl_filters.anonymization_date}
        AND pgsfe.ts_date 
            BETWEEN '{cfg.dl_filters.earliest_ts_date}' AND '{cfg.dl_filters.latest_ts_date}'
        AND u.participant_type = '{cfg.dl_filters.participant_type}'
        AND u.nr_selfies > {cfg.dl_filters.nr_selfies}
        AND pgsfe.selfie_link_id in (SELECT DISTINCT selfie_link_id FROM pg.measure_procedure)
    """
    return datalake.get_query(query).sort_values("user_id")


"""
SELECT pgsfe.[user_id]
        ,pgsfe.[ts_date]
        ,pgsfe.[id]
        ,pgsfe.[full_path]
    FROM [pg].[selfie] as pgsfe
    JOIN pg.users as u ON u.user_id = pgsfe.user_id
    JOIN pg.measure_procedure as pgmp ON pgmp.user_id = pgsfe.user_id

    WHERE error_code IS NULL
    AND pgsfe.anonymization_date IS NOT NULL
    AND pgsfe.ts_date 
        BETWEEN '2019-01-01' AND '2023-07-17'
    AND u.participant_type = 'SKINLY'
    AND pgsfe.selfie_link_id in (select distinct selfie_link_id from pg.measure_procedure)
    AND pgsfe.user_id in (SELECT user_id
            FROM pg.measure_procedure
            GROUP BY user_id
            HAVING COUNT(DISTINCT selfie_link_id) > 50) 
"""


def init_blob(
    credential: ManagedIdentityCredential,
    user_id: int,
    date: str,
    filename: str,
    container_name: str = "selfies",
):
    return BlobClient(
        account_url="https://claire1kstorage.blob.core.windows.net",
        container_name=container_name,
        blob_name=f"{date}/{user_id}/{filename}",
        credential=credential,
    )


def download_blob(blob, save_path: Path | str):
    save_path = Path(save_path)
    if save_path.exists():
        return

    with open(save_path, "wb") as f:
        try:
            data = blob.download_blob()
            data.readinto(f)
        except ResourceNotFoundError:
            raise


def get_selfie(dataframe_row: dict, save_dir: Path | str):
    path = Path(dataframe_row["full_path"])
    date = path.parts[3]
    filename = path.parts[5]
    user_id = dataframe_row["user_id"]

    blob = init_blob(credential, user_id, date, filename)

    save_dir = Path(save_dir)
    save_dir = save_dir / Path(f"{user_id}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / Path(f"{date}.jpg")

    try:
        download_blob(blob, save_path)
        return save_path
    except ResourceNotFoundError:
        # print(f"Blob for {user_id}-{date} not found.")
        return Exception(f"Blob for {user_id}-{date} not found.")


def get_selfies(df_selfies: pd.DataFrame, save_dir: Path | str = "../../data/selfies"):
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [
            executor.submit(get_selfie, row, save_dir=save_dir)
            for row in df_selfies.to_dict("records")
        ]
        for future in concurrent.futures.as_completed(futures):
            try:
                selfie_path = future.result()
                print(selfie_path)
            except Exception as e:
                print(f"{future} raised an exception {e}")
                continue


def validate_selfie(path: Path | str):
    path = Path(path)
    try:
        mp.Image.create_from_file(path.as_posix())
    except RuntimeError:
        path.unlink()


def validate_selfies(paths: list[Path | str]):
    with ThreadPoolExecutor(max_workers=12) as executor:
        futures = [executor.submit(validate_selfie, path) for path in paths]
        for future in concurrent.futures.as_completed(futures):
            try:
                selfie_path = future.result()
            except Exception as e:
                print(f"{future} raised an exception {e}")
                continue


# %%
with hydra.initialize(version_base=None, config_path="../../config"):
    cfg = hydra.compose(config_name="config")
    print(cfg.dl_filters)
df_selfies = get_user_selfie_data(cfg)
df_selfies
# %%

df_selfienrcomp = (
    df_selfies[["user_id", "id", "full_path", "selfie_link_id"]]
    .groupby("user_id")
    .nunique()
    .reset_index()
)
df_selfienrcomp
# %%


df_selfienrcomp = df_selfienrcomp.merge(
    df_selfies[["user_id", "nr_selfies"]], on="user_id", how="left"
).drop_duplicates()
df_selfienrcomp
# %%
print(df_selfienrcomp.head(10).to_markdown())
# %%
df_grouped_nunique = df_selfies.groupby("selfie_link_id").nunique().reset_index()
# %%
df_grouped_nunique.loc[df_grouped_nunique["full_path"] > 1]
# %%
df_selfies.loc[df_selfies.duplicated("full_path")]
# %%

# paths = list(Path("./data/selfies/").rglob("*.jpg"))
# paths = [path.as_posix() for path in paths]

# corrupted_images = []
# for path in tqdm(paths):
#     try:
#         mp.Image.create_from_file(path)
#     except RuntimeError:
#         corrupted_images.append(path)
# # %%
# pd.DataFrame(corrupted_images, columns=["path"]).to_csv(
#     "./corrupted_files.csv", index=False
# )
# %%
