# %%
import asyncio
import functools
from pathlib import Path

import pandas as pd
from azure.core.exceptions import ResourceNotFoundError
from azure.identity import ManagedIdentityCredential
from azure.storage.blob import BlobClient
from dl_conn import get_dl_conn
from dl_orm import DLorm

con = get_dl_conn()
datalake = DLorm(con)

credential = ManagedIdentityCredential()


def background(f):
    def wrapped(*args, **kwargs):
        loop = asyncio.get_event_loop()
        partial_func = functools.partial(f, *args, **kwargs)
        return loop.run_in_executor(None, partial_func)

    return wrapped


def init_blob(credential, user_id, date, filename, container_name: str = "selfies"):
    return BlobClient(
        account_url="https://claire1kstorage.blob.core.windows.net",
        container_name=container_name,
        blob_name=f"{date}/{user_id}/{filename}",
        credential=credential,
    )


def download_blob(blob, save_path: Path | str):
    if save_path.exists():
        pass

    with open(save_path, "wb") as f:
        try:
            data = blob.download_blob()
            data.readinto(f)
        except ResourceNotFoundError:
            raise


@background
def get_selfie(dataframe_row, save_directory: Path | str = "./data/selfies"):
    path = Path(dataframe_row["full_path"])
    date = path.parts[3]
    filename = path.parts[5]
    user_id = dataframe_row["user_id"]

    blob = init_blob(credential, user_id, date, filename)

    save_dir = Path(save_directory)
    save_dir = save_dir / Path(f"{user_id}")
    save_dir.mkdir(parents=True, exist_ok=True)
    save_path = save_dir / Path(f"{date}.jpg")

    try:
        download_blob(blob, save_path)
    except ResourceNotFoundError:
        print(f"Blob for {user_id}-{date} not found.")


def get_users_w_most_selfies(dataframe: pd.DataFrame, n_users: int):
    top_n_users = (
        dataframe.groupby("user_id")
        .nunique()["full_path"]
        .reset_index(name="n_selfies")
        .sort_values("n_selfies", ascending=False)
        .head(n_users)["user_id"]
        .values
    )
    return top_n_users


def get_df_users_w_most_selfies(dataframe: pd.DataFrame, n_users: int) -> pd.DataFrame:
    top_n_users = get_users_w_most_selfies(dataframe, n_users)
    return dataframe.loc[dataframe["user_id"].isin(top_n_users)]


query = """
SELECT [user_id]
      ,[full_path]
  FROM [pg].[selfie]
  WHERE error_code IS NULL
"""

# %%
df_selfies = datalake.get_query(query).sort_values("user_id")

# %%

df_top_100_users = get_df_users_w_most_selfies(df_selfies, 100)

# %%

for row in df_top_100_users.to_dict("records"):
    get_selfie(row, save_directory="./data/selfies")

# %%
