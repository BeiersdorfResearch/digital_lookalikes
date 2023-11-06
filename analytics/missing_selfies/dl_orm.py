import pandas as pd


class DLorm:
    def __init__(self, connector) -> None:
        self.connector = connector

    def get_query(self, query: str, **kwargs) -> pd.DataFrame:
        return pd.read_sql(query, con=self.connector, **kwargs)
