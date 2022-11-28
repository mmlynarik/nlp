from datetime import date
import warnings

import pandas as pd
import psycopg2

from topicmodel.queries import QUERY_OKRA_DATA_PG
from topicmodel.config import OKRA_DB


def read_okra_data_from_pg(date_from: date, date_to: date) -> pd.DataFrame:
    with warnings.catch_warnings():  # ignore warning for non-SQLAlchemy connecton, see pandas issue #45660
        warnings.simplefilter("ignore", UserWarning)
        with psycopg2.connect(**OKRA_DB) as conn:
            df_okra = pd.read_sql(QUERY_OKRA_DATA_PG.format(date_from=date_from, date_to=date_to), conn)
    return df_okra


class OKRADataModule:
    def __init__(self, date_from: date, date_to: date, batch_size: int = 64):
        self.date_from = date_from
        self.date_to = date_to
        self.batch_size = batch_size
        self.train_data = None
        self.val_data = None
        self.test_data = None


print(read_okra_data_from_pg(date_from=date(2011, 1, 1), date_to=date(2019, 1, 1)))
