import os
import warnings
import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
import tensorflow as tf
import tqdm
import psycopg2

from topicmodel.queries import QUERY_OKRA_DATA_PG
from topicmodel.config import OKRA_DB
from topicmodel.dataset import OKRAWord2VecDataset

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def read_okra_data_from_db(date_from: date, date_to: date) -> pd.DataFrame:
    with warnings.catch_warnings():  # ignore pandas issue #45660
        warnings.simplefilter("ignore", UserWarning)
        with psycopg2.connect(**OKRA_DB) as conn:
            df_okra = pd.read_sql(QUERY_OKRA_DATA_PG.format(date_from=date_from, date_to=date_to), conn)
    return df_okra


class OKRADataModule:
    def __init__(
        self,
        date_from: date,
        date_to: date,
        period_val_yrs: int,
        period_test_yrs: int,
        batch_size: int = 64,
        cache_dir: str = "./data",
    ):
        self.date_from = date_from
        self.date_to = date_to
        self.period_val_yrs = period_val_yrs
        self.period_test_yrs = period_test_yrs
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.train_data = None
        self.val_data = None
        self.test_data = None

    @property
    def _path_filtered_data(self) -> str:
        return os.path.join(
            self.cache_dir, f"filtered_data_from_{self.date_from:%Y%m%d}_to_{self.date_to:%Y%m%d}.csv",
        )

    def get_filtered_data(self) -> pd.DataFrame:
        df_filtered_data = pd.read_csv(self._path_filtered_data)
        return df_filtered_data

    def filter_data(self, df_data: pd.DataFrame) -> pd.DataFrame:
        df_data = df_data.drop_duplicates(subset=["title", "text"])
        df_data = df_data.drop(columns=["stars"])

        df_data["date"] = pd.to_datetime(df_data["date"])
        df_data["year"] = df_data.date.dt.year
        df_data["date"] = df_data.date.dt.date

        return df_data

    def _create_input_tensor(self, record: pd.Series) -> tf.Tensor:
        return tf.constant([record["text"]])

    def _create_output_tensor(self, record: pd.Series) -> tf.Tensor:
        return tf.constant([1])

    def _extract_tensors(self, df_data: pd.DataFrame, tqdm_desc: str) -> list[tuple[tf.Tensor, tf.Tensor]]:
        vectors = [
            (record["id"], self._create_input_tensor(record), self._create_output_tensor(record))
            for _, record in tqdm(df_data.iterrows(), desc=tqdm_desc, total=len(df_data))
        ]
        return vectors

    def prepare_data(self) -> None:
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            log.info(f"Cache directory {self.cache_dir} created.")

        if os.path.exists(self._path_filtered_data):
            log.info(f"Filtered data already exists in {self._path_filtered_data} file")
            return

        df_okra = read_okra_data_from_db(date_from=self.date_from, date_to=self.date_to)

        df_filtered_data = self.filter_data(df_okra)
        df_filtered_data.to_csv(self._path_filtered_data, index=False)

    def setup(self, stage: Optional[str] = None):
        df_filtered_data = self.get_filtered_data()

        date_start = datetime.fromtimestamp(min(df_filtered_data["date"]))
        date_end = datetime.fromtimestamp(max(df_filtered_data["date"]))
        dt_test_start = date_end - self.period_test_yrs
        dt_val_start = dt_test_start - self.period_val_yrs

        if stage is None or stage == "fit":
            mask_train = df_filtered_data["date"] < dt_val_start.timestamp()
            train_tensors = self._extract_tensors(df_filtered_data[mask_train], tqdm_desc="Loading trn data")
            log.info(
                f"Training dataset consists of {len(train_tensors)} reviews "
                f"between {date_start} to {dt_val_start}."
            )
            self.train_data = OKRAWord2VecDataset(train_tensors)

        if stage is None or stage == "validate":
            mask_val = (dt_val_start.timestamp() <= df_filtered_data["heat_datetime"]) & (
                df_filtered_data["heat_datetime"] < dt_test_start.timestamp()
            )
            val_tensors = self._extract_tensors(
                df_filtered_data[mask_val], tqdm_desc="Extracting validation vectors"
            )
            log.info(
                f"Validation dataset consists of {len(val_tensors)} heats "
                f"between {dt_val_start} to {dt_test_start} (heats with rare scrap were excluded)."
            )
            self.val_data = OKRAWord2VecDataset(val_tensors)

        if stage is None or stage == "test":
            mask_test = (dt_test_start.timestamp() <= df_filtered_data["heat_datetime"]) & ~df_filtered_data[
                "has_rare_scrap"
            ]
            test_tensors = self._extract_tensors(df_filtered_data[mask_test], tqdm_desc="Loading test data")
            log.info(
                f"Test dataset consists of {len(test_tensors)} reviews between {dt_test_start} to {date_end}"
            )
            self.test_data = OKRAWord2VecDataset(test_tensors)
