import os
import warnings
import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
import tensorflow as tf
import psycopg2
from tqdm.auto import tqdm
from dateutil.relativedelta import relativedelta

from topicmodel.datamodule.queries import QUERY_OKRA_DATA_PG
from topicmodel.config import OKRA_DB, MAX_SEQ_LEN, VOCAB_SIZE
from topicmodel.datamodule.dataset import OKRAWord2VecDataset
from topicmodel.datamodule.tokenizers import WordTokenizer
from topicmodel.utils import text_to_sentences, expand_sentences_into_rows

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


def read_okra_data_from_db(date_from: date, date_to: date) -> pd.DataFrame:
    with warnings.catch_warnings():  # ignore pandas issue #45660
        warnings.simplefilter("ignore", UserWarning)
        with psycopg2.connect(**OKRA_DB) as conn:
            df_okra = pd.read_sql(QUERY_OKRA_DATA_PG.format(date_from=date_from, date_to=date_to), conn)
    return df_okra


class OKRAWord2VecDataModule:
    def __init__(
        self,
        date_from: date,
        date_to: date,
        period_val: relativedelta,
        period_test: relativedelta,
        cache_dir: str,
        batch_size: int = 64,
    ):
        self.date_from = date_from
        self.date_to = date_to
        self.period_val = period_val
        self.period_test = period_test
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

    def _get_filtered_data(self) -> pd.DataFrame:
        df_filtered_data = pd.read_csv(self._path_filtered_data)
        return df_filtered_data

    def _filter_data(self, df_data: pd.DataFrame) -> pd.DataFrame:
        df_data = df_data.drop_duplicates(subset=["title", "text", "sentence"])
        df_data = df_data.drop(columns=["stars", "raw", "text", "sentences", "url"])

        df_data["date"] = pd.to_datetime(df_data["date"])
        df_data["year"] = df_data.date.dt.year

        return df_data

    def _create_input_tensor(self, record: pd.Series) -> str:
        return record["sentence"]

    def _create_output_tensor(self, record: pd.Series) -> int:
        return -1  # not applicable for word2vec

    def _extract_tensors(self, df_data: pd.DataFrame, desc: str) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        ids, inputs, outputs = [], [], []
        for _, record in tqdm(df_data.iterrows(), desc=desc, total=len(df_data)):
            ids.append(record["id"])
            inputs.append(self._create_input_tensor(record))
            outputs.append(self._create_output_tensor(record))

        return tf.constant(ids), tf.constant(inputs), tf.constant(outputs)

    def prepare_data(self) -> None:
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            log.info(f"Cache directory {self.cache_dir} created.")

        if os.path.exists(self._path_filtered_data):
            log.info(f"Filtered data already exists in {self._path_filtered_data} file")
            return

        # Read and preprocess data
        df_data = read_okra_data_from_db(date_from=self.date_from, date_to=self.date_to)
        df_data["sentences"] = df_data["text"].apply(lambda x: text_to_sentences(x))
        df_data = df_data.pipe(expand_sentences_into_rows)

        # Filter and export data to csv
        df_filtered_data = self._filter_data(df_data)
        df_filtered_data.to_csv(self._path_filtered_data, index=False)

        tokenizer = WordTokenizer(vocab_size=VOCAB_SIZE, max_seq_len=MAX_SEQ_LEN)

    def setup(self, stage: Optional[str] = None):
        df_filtered_data = self._get_filtered_data()

        date_col = df_filtered_data["date"]
        date_start, date_end = datetime.fromisoformat(min(date_col)), datetime.fromisoformat(max(date_col))
        dt_test_start = date_end - self.period_test
        dt_val_start = dt_test_start - self.period_val

        if stage is None or stage == "fit":
            mask_train = date_col < dt_val_start.isoformat()
            train_tensors = self._extract_tensors(df_filtered_data[mask_train], desc="Loading trn data")
            log.info(
                f"Training dataset has {len(train_tensors)} reviews between {date_start} to {dt_val_start}."
            )
            self.train_data = OKRAWord2VecDataset(train_tensors)

        if stage is None or stage == "validate":
            mask_val = (dt_val_start.isoformat() <= date_col) & (date_col < dt_test_start.isoformat())
            val_tensors = self._extract_tensors(df_filtered_data[mask_val], desc="Loading val vectors")
            log.info(
                f"Validation dataset has {len(val_tensors)} reviews between {dt_val_start} {dt_test_start}"
            )
            self.val_data = OKRAWord2VecDataset(val_tensors)

        if stage is None or stage == "test":
            mask_test = dt_test_start.isoformat() <= date_col
            test_tensors = self._extract_tensors(df_filtered_data[mask_test], desc="Loading test data")
            log.info(f"Test dataset has {len(test_tensors)} reviews between {dt_test_start} to {date_end}")
            self.test_data = OKRAWord2VecDataset(test_tensors)
