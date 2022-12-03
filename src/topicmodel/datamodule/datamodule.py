import os
import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
import tensorflow as tf
from tqdm.auto import tqdm
from dateutil.relativedelta import relativedelta

from topicmodel.config import SEQ_LEN, VOCAB_SIZE
from topicmodel.datamodule.dataset import OKRAWord2VecDataset
from topicmodel.datamodule.tokenizers import WordTokenizer
from topicmodel.datamodel import DateSpan
from topicmodel.datamodule.utils import text_to_sentences, expand_sentences_into_rows, read_okra_data_from_db

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


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
        df_data = df_data.pipe(expand_sentences_into_rows, outcol="sentence")

        # Filter data and export it to csv
        df_filtered_data = self._filter_data(df_data)
        df_filtered_data.to_csv(self._path_filtered_data, index=False)

    def get_split_mask(self, filtered_data: pd.DataFrame, split: str) -> tuple[pd.Series, DateSpan]:
        date_col = filtered_data["date"]
        date_from, date_to = datetime.fromisoformat(min(date_col)), datetime.fromisoformat(max(date_col))
        date_test_from = date_to - self.period_test
        date_val_from = date_test_from - self.period_val
        mask_train = date_col < date_val_from.isoformat()
        mask_val = (date_val_from.isoformat() <= date_col) & (date_col < date_test_from.isoformat())
        mask_test = date_test_from.isoformat() <= date_col

        if split == "train":
            return mask_train, DateSpan(date_from, date_val_from)
        elif split == "val":
            return mask_val, DateSpan(date_val_from, date_test_from)
        elif split == "test":
            return mask_test, DateSpan(date_test_from, date_to)
        raise ValueError("Incorrect split parameter value provided. Use one of train, val or test.")

    def setup(self, stage: Optional[str] = None):
        df_filtered_data = self._get_filtered_data()
        if stage is None or stage == "fit":
            mask, span = self.get_split_mask(df_filtered_data, split="train")
            train_tensors = self._extract_tensors(df_filtered_data[mask], desc="Setting up train dataset")
            log.info(f"Train dataset: {len(train_tensors)} reviews from {span.date_from} to {span.date_to}.")
            self.train_data = OKRAWord2VecDataset(train_tensors)

            self.tokenizer = WordTokenizer(max_tokens=VOCAB_SIZE, out_seq_len=SEQ_LEN, output_mode="int")
            self.tokenizer.adapt(self.train_data.map(lambda key, x, y: x))

        if stage is None or stage == "validate":
            val_mask, span = self.get_split_mask(df_filtered_data, split="val")
            val_tensors = self._extract_tensors(df_filtered_data[val_mask], desc="Setting up val dataset")
            log.info(f"Val dataset: {len(val_tensors)} reviews from {span.date_from} to {span.date_to}")
            self.val_data = OKRAWord2VecDataset(val_tensors)

        if stage is None or stage == "test":
            test_mask, span = self.get_split_mask(df_filtered_data, split="test")
            test_tensors = self._extract_tensors(df_filtered_data[test_mask], desc="Setting up test dataset")
            log.info(f"Test dataset: {len(test_tensors)} reviews from {span.date_from} to {span.date_to}")
            self.test_data = OKRAWord2VecDataset(test_tensors)
