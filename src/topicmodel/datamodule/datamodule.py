import os
import logging
from datetime import date, datetime
from typing import Optional

import pandas as pd
import numpy as np
import tensorflow as tf
from dateutil.relativedelta import relativedelta

from topicmodel.datamodule.dataset import TopicModelDataset, dataset_to_list_of_dicts, get_corpus_tensor
from word2vec.datamodule.utils import read_reviews_data_from_db, preprocess_text_document

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class TopicModelDataModule:
    def __init__(
        self,
        date_from: date,
        date_to: date,
        period_val: relativedelta,
        period_test: relativedelta,
        cache_dir: str,
    ):
        self.date_from = date_from
        self.date_to = date_to
        self.period_val = period_val
        self.period_test = period_test
        self.cache_dir = cache_dir
        self.train_dataset = None
        self.val_data = None
        self.test_data = None

    @property
    def _path_filtered_data(self) -> str:
        return os.path.join(
            self.cache_dir, f"top2vec_filtered_data_{self.date_from:%Y%m%d}_{self.date_to:%Y%m%d}.csv",
        )

    @property
    def _path_text_dataset(self) -> str:
        return os.path.join(self.cache_dir, "text_dataset.csv")

    def _get_filtered_data(self) -> pd.DataFrame:
        df_filtered_data = pd.read_csv(self._path_filtered_data)
        return df_filtered_data

    def _filter_data(self, df_data: pd.DataFrame) -> pd.DataFrame:
        df_data = df_data.drop_duplicates(subset=["doc"])
        df_data = df_data.drop(columns=["stars", "raw", "text", "url"])

        df_data["date"] = pd.to_datetime(df_data["date"])
        df_data["year"] = df_data.date.dt.year

        return df_data

    def _create_input_tensor(self, record: pd.Series) -> tf.Tensor:
        raise NotImplementedError()

    def _create_output_tensor(self, record: pd.Series) -> tf.Tensor:
        raise NotImplementedError()

    def _get_train_dataset_vectors(self, df_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keys = df_data["id"].to_numpy()
        inputs = df_data["doc"].to_numpy()
        outputs = -1 * np.ones_like(keys)
        return keys, inputs, outputs

    def _get_split_info(self, filtered_data: pd.DataFrame, split: str) -> tuple[pd.Series, date, date]:
        """Calculate mask, date_from and date_to for train, val or test data split."""
        date_col = filtered_data["date"]
        date_from, date_to = datetime.fromisoformat(min(date_col)), datetime.fromisoformat(max(date_col))
        date_test_from = date_to - self.period_test
        date_val_from = date_test_from - self.period_val
        mask_train = date_col < date_val_from.isoformat()

        if split == "train":
            return mask_train, date_from, date_val_from
        elif split == "val":
            mask_val = (date_val_from.isoformat() <= date_col) & (date_col < date_test_from.isoformat())
            return mask_val, date_val_from, date_test_from
        elif split == "test":
            mask_test = date_test_from.isoformat() <= date_col
            return mask_test, date_test_from, date_to
        raise ValueError("Split must be either train, val or test.")

    def prepare_data(self) -> None:
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            log.info(f"Cache directory {self.cache_dir} created.")

        if os.path.exists(self._path_filtered_data):
            log.info(f"Filtered data already exists in {self._path_filtered_data} file")
            return

        df_data = read_reviews_data_from_db(date_from=self.date_from, date_to=self.date_to)
        df_data["doc"] = df_data["text"].apply(lambda x: preprocess_text_document(x))
        df_filtered_data = self._filter_data(df_data)
        df_filtered_data.to_csv(self._path_filtered_data, index=False)

    def setup(self, stage: Optional[str] = None) -> None:
        df_filtered_data = self._get_filtered_data()
        if stage is None or stage == "fit":
            mask, date_from, date_to = self._get_split_info(df_filtered_data, "train")
            train_vectors = self._get_train_dataset_vectors(df_filtered_data[mask])
            log.info(f"Top2Vec training dataset has {len(train_vectors)} obs, from {date_from}-{date_to}.")
            self.train_dataset = TopicModelDataset(train_vectors)
            log.info(f"Train dataset size: {len(self.train_dataset)}")
            self.print_summary()

        if stage is None or stage == "validate":
            raise NotImplementedError("Validation set is not applicable in Word2Vec model.")

        if stage is None or stage == "test":
            raise NotImplementedError("Test set is not applicable in Word2Vec model.")

    def text_dataset_to_csv(self) -> None:
        data = dataset_to_list_of_dicts(self.train_dataset)
        pd.DataFrame(data).to_csv(self._path_text_dataset, index=False, encoding="UTF-8-SIG")

    def get_top2vec_input(self) -> list[str]:
        list_of_dicts = dataset_to_list_of_dicts(self.train_dataset)
        return [example["doc"] for example in list_of_dicts]

    def print_summary(self, n: int = 10):
        documents_lengths = sorted(len(s.numpy()) for s in get_corpus_tensor(self.train_dataset))
        print(f"Number of documents: {len(self.train_dataset):,.0f}")
        print(f"Number of tokens: {sum(documents_lengths):,.0f}")
        print(f"Avg number of tokens per document: {sum(documents_lengths)/len(self.train_dataset):,.2f}")
        print(f"Document lengths top-{n}: {documents_lengths[-n:]}")
        print(f"Document lengths bot-{n}: {documents_lengths[:n]}")

    def debug(self):
        self.print_summary()
        self.text_dataset_to_csv()
