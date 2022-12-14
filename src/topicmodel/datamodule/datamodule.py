import os
import logging
from datetime import date, datetime
from typing import Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from dateutil.relativedelta import relativedelta

from topicmodel.datamodule.dataset import (
    OKRAWord2VecStringSentenceDataset,
    OKRAWord2VecEncodedSentenceDataset,
    get_corpus_tensor,
    get_keys_tensor,
    get_outputs_tensor,
    get_printable_string_dataset,
)
from topicmodel.datamodule.tokenizers import WordTokenizer
from topicmodel.datamodule.utils import (
    split_text_to_sentences_regex,
    expand_sentences_into_rows,
    read_okra_data_from_db,
)

log = logging.getLogger(__name__)
log.addHandler(logging.NullHandler())


class OKRAWord2VecDataModule:
    def __init__(
        self,
        date_from: date,
        date_to: date,
        period_val: relativedelta,
        period_test: relativedelta,
        vocab_size: int,
        embedding_dim: int,
        seq_len: int,
        cache_dir: str,
        batch_size: int = 64,
    ):
        self.date_from = date_from
        self.date_to = date_to
        self.period_val = period_val
        self.period_test = period_test
        self.vocab_size = vocab_size
        self.embedding_dim = embedding_dim
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.string_dataset = None
        self.val_data = None
        self.test_data = None

    @property
    def _path_filtered_data(self) -> str:
        return os.path.join(
            self.cache_dir, f"filtered_data_from_{self.date_from:%Y%m%d}_to_{self.date_to:%Y%m%d}.csv",
        )

    @property
    def _path_string_dataset(self) -> str:
        return os.path.join(self.cache_dir, "string_dataset.csv")

    @property
    def _path_word_counts(self) -> str:
        return os.path.join(self.cache_dir, "word_counts.csv")

    def _get_filtered_data(self) -> pd.DataFrame:
        df_filtered_data = pd.read_csv(self._path_filtered_data)
        return df_filtered_data

    def _filter_data(self, df_data: pd.DataFrame) -> pd.DataFrame:
        df_data = df_data.drop_duplicates(subset=["title", "text", "sentence"])
        df_data = df_data.drop(columns=["stars", "raw", "text", "sentences", "url"])

        df_data["date"] = pd.to_datetime(df_data["date"])
        df_data["year"] = df_data.date.dt.year

        return df_data

    def _create_input_tensor(self, record: pd.Series) -> tf.Tensor:
        raise NotImplementedError()

    def _create_output_tensor(self, record: pd.Series) -> tf.Tensor:
        raise NotImplementedError()

    def _get_string_dataset_tensors(self, df_data: pd.DataFrame) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        keys = df_data["id"]
        inputs = df_data["sentence"]
        outputs = -1 * np.ones_like(keys)
        return tf.constant(keys), tf.constant(inputs), tf.constant(outputs)

    def _get_split_masks(
        self, filtered_data: pd.DataFrame
    ) -> tuple[tuple[pd.Series, pd.Series, pd.Series], tuple[date, date, date, date]]:
        date_col = filtered_data["date"]
        date_from, date_to = datetime.fromisoformat(min(date_col)), datetime.fromisoformat(max(date_col))
        date_test_from = date_to - self.period_test
        date_val_from = date_test_from - self.period_val
        mask_train = date_col < date_val_from.isoformat()
        mask_val = (date_val_from.isoformat() <= date_col) & (date_col < date_test_from.isoformat())
        mask_test = date_test_from.isoformat() <= date_col
        return (mask_train, mask_val, mask_test), (date_from, date_val_from, date_test_from, date_to)

    def _get_word_counts(self) -> dict[str, int]:
        """
        Generate word counts dictionary. Special tokens are set to 0, because they need to be excluded from unigram sampling distribution calculated later.
        """
        index_counts = defaultdict(int)
        for sentence in get_corpus_tensor(self.encoded_dataset).numpy():
            for idx in sentence:
                index_counts[idx] += 1

        index_counts[0] = 0  # PAD
        index_counts[1] = 0  # UNK

        idx2word = self.tokenizer.idx2word  # Local variable prevents repetitive calculation of vocabulary
        return {idx2word[idx]: count for idx, count in sorted(index_counts.items())}

    def _get_encoded_dataset_tensors(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        encoded_corpus = self.tokenizer.encode(get_corpus_tensor(self.string_dataset))
        keys = get_keys_tensor(self.string_dataset)
        outputs = get_outputs_tensor(self.string_dataset)
        return keys, encoded_corpus, outputs

    def prepare_data(self) -> None:
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            log.info(f"Cache directory {self.cache_dir} created.")

        if os.path.exists(self._path_filtered_data):
            log.info(f"Filtered data already exists in {self._path_filtered_data} file")
            return

        # Read and preprocess data
        df_data = read_okra_data_from_db(date_from=self.date_from, date_to=self.date_to)
        df_data["sentences"] = df_data["text"].apply(lambda x: split_text_to_sentences_regex(x))
        df_data = df_data.pipe(expand_sentences_into_rows, outcol="sentence", idcol="id")

        # Filter data and export it to csv
        df_filtered_data = self._filter_data(df_data)
        df_filtered_data.to_csv(self._path_filtered_data, index=False)

    def setup(self, stage: Optional[str] = None):
        df_filtered_data = self._get_filtered_data()
        if stage is None or stage == "fit":
            masks, dates = self._get_split_masks(df_filtered_data)
            string_dataset_tensors = self._get_string_dataset_tensors(df_filtered_data[masks[0]])
            log.info(f"Training dataset has {len(string_dataset_tensors)} obs from {dates[0]}-{dates[1]}.")
            self.string_dataset = OKRAWord2VecStringSentenceDataset(string_dataset_tensors)

            tokenizer = WordTokenizer(max_tokens=self.vocab_size, seq_len=self.seq_len)
            tokenizer.adapt(data=get_corpus_tensor(self.string_dataset))
            self.tokenizer = tokenizer
            self.encoded_dataset = OKRAWord2VecEncodedSentenceDataset(self._get_encoded_dataset_tensors())

            self.word_counts = self._get_word_counts()

        if stage is None or stage == "validate":
            raise NotImplementedError("Validation set is not applicable in Word2Vec model training.")

        if stage is None or stage == "test":
            raise NotImplementedError("Test set is not applicable in Word2Vec model training.")

    def word_counts_to_csv(self):
        pd.DataFrame(self.word_counts, index=["count"]).T.reset_index().to_csv(
            self._path_word_counts, index=False, encoding="UTF-8-SIG"
        )

    def string_dataset_to_csv(self):
        pd.DataFrame(get_printable_string_dataset(self.string_dataset)).to_csv(
            self._path_string_dataset, index=False, encoding="UTF-8-SIG"
        )

    # def vocab_to_csv(self):
    #     vocab = {word: 1 for word in self.tokenizer.get_vocabulary()}
    #     pd.DataFrame(vocab, index=["count"]).T.reset_index().to_csv(
    #         os.path.join(DEFAULT_CACHE_DIR, "vocab.csv"), index=False, encoding="UTF-8-SIG"
    #     )
