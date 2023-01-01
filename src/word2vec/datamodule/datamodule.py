import os
import math
import logging
import random
from datetime import date, datetime
from typing import Optional
from collections import defaultdict

import pandas as pd
import numpy as np
import tensorflow as tf
from tqdm.auto import tqdm
from dateutil.relativedelta import relativedelta

from word2vec.datamodule.dataset import (
    OKRAWord2VecTextDataset,
    OKRAWord2VecEncodedDataset,
    get_corpus_tensor,
    get_keys_tensor,
    get_outputs_tensor,
    get_encoded_sequences,
    text_dataset_to_list_of_dicts,
)
from word2vec.datamodule.tokenizers import WordTokenizer
from word2vec.datamodule.utils import (
    read_okra_data_from_db,
    preprocess_text_document,
    split_text_to_sentences,
    expand_sentences_into_rows,
    generate_multinomial_sample,
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
        min_count: int,
        num_neg_samples: int,
        scaling_factor: float,
        context_window_size: int,
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
        self.min_count = min_count
        self.num_neg_samples = num_neg_samples
        self.scaling_factor = scaling_factor
        self.context_window_size = context_window_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        self.cache_dir = cache_dir
        self.text_dataset = None
        self.val_data = None
        self.test_data = None

    @property
    def _path_filtered_data(self) -> str:
        return os.path.join(
            self.cache_dir, f"filtered_data_from_{self.date_from:%Y%m%d}_to_{self.date_to:%Y%m%d}.csv",
        )

    @property
    def _path_string_dataset(self) -> str:
        return os.path.join(self.cache_dir, "text_dataset.csv")

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

    def _get_text_dataset_vectors(self, df_data: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        keys = df_data["id"].to_numpy()
        inputs = df_data["sentence"].to_numpy()
        outputs = -1 * np.ones_like(keys)
        return keys, inputs, outputs

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

    def _get_encoded_dataset_tensors(self) -> tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
        encoded_corpus = self.tokenizer.encode(get_corpus_tensor(self.text_dataset))
        keys = get_keys_tensor(self.text_dataset)
        outputs = get_outputs_tensor(self.text_dataset)
        return keys, encoded_corpus, outputs

    def _get_truncated_word_counts(self) -> np.ndarray:
        """
        Generate word counts array for words with at least `min_count` absolute frequency in the corpus. Special tokens [PAD] and [UNK] on indices 0 and 1 need to be excluded from unigram sampling distribution calculated later so their word counts are zeroed out.
        """
        index_counts: dict[int, int] = defaultdict(int)
        sentences = get_corpus_tensor(self.encoded_dataset).numpy()
        for sentence in sentences:
            for idx in sentence:
                index_counts[idx] += 1

        index_counts[0] = 0  # PAD
        index_counts[1] = 0  # UNK

        last_token_id = max(token_id for token_id, count in index_counts.items() if count >= self.min_count)
        return np.array(
            [count for token_id, count in sorted(index_counts.items()) if token_id <= last_token_id]
        )

    def _get_truncated_word_counts_as_dict(self) -> dict[str, int]:
        idx2word = self.tokenizer.idx2word
        word_counts = self._get_truncated_word_counts()
        return {idx2word[token_id]: count for token_id, count in enumerate(word_counts)}

    def _get_noise_distribution_probas(self, scaling_factor: float) -> np.ndarray:
        """Calculate probabilities of scaled unigram distribution used for sampling noise training examples"""
        word_counts = self._get_truncated_word_counts()
        scaled_word_counts = np.power(word_counts, scaling_factor)
        return scaled_word_counts / sum(scaled_word_counts)

    def _clean_sequences(self, sequences: list[list[int]], last_token_id: int) -> list[list[int]]:
        """Remove from sequences token_ids for padding and low-frequency words."""
        return [[token_id for token_id in seq if 0 < token_id <= last_token_id] for seq in sequences]

    @staticmethod
    def get_keep_proba(token_id: int, word_counts: np.ndarray, threshold: float) -> float:
        """See https://github.com/chrisjmccormick/word2vec_commented/blob/master/word2vec.c#L914 for detail"""
        total_word_counts = sum(word_counts)
        return (
            (math.sqrt(word_counts[token_id] / (threshold * total_word_counts)) + 1)
            * (threshold * total_word_counts)
            / word_counts[token_id]
        )

    def _subsample_frequent_words(
        self, sequences: list[list[int]], word_counts: np.ndarray, threshold: float = 0.001
    ) -> list[list[int]]:
        """
        Final transformation of sequences by subsampling frequent words using formula from original Word2Vec paper. The token_id is kept in the sequence if its keep probability is greater than randomly generated probability. Discarded token_ids will therefore not influence generation of skipgrams neither as target nor as context word in line with original paper.
        """
        return [
            [
                token_id
                for token_id in seq
                if self.get_keep_proba(token_id, word_counts, threshold) >= random.random()
            ]
            for seq in sequences
        ]

    def _generate_negative_samples(self, noise_probas: np.ndarray, exclude_token_ids: list[int]) -> list[int]:
        """Generate `num_neg_samples` negative examples of context word used during training."""
        samples = set()
        while len(samples) < self.num_neg_samples:
            sample = generate_multinomial_sample(noise_probas)
            if sample not in exclude_token_ids:
                samples.add(sample)
        return sorted(samples)

    def _get_sequences_for_skipgrams(self) -> list[list[int]]:
        """Prepare encoded sequences for skipgrams generation and negative sampling."""
        word_counts = self._get_truncated_word_counts()
        sequences = get_encoded_sequences(self.encoded_dataset)
        cleaned_sequences = self._clean_sequences(sequences, last_token_id=len(word_counts) - 1)
        subsampled_sequences = self._subsample_frequent_words(cleaned_sequences, word_counts)
        return subsampled_sequences

    def _generate_skipgrams(self, sequence: list[int]) -> list[tuple[int, int]]:
        couples = []
        for i, token_i in enumerate(sequence):
            window_start = max(0, i - self.context_window_size)
            window_end = min(len(sequence), i + self.context_window_size + 1)
            for j in range(window_start, window_end):
                if j != i:
                    token_j = sequence[j]
                    couples.append((token_i, token_j))
        return couples

    def _generate_training_data(self) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        sequences = self._get_sequences_for_skipgrams()
        targets, contexts, labels = [], [], []
        noise_probas = self._get_noise_distribution_probas(scaling_factor=self.scaling_factor)
        for sequence in tqdm(sequences):
            skipgrams = self._generate_skipgrams(sequence)
            for target_word, positive_context_word in skipgrams:
                negative_context_words = self._generate_negative_samples(
                    noise_probas=noise_probas, exclude_token_ids=[target_word, positive_context_word]
                )

                targets.append(target_word)
                contexts.append([positive_context_word] + negative_context_words)
                labels.append([1] + self.num_neg_samples * [0])

        return np.array(targets), np.array(contexts), np.array(labels)

    def prepare_data(self) -> None:
        if not os.path.exists(self.cache_dir):
            os.makedirs(self.cache_dir)
            log.info(f"Cache directory {self.cache_dir} created.")

        if os.path.exists(self._path_filtered_data):
            log.info(f"Filtered data already exists in {self._path_filtered_data} file")
            return

        # Read and preprocess data
        df_data = read_okra_data_from_db(date_from=self.date_from, date_to=self.date_to)
        df_data["text"] = df_data["text"].apply(lambda x: preprocess_text_document(x))
        df_data["sentences"] = df_data["text"].apply(lambda x: split_text_to_sentences(x))
        df_data = df_data.pipe(expand_sentences_into_rows, outcol="sentence", idcol="id")

        # Filter data and export it to csv
        df_filtered_data = self._filter_data(df_data)
        df_filtered_data.to_csv(self._path_filtered_data, index=False)

    def setup(self, stage: Optional[str] = None) -> None:
        df_filtered_data = self._get_filtered_data()
        if stage is None or stage == "fit":
            masks, dates = self._get_split_masks(df_filtered_data)
            text_dataset_vectors = self._get_text_dataset_vectors(df_filtered_data[masks[0]])
            log.info(f"Training dataset has {len(text_dataset_vectors)} obs from {dates[0]}-{dates[1]}.")
            self.text_dataset = OKRAWord2VecTextDataset(text_dataset_vectors)

            self.tokenizer = WordTokenizer(max_tokens=self.vocab_size, seq_len=self.seq_len)
            self.tokenizer.adapt(data=get_corpus_tensor(self.text_dataset))
            self.encoded_dataset = OKRAWord2VecEncodedDataset(self._get_encoded_dataset_tensors())
            training_data = self._generate_training_data()

        if stage is None or stage == "validate":
            raise NotImplementedError("Validation set is not applicable in Word2Vec model.")

        if stage is None or stage == "test":
            raise NotImplementedError("Test set is not applicable in Word2Vec model.")

    def word_counts_to_csv(self) -> None:
        data = self._get_truncated_word_counts_as_dict()
        pd.DataFrame(data, index=["count"]).T.reset_index().to_csv(
            self._path_word_counts, index=False, encoding="UTF-8-SIG"
        )

    def text_dataset_to_csv(self) -> None:
        data = text_dataset_to_list_of_dicts(self.text_dataset)
        pd.DataFrame(data).to_csv(self._path_string_dataset, index=False, encoding="UTF-8-SIG")

    def print_summary(self, n: int = 10):
        sentence_lengths = sorted(len(s.numpy()) for s in get_corpus_tensor(self.text_dataset))
        print(f"Number of sentences: {len(self.text_dataset):,.0f}")
        print(f"Number of tokens: {sum(sentence_lengths):,.0f}")
        print(f"Average number of tokens per sentence: {sum(sentence_lengths)/len(self.text_dataset):,.2f}")
        print(f"Vocab size (maxtokens): {self.tokenizer._max_tokens:,.0f}")
        print(f"Vocab size (tokenizer): {self.tokenizer.vocabulary_size():,.0f}")
        print(f"Vocab size (truncated): {len(self._get_truncated_word_counts()):,.0f}")
        print(f"Word index ['PAD']: {self.tokenizer.word2idx['']}")
        print(f"Word index ['UNK']: {self.tokenizer.word2idx['[UNK]']}")
        print(f"Corpus sentence lengths top-{n}: {sentence_lengths[-n:]}")
        print(f"Corpus sentence lengths bot-{n}: {sentence_lengths[:n]}")
        print(f"Word counts top-{n}: {list(self._get_truncated_word_counts_as_dict().items())[:n]}")
        print(f"Word counts bot-{n}: {list(self._get_truncated_word_counts_as_dict().items())[-n:]}")
        print(f"Note: Word count for ['PAD'] has been set to zero for purposes of noise distribution.")
