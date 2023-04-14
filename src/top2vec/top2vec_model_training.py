import top2vec

from word2vec.datamodule.datamodule import Word2VecDataModule
from transformers import BertTokenizer
from tokenizers import Tokenizer, Encoding
from tokenizers.models import WordPiece

import argparse
import os
import pickle
from datetime import date, datetime

import tensorflow as tf
from keras.callbacks import TensorBoard, ModelCheckpoint
from dateutil.relativedelta import relativedelta
from word2vec.datamodule.datamodule import Word2VecDataModule
from word2vec.word2vec_model import Word2VecModel
from word2vec.config import (
    DEFAULT_LOG_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    SEQ_LEN,
    MIN_COUNT,
    MAX_VOCAB_SIZE,
    EMBEDDING_DIM,
    NUM_NEG_SAMPLES,
    SCALING_FACTOR,
    CONTEXT_WINDOW_SIZE,
    NUM_EPOCHS,
    BATCH_SIZE,
)


# max_vocab_size = MAX_VOCAB_SIZE
# embedding_dim = EMBEDDING_DIM
# seq_len = SEQ_LEN
# min_count = MIN_COUNT
# num_neg_samples = NUM_NEG_SAMPLES
# scaling_factor = SCALING_FACTOR
# context_window_size = CONTEXT_WINDOW_SIZE
# batch_size = BATCH_SIZE
# num_epochs = NUM_EPOCHS

# dataset_config = {
#     "date_from": date(2011, 1, 1),
#     "date_to": date(2019, 12, 31),
#     "period_val": 0,
#     "period_test": 0,
#     "max_vocab_size": max_vocab_size,
#     "min_count": min_count,
#     "num_neg_samples": num_neg_samples,
#     "scaling_factor": scaling_factor,
#     "context_window_size": context_window_size,
#     "batch_size": batch_size,
#     "seq_len": seq_len,
#     "cache_dir": DEFAULT_CACHE_DIR,
# }

# datamodule = Word2VecDataModule(**dataset_config)
# df = datamodule._get_filtered_data()
# print(df.head(10))

# tokenizer: BertTokenizer = BertTokenizer.from_pretrained("bert-base-uncased", proxies={"https": ""})
# bert_tokenizer = Tokenizer(WordPiece(unk_token="[UNK]"))
# text = "ahoj tu som, a ty?"
# output: Encoding = bert_tokenizer.encode(text)

print("AAA")
