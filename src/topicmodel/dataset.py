import math
from typing import List, Tuple

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.preprocessing.sequence import skipgrams
from keras.layers.preprocessing import text_vectorization as text

from topicmodel.utils import text_to_sentences


tf.random.set_seed(1)

BATCH_SIZE = 1
MAX_SEQ_LEN = 4
VOCAB_SIZE = 8
EMBEDDING_DIM = 5
DENSE_DIM = 2
HIDDEN_DIM = 3

TV = text.TextVectorization()

token_ids = np.array([[1, 2, 0, 0]])
data_out = np.arange(24).reshape(BATCH_SIZE, MAX_SEQ_LEN, 2 * HIDDEN_DIM)

# Flatten layer does not propagate nor consume mask produced by Embedding layer.


class OKRADataset(tf.data.Dataset):
    def __new__(cls, data: List[Tuple[tf.Tensor, tf.Tensor]]):
        return tf.data.Dataset.from_tensor_slices(data)


class OKRADataLoader(tf.data.Dataset):
    def __new__(cls, dataset: tf.data.Dataset, batch_size: int):
        batched_dataset = dataset.batch(batch_size)
        # TODO: Add .map() transformations here e.g. to tokenize the input strings
        return batched_dataset


class WordTokenizer(text.TextVectorization):
    """Word-level tokenizer splitting strings on whitespace after lowercasing and stripping punctuation."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        standardize: str = "lower_and_strip_punctuation",
        split: str = "whitespace",
    ):
        super().__init__(
            max_tokens=vocab_size,
            output_sequence_length=max_seq_len,
            standardize=standardize,
            split=split,
        )

    def encode(self, inputs: str) -> tf.Tensor:
        """Get encoded input strings, each represented as a sequence of integers using learned vocabulary."""
        return self.call(inputs)


class LSTMOkraModel(keras.Model):
    def __init__(
        self,
        max_seq_len: int,
        embedding_dim: int,
        vocab_size: int,
        dense_dim: int,
        hidden_dim: int,
    ):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
        self.lstm = layers.Bidirectional(layers.LSTM(hidden_dim), merge_mode="sum")
        self.dense = layers.Dense(2 * hidden_dim)
        self.call(layers.Input(shape=(max_seq_len,)))

    def call(self, input_tensor):
        x = self.embedding(input_tensor)
        x = self.lstm(x)
        x = self.dense(x)
        return x

    @property
    def output_shape(self):
        return self.layers[-1].output_shape[1:]


model = LSTMOkraModel(
    max_seq_len=MAX_SEQ_LEN,
    embedding_dim=EMBEDDING_DIM,
    vocab_size=VOCAB_SIZE,
    dense_dim=DENSE_DIM,
    hidden_dim=HIDDEN_DIM,
)
model.compile(loss="mse", optimizer="adam")
preds = model.predict(token_ids)
loss = model.evaluate(token_ids, data_out, verbose=0)

print("Predictions:\n", preds)
print("Loss:", loss)
# print("Loss (recalc):", np.sum(np.square(preds[0, :2] - data_out[0, :2])) / math.prod(model.output_shape))

model.summary()

tokenizer = WordTokenizer(vocab_size=10000, max_seq_len=10)
corpus = ["Hello!", "I have a dream. What about you? Are you okay, Mr. Bean?", "Nice job."]
print(text_to_sentences(corpus[1]))
tokenizer.adapt(corpus)
print(tokenizer.encode(corpus[1]))
