import math

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.layers.preprocessing import text_vectorization as text

tf.random.set_seed(1)

BATCH_SIZE = 1
MAX_SEQ_LEN = 4
VOCAB_SIZE = 8
EMBEDDING_DIM = 5
DENSE_DIM = 2
LSTM_HIDDEN_DIM = 3

TV = text.TextVectorization()

token_ids = np.array([[1, 2, 0, 0]])
data_out = np.arange(12).reshape(BATCH_SIZE, MAX_SEQ_LEN, LSTM_HIDDEN_DIM)

# Flatten layer does not propagate nor consume mask produced by Embedding layer.


class LSTMOkraModel(keras.Model):
    def __init__(
        self,
        batch_size: int,
        max_seq_len: int,
        embedding_dim: int,
        vocab_size: int,
        dense_dim: int,
        lstm_hidden_dim: int,
    ):
        super().__init__()
        self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True, name="embed")
        self.dense = layers.Dense(dense_dim, name="dense")
        self.lstm = layers.LSTM(lstm_hidden_dim, return_sequences=True, name="lstm")
        self.call(layers.Input(shape=(max_seq_len,), batch_size=batch_size))

    def call(self, inputs):
        x = self.embedding(inputs)
        x = self.dense(x)
        x = self.lstm(x)
        return x

    @property
    def output_shape(self):
        return self.layers[-1].output_shape


model = LSTMOkraModel(
    batch_size=BATCH_SIZE,
    max_seq_len=MAX_SEQ_LEN,
    embedding_dim=EMBEDDING_DIM,
    vocab_size=VOCAB_SIZE,
    dense_dim=DENSE_DIM,
    lstm_hidden_dim=LSTM_HIDDEN_DIM,
)
model.compile(loss="mse", optimizer="adam")
preds = model.predict(token_ids)
loss = model.evaluate(token_ids, data_out, verbose=0)

print("Predictions:\n", preds)
print("Loss:", loss)
print("Loss (recalc):", np.sum(np.square(preds[0, :2] - data_out[0, :2])) / math.prod(model.output_shape))

model.summary()
