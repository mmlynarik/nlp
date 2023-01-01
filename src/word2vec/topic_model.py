import math

import keras
import numpy as np
import tensorflow as tf
from keras import layers
from keras.preprocessing.sequence import skipgrams
from keras.layers.preprocessing import text_vectorization as text


tf.random.set_seed(1)

BATCH_SIZE = 1
MAX_SEQ_LEN = 4
VOCAB_SIZE = 8
EMBEDDING_DIM = 5
DENSE_DIM = 2
HIDDEN_DIM = 3

token_ids = np.array([[1, 2, 0, 0]])
data_out = np.arange(24).reshape(BATCH_SIZE, MAX_SEQ_LEN, 2 * HIDDEN_DIM)

# Flatten layer does not propagate nor consume mask produced by Embedding layer.


# class LSTMOkraModel(keras.Model):
#     def __init__(
#         self, max_seq_len: int, embedding_dim: int, vocab_size: int, dense_dim: int, hidden_dim: int,
#     ):
#         super().__init__()
#         self.embedding = layers.Embedding(vocab_size, embedding_dim, mask_zero=True)
#         self.lstm = layers.Bidirectional(layers.LSTM(hidden_dim), merge_mode="sum")
#         self.dense = layers.Dense(2 * hidden_dim)
#         self.call(layers.Input(shape=(max_seq_len,)))

#     def call(self, input_tensor):
#         x = self.embedding(input_tensor)
#         x = self.lstm(x)
#         x = self.dense(x)
#         return x

#     @property
#     def output_shape(self):
#         return self.layers[-1].output_shape[1:]


# model = LSTMOkraModel(
#     max_seq_len=MAX_SEQ_LEN,
#     embedding_dim=EMBEDDING_DIM,
#     vocab_size=VOCAB_SIZE,
#     dense_dim=DENSE_DIM,
#     hidden_dim=HIDDEN_DIM,
# )
# model.compile(loss="mse", optimizer="adam")
# preds = model.predict(token_ids)
# loss = model.evaluate(token_ids, data_out, verbose=0)

# print("Predictions:\n", preds)
# print("Loss:", loss)
# # print("Loss (recalc):", np.sum(np.square(preds[0, :2] - data_out[0, :2])) / math.prod(model.output_shape))

# model.summary()

# tokenizer = WordTokenizer(max_tokens=10000, seq_len=20, output_mode="int")
# corpus = ["Hello!", "I have a dream. What about you? Are you okay, Mr. Bean?", "Nice job."]
# print(split_text_to_sentences_regex(corpus[1]))
# tokenizer.adapt(corpus)
# print(tokenizer.encode(corpus[1]))
# print(tokenizer.get_vocabulary())
