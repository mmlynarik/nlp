import math

import keras
import numpy as np
import tensorflow as tf
from keras import layers, models
from keras.layers.preprocessing import text_vectorization as text

tf.random.set_seed(1)

BATCH_SIZE = 1
MAX_SEQ_LEN = 4
VOCAB_SIZE = 8
EMBEDDING_DIM = 5
DENSE_DIM = 3
LSTM_HIDDEN_DIM = 3

TV = text.TextVectorization()

token_ids = np.array([[1, 2, 0, 0]])
data_out = np.arange(12).reshape(BATCH_SIZE, MAX_SEQ_LEN, LSTM_HIDDEN_DIM)

# Flatten layer does not propagate nor consume mask produced by Embedding layer.


class LSTMOkraModel(keras.Model):
    def __init__(self):
        super().__init__()
        self.embed = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True)
        self.dense = layers.Dense(DENSE_DIM)
        self.hidden = layers.LSTM(LSTM_HIDDEN_DIM, return_sequences=True)

    def call(self, inputs):
        x = self.embed(inputs)
        x = self.dense(x)
        x = self.hidden(x)
        return x

    def summary(self):
        x = layers.Input(shape=(MAX_SEQ_LEN,), batch_size=BATCH_SIZE, name="token_ids")
        model = keras.Model(inputs=[x], outputs=self.call(x))
        return model.summary()


model = LSTMOkraModel()
model.compile(loss="mse", optimizer="adam")
preds = model.predict(token_ids)
loss = model.evaluate(token_ids, data_out, verbose=0)

print("Predictions:\n", preds)
print("Loss:", loss)
print("Loss (recalc):", np.sum(np.square(preds[0, :2] - data_out[0, :2])) / LSTM_HIDDEN_DIM / MAX_SEQ_LEN)

model.summary()
