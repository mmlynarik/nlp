import math

import tensorflow as tf
import numpy as np
from keras import layers, models
from keras.layers.preprocessing.text_vectorization import TextVectorization

tf.random.set_seed(1)

MAX_SEQ_LEN = 4
BATCH_SIZE = 1
HIDDEN_DIM = 3
EMBEDDING_DIM = 5
VOCAB_SIZE = 8

token_ids = np.array([[1, 2, 0, 0]])
data_out = np.arange(12).reshape(BATCH_SIZE, MAX_SEQ_LEN, HIDDEN_DIM)

input = layers.Input(shape=(MAX_SEQ_LEN,), batch_size=BATCH_SIZE, name="token_ids")
embed = layers.Embedding(VOCAB_SIZE, EMBEDDING_DIM, mask_zero=True, input_length=4)(input)
dense = layers.Dense(3)(embed)
output = layers.LSTM(3, return_sequences=True)(dense)

model = models.Model(inputs=input, outputs=output)
model.compile(loss="mse", optimizer="adam")
preds = model.predict(token_ids)
loss = model.evaluate(token_ids, data_out, verbose=0)

print("Predictions:\n", preds)
print("Computed Loss:", loss)
print("Recomputed Loss", np.sum(np.square(preds[0, :2] - data_out[0, :2])) / math.prod(output.shape[1:]))

model.summary()
