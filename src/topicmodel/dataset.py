import tensorflow as tf
import numpy as np
from keras.layers import Input, Embedding, Dense
from keras.models import Model
from keras.layers.preprocessing.text_vectorization import TextVectorization

tf.random.set_seed(1)

data_in = np.array([[1, 2, 0, 0]])
data_out = np.arange(12).reshape(1, 4, 3)

x = Input(shape=(4,))
e = Embedding(5, 5, mask_zero=True)(x)
d = Dense(3)(e)

m = Model(inputs=x, outputs=d)
m.compile(loss="mse", optimizer="adam")
preds = m.predict(data_in)
loss = m.evaluate(data_in, data_out, verbose=0)

print(preds)
print("Computed Loss:", loss)
print("Recalculated Loss", np.square(preds[0, :2] - data_out[0, :2]).mean())
