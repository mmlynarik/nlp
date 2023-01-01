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


class Word2Vec(keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int, num_neg_samples: int):
        super().__init__()
        self.target_embedding = layers.Embedding(
            vocab_size, embedding_dim, input_length=1, name="word_embeddings"
        )
        self.context_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=num_neg_samples + 1)
        # self.call(layers.Input(shape=(,)))

    def call(self, input_tensor_pair):
        target, context = input_tensor_pair
        target_embedding = self.target_embedding(target)
        context_embedding = self.context_embedding(context)
        dots = tf.einsum("be,bce->bc", target_embedding, context_embedding)
        return dots

    @property
    def output_shape(self):
        return self.layers[-1].output_shape[1:]
