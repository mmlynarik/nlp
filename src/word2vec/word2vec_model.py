import keras
import tensorflow as tf
from keras import layers


class Word2Vec(keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int, num_neg_samples: int):
        super().__init__()
        self.target_embedding = layers.Embedding(
            input_dim=vocab_size, output_dim=embedding_dim, input_length=1, name="word_embeddings"
        )
        self.context_embedding = layers.Embedding(
            input_dim=vocab_size,
            output_dim=embedding_dim,
            input_length=num_neg_samples + 1,
            name="context_embeddings",
        )
        # self.call(layers.Input(shape=(,)))

    def call(self, pair):
        """Calculate dot products of target word embedding with context words embeddings."""
        target, context = pair
        target_embedding = self.target_embedding(target)
        context_embeddings = self.context_embedding(context)
        dots = tf.einsum("be,bce->bc", target_embedding, context_embeddings)
        return dots

    @property
    def output_shape(self):
        return self.layers[-1].output_shape[1:]
