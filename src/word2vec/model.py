import keras
import tensorflow as tf
from keras import layers


class Word2Vec(keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int, num_neg_samples: int):
        super().__init__()
        self.target_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=1, name="embeddings")
        self.context_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=num_neg_samples + 1)

    def call(self, pair):
        target, context = pair
        target_embedding = self.target_embedding(target)
        context_embeddings = self.context_embedding(context)
        dots = tf.einsum("be,bce->bc", target_embedding, context_embeddings)
        return dots

    @property
    def output_shape(self):
        return self.layers[-1].output_shape[1:]


def custom_loss(x_logit, y_true):
    return tf.nn.sigmoid_cross_entropy_with_logits(logits=x_logit, labels=y_true)
