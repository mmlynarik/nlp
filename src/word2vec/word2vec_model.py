import keras
import tensorflow as tf
from keras import layers, optimizers


class Word2VecModel(keras.Model):
    def __init__(self, vocab_size: int, embedding_dim: int, num_neg_samples: int):
        super().__init__()
        self.target_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=1, name="embeddings")
        self.context_embedding = layers.Embedding(vocab_size, embedding_dim, input_length=num_neg_samples + 1)
        self.optimizer = get_adam_optimizer()
        self.loss = tf.nn.sigmoid_cross_entropy_with_logits

    def call(self, pair):
        target, context = pair
        target_embedding = self.target_embedding(target)
        context_embeddings = self.context_embedding(context)
        dots = tf.einsum("be,bce->bc", target_embedding, context_embeddings)
        return dots

    @property
    def output_shape(self):
        return self.layers[-1].output_shape[1:]


def get_adam_optimizer():
    """
    The use of tf.Variable instead of floats in optimizer constructor is needed in order to fix the model weights loading issue where the optimizer parameters would not be loaded and warning would be issued.
    """
    adam = optimizers.Adam(
        learning_rate=tf.Variable(0.001),
        beta_1=tf.Variable(0.9),
        beta_2=tf.Variable(0.999),
        epsilon=tf.Variable(1e-7),
    )
    adam.decay = tf.Variable(0.0)
    adam.iterations
    return adam
