import tensorflow as tf

from word2vec.config import (
    DEFAULT_MODEL_DIR,
    EMBEDDING_DIM,
    NUM_NEG_SAMPLES,
    MAX_VOCAB_SIZE,
)
from word2vec.model import Word2VecModel


def configure_embedding_weights_shape(model: Word2VecModel) -> Word2VecModel:
    model((tf.Variable([1, 1]), tf.Variable([[1, 1], [1, 1]]),))
    return model


def get_latest_word2vec_model():
    latest_checkpoint = tf.train.latest_checkpoint(DEFAULT_MODEL_DIR)
    model = Word2VecModel(MAX_VOCAB_SIZE, EMBEDDING_DIM, NUM_NEG_SAMPLES)
    model = configure_embedding_weights_shape(model)
    model.load_weights(latest_checkpoint)
    return model


def get_word2vec_embeddings():
    model = get_latest_word2vec_model()
    return model.get_layer("embeddings").get_weights()


if __name__ == "__main__":
    print(get_word2vec_embeddings())
