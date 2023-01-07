import tensorflow as tf
import numpy as np

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


def get_word2vec_embeddings() -> np.ndarray:
    model = get_latest_word2vec_model()
    return model.get_layer("embeddings").get_weights()[0]


def normalize_embeddings(embeddings: np.ndarray):
    norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
    norms = np.reshape(norms, (len(norms), 1))
    return embeddings / norms


def get_normalized_embeddings():
    return normalize_embeddings(get_word2vec_embeddings())


# def get_topn_similar(word: str, topN: int = 10):
#     word_id = vocab[word]
#     if word_id == 0:
#         print("Out of vocabulary word")
#         return

#     norm_embeddings = get_normalized_embeddings()

#     word_vec = norm_embeddings[word_id]
#     word_vec = np.reshape(word_vec, (len(word_vec), 1))
#     dists = np.matmul(norm_embeddings, word_vec).flatten()
#     topN_ids = np.argsort(-dists)[1 : topN + 1]

#     topN_dict = {}
#     for sim_word_id in topN_ids:
#         sim_word = vocab.lookup_token(sim_word_id)
#         topN_dict[sim_word] = dists[sim_word_id]
#     return topN_dict


if __name__ == "__main__":
    print(np.linalg.norm(get_normalized_embeddings()[0]))
