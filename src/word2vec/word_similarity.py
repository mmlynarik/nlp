import os
import pickle

import numpy as np
import tensorflow as tf

from word2vec.config import DEFAULT_MODEL_DIR, EMBEDDING_DIM, MAX_VOCAB_SIZE, NUM_NEG_SAMPLES
from word2vec.model import Word2VecModel


def configure_embedding_weights_shape(model: Word2VecModel) -> Word2VecModel:
    model((tf.Variable([1, 1]), tf.Variable([[1, 1], [1, 1]]),))
    return model


def get_latest_word2vec_model() -> Word2VecModel:
    latest_checkpoint = tf.train.latest_checkpoint(DEFAULT_MODEL_DIR)
    model = Word2VecModel(MAX_VOCAB_SIZE, EMBEDDING_DIM, NUM_NEG_SAMPLES)
    model = configure_embedding_weights_shape(model)
    model.load_weights(latest_checkpoint)
    return model


def get_latest_word2vec_idx2word() -> list[str]:
    latest_checkpoint = tf.train.latest_checkpoint(DEFAULT_MODEL_DIR)
    checkpoint_path = os.path.join(DEFAULT_MODEL_DIR, latest_checkpoint)
    with open(checkpoint_path + ".idx2word.pkl", "rb") as f:
        idx2word = pickle.load(f)
    return idx2word


def get_word2vec_embeddings() -> np.ndarray:
    model = get_latest_word2vec_model()
    return model.get_layer("embeddings").get_weights()[0]


def normalize_embeddings(embeddings: np.ndarray) -> np.ndarray:
    norms = (embeddings ** 2).sum(axis=1) ** (1 / 2)
    norms = np.reshape(norms, (len(norms), 1))
    return embeddings / norms


def get_normalized_embeddings() -> np.ndarray:
    return normalize_embeddings(get_word2vec_embeddings())


def get_topn_similar_words(word: str, n: int = 10):
    idx2word = get_latest_word2vec_idx2word()
    try:
        word_idx = idx2word.index(word)
    except ValueError:
        print("No embedding for out of vocabulary word")
        return

    norm_embeddings = get_normalized_embeddings()

    try:
        embedding = norm_embeddings[word_idx]
    except IndexError:
        print("No embedding for low frequency word.")
        return

    distances = norm_embeddings.dot(embedding)

    topn_ids = np.argsort(distances)[-n:]
    topn_dict = {}
    for word_id in topn_ids:
        word = idx2word[word_id]
        topn_dict[word] = distances[word_id]
    for word, similarity in topn_dict.items():
        print(f"{word}: {similarity:.3f}")


if __name__ == "__main__":
    get_topn_similar_words("has")
