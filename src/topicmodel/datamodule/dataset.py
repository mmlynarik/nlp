import numpy as np
import tensorflow as tf
from keras.preprocessing.sequence import skipgrams
from keras.layers.preprocessing import text_vectorization as text


class OKRAWord2VecSentenceDataset(tf.data.Dataset):
    """Sentence-level dataset class used for traning Word2Vec model."""

    def __new__(cls, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        return tf.data.Dataset.from_tensor_slices(data)


class OKRAWord2VecDataLoader(tf.data.Dataset):
    def __new__(cls, dataset: tf.data.Dataset, batch_size: int):
        batched_dataset = dataset.batch(batch_size)
        # TODO: Add .map() transformations here e.g. to tokenize the input strings
        return batched_dataset
