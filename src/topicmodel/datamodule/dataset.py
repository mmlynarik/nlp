import tensorflow as tf
from keras.preprocessing.sequence import skipgrams
from keras.layers.preprocessing import text_vectorization as text


class OKRAWord2VecDataset(tf.data.Dataset):
    """Sentence-level dataset class used for traning Word2Vec model."""

    def __new__(cls, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        return tf.data.Dataset.from_tensor_slices(data)
