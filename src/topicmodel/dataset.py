import tensorflow as tf
from keras import layers
from keras.preprocessing.sequence import skipgrams
from keras.layers.preprocessing import text_vectorization as text

from topicmodel.utils import text_to_sentences


class OKRAWord2VecDataset(tf.data.Dataset):
    def __new__(cls, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        return tf.data.Dataset.from_tensor_slices(data)
