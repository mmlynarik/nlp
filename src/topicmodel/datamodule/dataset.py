import tensorflow as tf
from keras.preprocessing.sequence import skipgrams
from keras.layers.preprocessing import text_vectorization as text


class OKRAWord2VecTextSentenceDataset:
    """Sentence-level text dataset class."""

    def __new__(cls, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        return tf.data.Dataset.from_tensor_slices(data)


class OKRAWord2VecEncodedSentenceDataset:
    """Sentence-level integer-encoded dataset class."""

    def __new__(cls, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        return tf.data.Dataset.from_tensor_slices(data)


def get_keys_tensor(dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.constant([item for item in dataset.map(lambda key, x, y: key).as_numpy_iterator()])


def get_outputs_tensor(dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.constant([item for item in dataset.map(lambda key, x, y: y).as_numpy_iterator()])


def get_corpus_tensor(dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.constant([item for item in dataset.map(lambda key, x, y: x).as_numpy_iterator()])


def get_corpus_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.map(lambda key, x, y: x)


def text_dataset_to_list_of_dicts(dataset: tf.data.Dataset) -> list[dict]:
    return [
        {"key": key, "review": x.decode("utf-8"), "output": y} for key, x, y in dataset.as_numpy_iterator()
    ]
