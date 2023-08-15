import tensorflow as tf
from keras.preprocessing.sequence import skipgrams


class TopicModelDataset:
    """Document-level dataset class."""

    def __new__(cls, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        return tf.data.Dataset.from_tensor_slices(data)


def get_keys_tensor(dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.constant([item for item in dataset.map(lambda key, x, y: key).as_numpy_iterator()])


def get_corpus_tensor(dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.constant([item for item in dataset.map(lambda key, x, y: x).as_numpy_iterator()])


def get_outputs_tensor(dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.constant([item for item in dataset.map(lambda key, x, y: y).as_numpy_iterator()])


def get_corpus_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.map(lambda key, x, y: x)


def dataset_to_list_of_dicts(dataset: tf.data.Dataset) -> list[dict]:
    return [
        {"key": key, "doc": x.decode("utf-8"), "output": y} for key, x, y in dataset.as_numpy_iterator()
    ]
