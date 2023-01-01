import tensorflow as tf
from keras.preprocessing.sequence import skipgrams


class OKRAWord2VecTextDataset:
    """Sentence-level text dataset class."""

    def __new__(cls, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        return tf.data.Dataset.from_tensor_slices(data)


class OKRAWord2VecEncodedDataset:
    """Sentence-level integer-encoded dataset class."""

    def __new__(cls, data: tuple[tf.Tensor, tf.Tensor, tf.Tensor]):
        return tf.data.Dataset.from_tensor_slices(data)


class OKRAWord2VecSGNSDataset:
    """Skip-gram negative sampling training dataset class."""

    def __new__(cls, data: tuple[tuple[tf.Tensor, tf.Tensor], tf.Tensor]):
        return tf.data.Dataset.from_tensor_slices(data)


def get_keys_tensor(dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.constant([item for item in dataset.map(lambda key, x, y: key).as_numpy_iterator()])


def get_outputs_tensor(dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.constant([item for item in dataset.map(lambda key, x, y: y).as_numpy_iterator()])


def get_corpus_tensor(dataset: tf.data.Dataset) -> tf.Tensor:
    return tf.constant([item for item in dataset.map(lambda key, x, y: x).as_numpy_iterator()])


def get_encoded_sequences(dataset: OKRAWord2VecEncodedDataset) -> list[list[int]]:
    sequences = dataset.map(lambda key, x, y: x).as_numpy_iterator()
    return [[token_id for token_id in seq] for seq in sequences]


def get_decoded_sequences(sequences: list[list[int]], idx2word: list[str]) -> list[str]:
    return [[idx2word[idx] for idx in seq] for seq in sequences]


def get_corpus_dataset(dataset: tf.data.Dataset) -> tf.data.Dataset:
    return dataset.map(lambda key, x, y: x)


def text_dataset_to_list_of_dicts(dataset: tf.data.Dataset) -> list[dict]:
    return [
        {"key": key, "review": x.decode("utf-8"), "output": y} for key, x, y in dataset.as_numpy_iterator()
    ]
