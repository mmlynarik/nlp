import re
import string

import tensorflow as tf
from keras.layers.preprocessing import text_vectorization as text


def custom_standardization(input_data):
    lowercase = tf.strings.lower(input_data)
    stripped_html = tf.strings.regex_replace(lowercase, "<br />", " ")
    return tf.strings.regex_replace(stripped_html, "[%s]" % re.escape(string.punctuation), "")


class WordTokenizer(text.TextVectorization):
    """
    Word-level tokenizer splitting strings on whitespace after lowercasing and stripping punctuation
    and padding the output token sequence of integers with <PAD> token to predefined output sequence length.
    """

    def __init__(
        self,
        max_tokens: int,
        seq_len: int,
        output_mode: int = "int",
        standardize: str = "lower_and_strip_punctuation",
        split: str = "whitespace",
    ):
        super().__init__(
            max_tokens=max_tokens,
            output_sequence_length=seq_len,
            standardize=standardize,
            split=split,
            output_mode=output_mode,
        )

    def encode(self, input_tensor: tf.Tensor) -> tf.Tensor:
        """Encode input strings tensor, into a sequence of integers using learned vocabulary."""
        return self.call(input_tensor)

    @property
    def idx2word(self) -> list[str]:
        return self.get_vocabulary()

    @property
    def word2idx(self) -> dict[str, int]:
        return {word: idx for idx, word in enumerate(self.idx2word)}

    @property
    def vocab_size(self):
        return self.vocabulary_size()
