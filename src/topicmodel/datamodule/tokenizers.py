import tensorflow as tf
from keras.layers.preprocessing import text_vectorization as text


class WordTokenizer(text.TextVectorization):
    """Word-level tokenizer splitting strings on whitespace after lowercasing and stripping punctuation."""

    def __init__(
        self,
        vocab_size: int,
        max_seq_len: int,
        standardize: str = "lower_and_strip_punctuation",
        split: str = "whitespace",
    ):
        super().__init__(
            max_tokens=vocab_size, output_sequence_length=max_seq_len, standardize=standardize, split=split,
        )

    def encode(self, inputs: str) -> tf.Tensor:
        """Get encoded input strings, each represented as a sequence of integers using learned vocabulary."""
        return self.call(inputs)
