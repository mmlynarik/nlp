import argparse
from datetime import date, datetime

import tensorflow as tf
from keras.callbacks import TensorBoard
from keras.losses import CategoricalCrossentropy
from dateutil.relativedelta import relativedelta

from word2vec.datamodule.datamodule import Word2VecDataModule
from word2vec.model import Word2Vec, custom_loss
from word2vec.config import (
    DEFAULT_LOG_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    SEQ_LEN,
    VOCAB_SIZE,
    EMBEDDING_DIM,
    MIN_COUNT,
    NUM_NEG_SAMPLES,
    SCALING_FACTOR,
    CONTEXT_WINDOW_SIZE,
    BATCH_SIZE,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OKRA Word2Vec model")

    parser.add_argument(
        "-s",
        "--startdate",
        help="The start date - format YYYY-MM-DD",
        required=True,
        type=lambda dt: datetime.strptime(dt, "%Y-%m-%d").date(),
    )
    parser.add_argument(
        "-e",
        "--enddate",
        help="The end date format YYYY-MM-DD",
        required=True,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d").date(),
    )
    parser.add_argument(
        "-v", "--valperiod", help="Length of validation dataset in months", type=int,
    )
    parser.add_argument(
        "-t", "--testperiod", help="Length of test dataset in months", type=int,
    )
    parser.add_argument(
        "--cache-dir",
        help="Path to a directory where training, validation and test data are cached",
        type=str,
        default=DEFAULT_CACHE_DIR,
    )
    parser.add_argument(
        "--log-dir",
        help="Path to a directory where training statistics are stored",
        type=str,
        default=DEFAULT_LOG_DIR,
    )
    parser.add_argument(
        "--model-dir",
        help="Path to a directory where trained models are stored",
        type=str,
        default=DEFAULT_MODEL_DIR,
    )
    return parser.parse_args()


def train_word2vec_model(
    date_from: date,
    date_to: date,
    period_val: relativedelta,
    period_test: relativedelta,
    cache_dir: str,
    log_dir: str,
    model_dir: str,
) -> None:
    """Entrypoint function to train Train reviews Word2Vec model."""

    vocab_size = VOCAB_SIZE
    embedding_dim = EMBEDDING_DIM
    seq_len = SEQ_LEN
    min_count = MIN_COUNT
    num_neg_samples = NUM_NEG_SAMPLES
    scaling_factor = SCALING_FACTOR
    context_window_size = CONTEXT_WINDOW_SIZE
    batch_size = BATCH_SIZE

    datamodule = Word2VecDataModule(
        date_from=date_from,
        date_to=date_to,
        period_val=period_val,
        period_test=period_test,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        min_count=min_count,
        num_neg_samples=num_neg_samples,
        scaling_factor=scaling_factor,
        context_window_size=context_window_size,
        batch_size=batch_size,
        seq_len=seq_len,
        cache_dir=cache_dir,
    )

    datamodule.prepare_data()
    datamodule.setup("fit")

    loss = CategoricalCrossentropy(from_logits=True)  # custom_loss
    model = Word2Vec(vocab_size=vocab_size, embedding_dim=embedding_dim, num_neg_samples=num_neg_samples)
    model.compile(loss=loss, optimizer="adam", metrics=["accuracy"])

    tensorboard_callback = TensorBoard(log_dir="logs")
    model.fit(datamodule.train_dataset, epochs=5, callbacks=[tensorboard_callback])


def main():
    args = parse_args()

    train_word2vec_model(
        date_from=args.startdate,
        date_to=args.enddate,
        period_val=relativedelta(months=args.valperiod),
        period_test=relativedelta(months=args.testperiod),
        cache_dir=args.cache_dir,
        log_dir=args.log_dir,
        model_dir=args.model_dir,
    )


if __name__ == "__main__":
    main()
