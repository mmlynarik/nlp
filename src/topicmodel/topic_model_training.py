import argparse
from datetime import date, datetime

from dateutil.relativedelta import relativedelta

from topicmodel.datamodule.datamodule import OKRAWord2VecDataModule
from topicmodel.config import DEFAULT_LOG_DIR, DEFAULT_CACHE_DIR, DEFAULT_MODEL_DIR
from topicmodel.datamodule.utils import tf_decode


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


def train_okra_word2vec_model(
    date_from: date,
    date_to: date,
    period_val: relativedelta,
    period_test: relativedelta,
    cache_dir: str,
    log_dir: str,
    model_dir: str,
) -> None:
    """Entrypoint function to train OKRA Word2Vec model."""
    datamodule = OKRAWord2VecDataModule(
        date_from=date_from,
        date_to=date_to,
        period_val=period_val,
        period_test=period_test,
        cache_dir=cache_dir,
    )

    datamodule.prepare_data()
    datamodule.setup("fit")

    print(f"Vocab (20 words): {datamodule.tokenizer.idx2word[:20]}")
    print(f"Vocab size: {datamodule.tokenizer.vocabulary_size()}")
    print(f"Corpus: {datamodule.sentence_corpus[:10]}")
    print(f"Encoded corpus: {datamodule.encoded_corpus[:10]}")
    print(f"Word counts: {list(datamodule.word_counts.items())[:10]}")


def main():
    args = parse_args()

    train_okra_word2vec_model(
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
