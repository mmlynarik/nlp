import argparse
from datetime import date, datetime

from dateutil.relativedelta import relativedelta
from topicmodel.datamodule import OKRADataModule
from topicmodel.utils import text_to_sentences

from topicmodel.config import DEFAULT_LOG_DIR, DEFAULT_CACHE_DIR, DEFAULT_MODEL_DIR


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train OKRA Word2Vec model")

    parser.add_argument(
        "-s",
        "--startdate",
        help="The start date - format YYYY-MM-DD",
        required=True,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
    )
    parser.add_argument(
        "-e",
        "--enddate",
        help="The end date format YYYY-MM-DD",
        required=True,
        type=lambda s: datetime.strptime(s, "%Y-%m-%d"),
    )
    parser.add_argument(
        "-v", "--valperiod", help="Length of validation dataset in months", type=int,
    )
    parser.add_argument(
        "-t", "--testperiod", help="Length of test dataset in months", type=int,
    )
    parser.add_argument(
        "--cachedir",
        help="Path to a directory where training, validation and test data are cached",
        type=str,
        default=DEFAULT_CACHE_DIR,
    )
    parser.add_argument(
        "--logdir",
        help="Path to a directory where training statistics are stored",
        type=str,
        default=DEFAULT_LOG_DIR,
    )
    parser.add_argument(
        "--modeldir",
        help="Path to a directory where trained models are stored",
        type=str,
        default=DEFAULT_MODEL_DIR,
    )
    return parser.parse_args()


datamodule = OKRADataModule(
    date_from=date(2011, 1, 1),
    date_to=date(2019, 1, 1),
    period_val=relativedelta(months=1),
    period_test=relativedelta(months=1),
)


def train_okra_word2vec_model():
    datamodule.prepare_data()
    datamodule.setup()

    print(text_to_sentences(list(datamodule.train_data.as_numpy_iterator())[24][1].decode("UTF8")))


def main():
    args = parse_args()

    train_okra_word2vec_model(
        args.chems,
        args.startdate.date(),
        args.enddate.date(),
        relativedelta(months=args.valperiod),
        relativedelta(months=args.testperiod),
        args.cachedir,
        args.logdir,
        args.modeldir,
    )


if __name__ == "__main__":
    main()
