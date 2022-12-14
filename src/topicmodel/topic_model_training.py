import argparse
from datetime import date, datetime

from dateutil.relativedelta import relativedelta

from topicmodel.datamodule.datamodule import OKRAWord2VecDataModule
from topicmodel.config import (
    DEFAULT_LOG_DIR,
    DEFAULT_CACHE_DIR,
    DEFAULT_MODEL_DIR,
    SEQ_LEN,
    VOCAB_SIZE,
    EMBEDDING_DIM,
)
from topicmodel.datamodule.dataset import get_corpus_tensor


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

    vocab_size = VOCAB_SIZE
    embedding_dim = EMBEDDING_DIM
    seq_len = SEQ_LEN

    datamodule = OKRAWord2VecDataModule(
        date_from=date_from,
        date_to=date_to,
        period_val=period_val,
        period_test=period_test,
        vocab_size=vocab_size,
        embedding_dim=embedding_dim,
        seq_len=seq_len,
        cache_dir=cache_dir,
    )

    datamodule.prepare_data()
    datamodule.setup("fit")

    print(f"Vocab (20 words): {datamodule.tokenizer.idx2word[:20]}")
    print(f"Vocab size: {len(datamodule.tokenizer.get_vocabulary())}")
    print(f"Word counts top-10: {list(datamodule.word_counts.items())[:10]}")
    print(f"Word counts bottom-10: {list(datamodule.word_counts.items())[-10:]}")
    words = ["the", "to", "", "[UNK]"]
    for word in words:
        print(f"Index of {word}: {datamodule.tokenizer.word2idx[word]}")
    print(f"Word count [UNK]: {datamodule.word_counts.get('[UNK]')}")
    print(f"Word count len: {len(datamodule.word_counts)}")
    print(f"{sorted(len(s.numpy()) for s in get_corpus_tensor(datamodule.string_dataset))}")

    datamodule.word_counts_to_csv()
    datamodule.string_dataset_to_csv()


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


# print(f"Corpus: {datamodule.sentence_corpus[:10]}")
# print(f"Encoded corpus: {datamodule.encoded_corpus[:10]}")

# old_text = "Hello, how are you!I'm fine, thanks!And you?"
# text = """On 10th May I received a refund from Virgin for a trip to Preston from Edinburgh on 22nd April which had arrived over 2 hours late.Unfortunately there were 5 of us on the trip but the refund was only for 1 ticket.I immediately went on live chat as requested but they said they couldn't deal with that and I should email in.I also tried phoning but it just kept ringing.Since then I have emailed twice but only get automated responses and then nothing.The letter I received is very pleased with itself for dealing with my complaint so promptly .If only it was dealt with as I am due to refund money to the others and can't get it finished with.If there was a decent alternative I would never use this lot again as trains I get are invariably late."""

# print(split_text_to_sentences_regex(old_text), "\n")

# text = "Aaaaa!Ahoj , ako sa mas? mam sa ok!ahoj."
# text = "This is the worst customer service based company ever only interested in your money not your needs,not sour grapes just sick that I had no option but to pay extra \u20ac480 for a train when they ( eurostar) said I had plenty of time to catch a connecting tgv they booked.  even when one of the party was 73 not a marathon runner 4 other adults cases and being put on last carriage of train and along with the hundreds of other people leaving the train trying to go places don't believe them when they say you have 50 min till your other connection plenty time they know how busy it is they book it for you remember but when you miss the tgv it's your fault they were busy worst company ever do yourself a favour fly, first and last they don't care really don't care personally think they and the tgv are in it together to well rehearsed it's obviously not there first time they have shafted someone seriously stay clear fly or drive I do it all the time fancied a change mmm guess what I'm in charge now of my holiday not some company out to ruin it"
# print(split_text_to_sentences_regex(text))
