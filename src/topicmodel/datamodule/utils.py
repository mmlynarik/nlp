import re
import warnings
from datetime import date

import pandas as pd
import tensorflow as tf
import psycopg2

from topicmodel.config import OKRA_DB
from topicmodel.datamodule.queries import QUERY_OKRA_DATA_PG


RE_SPLITTER_TEXT_ON_TYPO_UNSPLIT_SENTENCES = re.compile(r"(?<=[^A-Z].[.!?])\b(?=[A-Z])")
RE_SPLITTER_SENTENCE = re.compile(r"(?<=[^A-Z].[.!?]) +(?=[A-Z])")
RE_CONTEXT = re.compile(r".......")


def read_okra_data_from_db(date_from: date, date_to: date) -> pd.DataFrame:
    with warnings.catch_warnings():  # ignore pandas issue #45660
        warnings.simplefilter("ignore", UserWarning)
        with psycopg2.connect(**OKRA_DB) as conn:
            df_okra = pd.read_sql(QUERY_OKRA_DATA_PG.format(date_from=date_from, date_to=date_to), conn)
    return df_okra


def find_typo_unsplit_sentences(string: str) -> list[str]:
    """Find all typos with no whitespace between [.?!] and new sentence."""
    pattern = "".join(x.pattern for x in [RE_CONTEXT, RE_SPLITTER_TEXT_ON_TYPO_UNSPLIT_SENTENCES, RE_CONTEXT])
    return re.findall(pattern, string)


def get_value_counts_of_typo_unsplit_sentences(df: pd.DataFrame, col="text") -> pd.Series:
    return df[col].apply(lambda x: tuple(find_typo_unsplit_sentences(x))).value_counts()


def split_typo_unsplit_sentences(string: str) -> str:
    """Find all cases of no whitespace between [.?!] and new sentence and fixes them by adding a space."""
    return " ".join(re.split(RE_SPLITTER_TEXT_ON_TYPO_UNSPLIT_SENTENCES, string))


def split_text_to_sentences_regex(string: str) -> list[str]:
    """First fix unsplit sentences typos and then split the text on correct end-of-sentences whitespace."""
    string = split_typo_unsplit_sentences(string)
    return re.split(RE_SPLITTER_SENTENCE, string)


def expand_sentences_into_rows(df_data: pd.DataFrame, idcol: str, outcol: str) -> pd.DataFrame:
    output_data = []
    for _, row in df_data.iterrows():
        for idx, sentence in enumerate(row["sentences"]):
            output_data.append({**row.to_dict(), idcol: row[idcol] + idx / 10, outcol: sentence})
    return pd.DataFrame(output_data)


def tf_decode(tensor: tf.Tensor):
    return tensor.numpy().decode("utf-8")


old_text = "Hello, how are you!I'm fine, thanks!And you?"
text = """On 10th May I received a refund from Virgin for a trip to Preston from Edinburgh on 22nd April which had arrived over 2 hours late.Unfortunately there were 5 of us on the trip but the refund was only for 1 ticket.I immediately went on live chat as requested but they said they couldn't deal with that and I should email in.I also tried phoning but it just kept ringing.Since then I have emailed twice but only get automated responses and then nothing.The letter I received is very pleased with itself for dealing with my complaint so promptly .If only it was dealt with as I am due to refund money to the others and can't get it finished with.If there was a decent alternative I would never use this lot again as trains I get are invariably late."""

print(split_text_to_sentences(old_text), "\n")
print(split_text_to_sentences_regex(old_text), "\n")
