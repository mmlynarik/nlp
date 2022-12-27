import regex
import warnings
from datetime import date

import pandas as pd
import psycopg2

from topicmodel.config import OKRA_DB
from topicmodel.datamodule.queries import QUERY_OKRA_DATA_PG


RE_SPLITTER_SENTENCE = regex.compile(r"(?<=[^A-Z].[.!?]) +(?=[A-Z])")

RE_TYPO_BASE_DELIMITERS_AFTER_SPACE = regex.compile(r"(?<=\p{L}\p{L}) ([.,?!]+)")
RE_TYPO_BASE_DELIMITERS_SQUEEZED = regex.compile(r"(?<=\p{L}\p{L})([.,?!]+)(?=\p{L}\p{L})")
RE_TYPO_BASE_DELIMITERS_CENTERED = regex.compile(r"(?<=\p{L}\p{L}) ([.,?!]+) (?=\p{L}\p{L})")
RE_TYPO_MULTIPLE_SPACE = regex.compile(r" +")
RE_TYPO_LOWERCASE_START_OF_SENTENCE = regex.compile(r"(?<=\p{L}\p{L}[.?!] )(\p{Ll})")
RE_MULTIPLE_EMOTION_DELIMITERS = regex.compile(r"([!?]+)")

RE_FIX_BASE_DELIMITERS = r"\1 "
RE_FIX_UPPERCASE_START_OF_SENTENCE = lambda x: x.group(1).upper()
RE_FIX_SINGLE_SPACE = r" "
RE_FIX_SINGLE_EMOTION_DELIMITER = lambda x: x.group(1)[0]

RE_EN_TIME = r"(?<=\b)(\d?\d[.:]\d\d ?([ap]m))|(\d?\d[.:]\d\d)|(\d?\d ?([ap]m))(?=\b)"
RE_EN_CCY = r"\p{Sc}\d+(,\d+)*(\.\d\d)?"
RE_EN_PERIOD = r"\d+\.?\d? ?(days?|months?|years?|hours?|minutes?|weeks?|mins?|hrs?)"


def read_okra_data_from_db(date_from: date, date_to: date) -> pd.DataFrame:
    with warnings.catch_warnings():  # ignore pandas issue #45660
        warnings.simplefilter("ignore", UserWarning)
        with psycopg2.connect(**OKRA_DB) as conn:
            df_okra = pd.read_sql(QUERY_OKRA_DATA_PG.format(date_from=date_from, date_to=date_to), conn)
    return df_okra


def join_regexes(patterns: list[regex.Pattern]):
    return "".join(x.pattern for x in patterns)


def rectify_typos(string: str) -> str:
    """Fix typos in text regarding delimiters and lowercase start of sentence."""
    replacements = [
        (RE_TYPO_BASE_DELIMITERS_AFTER_SPACE, RE_FIX_BASE_DELIMITERS),
        (RE_TYPO_BASE_DELIMITERS_SQUEEZED, RE_FIX_BASE_DELIMITERS),
        (RE_TYPO_BASE_DELIMITERS_CENTERED, RE_FIX_BASE_DELIMITERS),
        (RE_TYPO_MULTIPLE_SPACE, RE_FIX_SINGLE_SPACE),
        (RE_TYPO_LOWERCASE_START_OF_SENTENCE, RE_FIX_UPPERCASE_START_OF_SENTENCE),
        (RE_MULTIPLE_EMOTION_DELIMITERS, RE_FIX_SINGLE_EMOTION_DELIMITER),
    ]
    for typo, fix in replacements:
        string = regex.sub(typo, fix, string)
    return string


def mask_ccy(string: str, token="[CCY]") -> str:
    """Replace currency expressions with a special token to reduce vocab size."""
    return regex.sub(RE_EN_CCY, token, string)


def mask_time(string: str, token="[TIME]") -> str:
    """Replace time expressions in text with a special token to reduce vocab size."""
    return regex.sub(RE_EN_TIME, token, string)


def mask_period(string: str, token="[PERIOD]") -> str:
    return regex.sub(RE_EN_PERIOD, token, string)


def mask_symbols(string: str) -> str:
    return mask_period(mask_time(mask_ccy(string)))


def preprocess(string: str) -> str:
    return mask_symbols(rectify_typos(string))


def split_text_to_sentences(string: str) -> list[str]:
    """Split the text on correct end-of-sentences delimiters."""
    return regex.split(RE_SPLITTER_SENTENCE, string)


def expand_sentences_into_rows(df_data: pd.DataFrame, idcol: str, outcol: str) -> pd.DataFrame:
    output_data = []
    for _, row in df_data.iterrows():
        for idx, sentence in enumerate(row["sentences"]):
            output_data.append(
                {
                    **row.to_dict(),
                    idcol: row[idcol] + (idx + 1) / 100,
                    outcol: sentence,
                    "length": len(sentence),
                }
            )
    return pd.DataFrame(output_data)
