import warnings
from datetime import date

import numpy as np
import pandas as pd
import psycopg2
import regex

from word2vec.config import APP_DB
from word2vec.datamodule.queries import QUERY_REVIEWS_DATA_PG
from word2vec.datamodule.regexp import mask_non_words, rectify_typos


def read_reviews_data_from_db(date_from: date, date_to: date) -> pd.DataFrame:
    with warnings.catch_warnings():  # ignore pandas issue #45660
        warnings.simplefilter("ignore", UserWarning)
        with psycopg2.connect(**APP_DB) as conn:
            df_data = pd.read_sql(QUERY_REVIEWS_DATA_PG.format(date_from=date_from, date_to=date_to), conn)
    return df_data


def preprocess_text_document(string: str) -> str:
    return mask_non_words(rectify_typos(string))


def split_text_to_sentences(string: str) -> list[str]:
    """Split the text on correct end-of-sentences delimiters."""
    return regex.split(r"(?<=[^A-Z].[.!?]) +(?=[A-Z])", string)


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


def generate_multinomial_sample(probas: np.ndarray) -> int:
    sample = np.random.multinomial(1, probas)
    return np.where(sample == 1)[0][0]
