import argparse
import os
import pickle
from datetime import date, datetime

from top2vec import Top2Vec

from dateutil.relativedelta import relativedelta

from topicmodel.datamodule.datamodule import TopicModelDataModule
from word2vec.config import DEFAULT_CACHE_DIR


dataset_config = {
    "date_from": date(2011, 1, 1),
    "date_to": date(2019, 12, 31),
    "period_val": relativedelta(months=0),
    "period_test": relativedelta(months=0),
    "cache_dir": DEFAULT_CACHE_DIR,
}

datamodule = TopicModelDataModule(**dataset_config)
datamodule.prepare_data()
datamodule.setup("fit")
top2vec_docs = datamodule.get_top2vec_input()

sbert_models = [
    "distiluse-base-multilingual-cased",
    "all-MiniLM-L6-v2",
    "paraphrase-multilingual-MiniLM-L12-v2",
]

# UMAP needs random state pinning, HDBSCAN and pretrained embedding models (SBERT/USE) are deterministic.
top2vec = Top2Vec(
    documents=top2vec_docs,
    min_count=5,
    # embedding_model=sbert_models[2],
    umap_args={"random_state": 100},
)
topic_words, scores, nums = top2vec.get_topics()
print(topic_words, scores, nums)
