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

top2vec = Top2Vec(datamodule.get_top2vec_input())
