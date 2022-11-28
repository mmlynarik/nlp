from datetime import date

from topicmodel.datamodule import OKRADataModule


datamodule = OKRADataModule(date_from=date(2011, 1, 1), date_to=date(2019, 1, 1))
datamodule.prepare_data()
