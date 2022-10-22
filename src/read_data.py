from datetime import date
import json
import pandas as pd

f = open("../data/train_reviews.json")
reviews = json.load(f)

df = pd.DataFrame(reviews)
