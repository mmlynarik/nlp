import json

import pandas as pd

f = open("data/train_reviews.json")
reviews = json.load(f)
a = 2

df = pd.DataFrame(reviews)


print(df)
