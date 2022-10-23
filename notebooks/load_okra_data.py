import django

django.setup()

from okra.models import OKRAReviews


OKRAReviews.from_json("data/train_reviews.json")
