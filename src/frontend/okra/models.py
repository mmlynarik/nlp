import json

from django.db import models


class OKRAReviews(models.Model):
    date = models.DateTimeField()
    title = models.CharField(max_length=512)
    text = models.CharField(max_length=10000)
    url = models.CharField(max_length=100)
    stars = models.CharField(max_length=50)
    raw = models.JSONField()

    @classmethod
    def from_json(cls, path: str) -> "OKRAReviews":
        OKRAReviews.objects.all().delete()
        with open(path, "r") as f:
            reviews = json.load(f)
        for review in reviews:
            OKRAReviews.objects.create(
                date=review["date"][:19],
                title=review["title"],
                text=review["text"],
                url=review["url"],
                stars=review["stars"],
                raw=review,
            )
