import json
import os
from pathlib import Path
from django.db import models

# Create your models here.

PWD = Path(os.getcwd()).parent.parent


class OKRAReviews(models.Model):
    date = models.DateTimeField()
    title = models.CharField(max_length=512)
    text = models.CharField(max_length=10000)
    url = models.CharField(max_length=100)
    stars = models.CharField(max_length=50)
    raw = models.JSONField()

    @classmethod
    def from_json(cls, path: str) -> "OKRAReviews":
        with open(PWD / path, "r") as f:
            reviews = json.load(f)
        for r in reviews:
            OKRAReviews.objects.create(
                date=r["date"][:19],
                title=r["title"],
                text=r["text"],
                url=r["url"],
                stars=r["stars"],
                raw=r,
            )
