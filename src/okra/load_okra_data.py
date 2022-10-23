import django

django.setup()

from okra.models import OKRAReviews


def main():
    OKRAReviews.from_json("data/train_reviews.json")


if __name__ == "__main__":
    main()
