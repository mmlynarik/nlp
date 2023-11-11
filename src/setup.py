"""A setuptools based setup module.

See:
https://packaging.python.org/guides/distributing-packages-using-setuptools/
https://github.com/pypa/sampleproject
"""

from pathlib import Path
from setuptools import setup

ROOT_DIR = Path(__file__).resolve().parent.parent

with open(ROOT_DIR / "README.md", encoding="utf-8") as f:
    long_description = f.read()


setup(
    name="src",
    version="0.0.1",
    package_data={},
    packages=["word2vec", "topicmodel", "djangoproject", "summarization"],
    description="My NLP home projects",
    long_description=long_description,
    long_description_content_type="text/markdown",
    install_requires=[],
    zip_safe=False,
    entry_points={
        "console_scripts": [
            "load_train_reviews_data=word2vec.load_train_reviews_from_json_to_db:main",
            "train_word2vec_model=word2vec.word2vec_model_training:main",
            "test_word_similarity=word2vec.word_similarity:main",
            "train_tokenizer=summarization.entrypoints.train_tokenizer:main",
            "train_model=summarization.entrypoints.train_model:main",
        ]
    },
)
