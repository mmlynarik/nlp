import os


OKRA_DB = {
    "host": os.environ.get("OKRA_DB_HOST"),
    "port": os.environ.get("OKRA_DB_PORT"),
    "dbname": os.environ.get("OKRA_DB_NAME"),
    "user": os.environ.get("OKRA_DB_USER"),
    "password": os.environ.get("OKRA_DB_PASSWORD"),
}

DEFAULT_CACHE_DIR = os.path.join("./data")
DEFAULT_LOG_DIR = os.path.join("./logs")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")

MAX_VOCAB_SIZE = 32768
MAX_SEQ_LEN = 1024
