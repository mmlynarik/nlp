import os


OKRA_DB = {
    "host": os.environ.get("OKRA_DB_HOST"),
    "port": os.environ.get("OKRA_DB_PORT"),
    "dbname": os.environ.get("OKRA_DB_NAME"),
    "user": os.environ.get("OKRA_DB_USER"),
    "password": os.environ.get("OKRA_DB_PASSWORD"),
}

VOCAB_SIZE = 30000
EMBEDDING_DIM = 300
SEQ_LEN = 1024
MIN_COUNT = 5
NUM_NEG_SAMPLES = 10
SCALING_FACTOR = 0.75
CONTEXT_WINDOW_SIZE = 5

DEFAULT_CACHE_DIR = os.path.join("./data")
DEFAULT_LOG_DIR = os.path.join("./logs")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")