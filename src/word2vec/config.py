import os


APP_DB = {
    "host": os.environ.get("POSTGRES_HOST"),
    "port": os.environ.get("POSTGRES_PORT"),
    "dbname": os.environ.get("POSTGRES_NAME"),
    "user": os.environ.get("POSTGRES_USER"),
    "password": os.environ.get("POSTGRES_PASSWORD"),
}

MAX_VOCAB_SIZE = 30000
SEQ_LEN = 1024
MIN_COUNT = 5

NUM_NEG_SAMPLES = 10
SCALING_FACTOR = 0.75
CONTEXT_WINDOW_SIZE = 5

EMBEDDING_DIM = 300
NUM_EPOCHS = 20
BATCH_SIZE = 512

DEFAULT_CACHE_DIR = os.path.join("./data")
DEFAULT_LOG_DIR = os.path.join("./logs")
DEFAULT_MODEL_DIR = os.path.join(os.path.dirname(__file__), "trained_models")
