import os

OKRA_DB = {
    "host": os.environ.get("OKRA_DB_HOST"),
    "port": os.environ.get("OKRA_DB_PORT"),
    "dbname": os.environ.get("OKRA_DB_NAME"),
    "user": os.environ.get("OKRA_DB_USER"),
    "password": os.environ.get("OKRA_DB_PASSWORD"),
}
