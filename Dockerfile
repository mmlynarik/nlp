FROM python:3.9.5-slim

WORKDIR /app

RUN apt-get update --fix-missing && \
    apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    libgmp-dev \
    libssl-dev \
    python3-dev \
    libpq-dev && \
    apt-get clean

ENV PYTHONUNBUFFERED=true
ENV SKLEARN_ALLOW_DEPRECATED_SKLEARN_PACKAGE_INSTALL=True

COPY pyproject.toml poetry.lock ./

# In case proxy is needed, set ENV HTTPS_PROXY= and HTTP_PROXY= and add --proxy ${HTTP_PROXY} to pip cmds
RUN pip install -U pip setuptools wheel
RUN pip install poetry

RUN python3.9 -m venv .venv && \
    poetry config virtualenvs.in-project true && \
    poetry install --without dev --no-root

COPY . .

RUN poetry install --without dev
