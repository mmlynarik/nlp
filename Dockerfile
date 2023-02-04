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

ENV PATH="/root/.local/bin:$PATH"
ENV PYTHONUNBUFFERED=true

COPY ./pyproject.toml ./poetry.lock ./

RUN curl -sSL https://install.python-poetry.org | python3.9 -

RUN python3.9 -m venv .venv && \
    mkdir src/ && touch src/setup.py && \
    poetry config virtualenvs.in-project true && \
    poetry run pip install -U pip setuptools wheel && \
    poetry install --without dev

COPY ./src src/

ENTRYPOINT ["/bin/bash", "-c", "poetry shell && python manage.py migrate && gunicorn -w 4 -b 0.0.0.0:8080 nlp.wsgi:application"]
