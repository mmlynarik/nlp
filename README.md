# NLP projects repository

This repository contains python code for my personal projects related to NLP.
- Word2Vec word embeddings model using skip-gram negative sampling
- Topic modelling (coming soon)

## Set up repository
- Install poetry package and dependency manager:
```bash
make poetry
```

- Create virtual environment and install dependencies:
```bash
make venv
```

## Word2Vec model
- Setting up PostgreSQL database and load training data into database table (docker is required):
```bash
make db
```

- Initialize Word2Vec model training:
```bash
make train
```
- Test trained embeddings on word similarity task:
```bash
make wordsim
```
