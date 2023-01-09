# NLP projects repository

This repository contains python code for my personal projects related to NLP. Currently finished project is **Word2Vec
Skip-gram negative sampling embedding model**. In the following weeks, topic modelling project will be added.

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
