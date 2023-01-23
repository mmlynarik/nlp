.PHONY: django, dramatiq, dramatiqr, test migrate, makemigrations, static, db, loadprod, loadtest, flush, superuser, venv, venvd, ssh, git, poetry, repo

ssh:
	ssh-keygen -t rsa -b 4096 -C "miroslav.mlynarik@gmail.com" -N '' -f ~/.ssh/id_rsa
	cat ~/.ssh/id_rsa.pub

git:
	git config --global user.name "Miroslav Mlynarik"
	git config --global user.email "miroslav.mlynarik@gmail.com"
	git config --global remote.origin.prune true

poetry:
	curl -sSL https://install.python-poetry.org | python3.9 -

venv:
	poetry config virtualenvs.in-project true
	python3.9 -m venv .venv; \
	cp .env_tmpl .env; \
	nano .env; \
	echo "set -a && . ./.env && set +a" >> .venv/bin/activate; \
	. .venv/bin/activate; \
	pip install -U pip setuptools wheel; \
	sudo apt install libpq-dev; \
	poetry install


db:
	docker run -d --name postgres -e POSTGRES_USER=$${OKRA_DB_USER} -e POSTGRES_PASSWORD=$${OKRA_DB_PASSWORD} -p 5432:5432 -v ${HOME}/data:/var/lib/postgresql/data postgres:15
	sleep 2
	cd src/djangoproject/; \
	python manage.py migrate
	load_train_reviews_data

dbd:
	sudo rm -rf ~/data
	docker rm -f postgres

train:
	train_word2vec_model -s 2011-01-01 -e 2019-12-31 -v 0 -t 0

wordsim:
	test_word_similarity

precommit:
	pre-commit run --all-files

posh:
	mkdir ~/.poshthemes/
	wget https://github.com/JanDeDobbeleer/oh-my-posh/releases/latest/download/posh-linux-amd64
	sudo mv posh-linux-amd64 /usr/local/bin/oh-my-posh
	wget https://raw.githubusercontent.com/mmlynarik/python/master/config/paradox.omp.json
	mv paradox.omp.json ~/.poshthemes/paradox.omp.json
	sudo chmod +x /usr/local/bin/oh-my-posh
	echo eval "$$(sudo oh-my-posh --init --shell bash --config ~/.poshthemes/paradox.omp.json)" >> ~/.bashrc

shell:
	cd src/djangoproject; \
	python manage.py shell

venvd:
	rm -rf .venv

app:
	cd src/djangoproject/; \
	python manage.py collectstatic; \
	python manage.py migrate; \
	python manage.py createsuperuser

django:
	cd src/djangoproject/; \
	python manage.py runserver

test:
	python -m unittest discover -s tests -t .

superuser:
	cd src/djangoproject/; \
	python manage.py createsuperuser

migrate:
	cd src/djangoproject/; \
	python manage.py migrate

makemigrations:
	cd src/djangoproject; \
	python manage.py makemigrations

static:
	cd src/djangoproject/; \
	python manage.py collectstatic

loaddata:
	cd djangoproject/; \
	python manage.py migrate; \
	python manage.py flush --no-input; \
	python manage.py loaddata db_backup.json

flush:
	cd djangoproject/; \
	python manage.py flush
