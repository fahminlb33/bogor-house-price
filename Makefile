SHELL := /bin/bash

install:
	pip install -r requirements.txt

format:
	yapf -i -r --style google --exclude 'transform/target/**/*.py' .

scrape:
	pushd scraper && scrapy crawl rumah123 -s JOBDIR=crawls/rumah123 && popd

train_catboost:
	python predictions/train_catboost.py

train_tensorflow:
	python predictions/train_tensorflow.py

dev:
	cd app && FLASK_APP=app.py FLASK_ENV=development FLASK_DEBUG=1 flask run --reload

etl:
	cd transform && dbt run && dbt docs generate

etl_test:
	cd transform && dbt test

