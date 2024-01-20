SHELL := /bin/bash

install:
	pip install -r requirements.txt

format:
	yapf -i -r --style google .

scrape:
	pushd scraper && scrapy crawl rumah123 -s JOBDIR=crawls/rumah123 && popd

train_catboost:
	python predictions/train_catboost.py

train_tensorflow:
	python predictions/train_tensorflow.py
