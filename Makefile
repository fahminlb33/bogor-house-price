install:
	pip install -r requirements.txt

format:
	yapf -i -r --style google --exclude 'transform/target/**/*.py' --exclude 'app_docs/**/*.py' .

scrape:
	cd scraper && \
	scrapy crawl rumah123 -s JOBDIR=crawls/rumah123

dev:
	cd app && \
	FLASK_APP=app.py FLASK_ENV=development FLASK_DEBUG=1 flask run --reload

deploy: etl_docs
	docker compose up -d

etl:
	cd transform && \
	dbt run && \
	dbt docs generate

etl_test:
	cd transform && \
	dbt test

etl_docs:
	cd transform && \
	dbt docs generate && \
	cp -rf target/* ../app_docs
