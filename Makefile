install:
	pip install -r requirements.txt

format:
	yapf -i -r --style yapf --exclude 'transform/target/**/*.py' --exclude 'app_docs/**/*.py' .

scrape:
	cd scraper && \
	scrapy crawl rumah123 -s JOBDIR=crawls/rumah123

dev:
	cd app && \
	streamlit run Dasbor.py

deploy: etl_docs
	docker compose up -d

etl:
	cd transform && \
	dbt run

etl_test:
	cd transform && \
	dbt test

etl_docs:
	cd transform && \
	dbt docs generate && \
	mkdir -p ../app_docs && \
	cp -rf target/* ../app_docs
