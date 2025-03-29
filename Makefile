install:
	uv sync

format:
	ruff format scripts/ src/

scrape:
	cd scraper && \
	scrapy crawl rumah123 -s JOBDIR=crawls/rumah123

dev:
	cd app && \
	streamlit run Beranda.py

docker:
	docker compose up -d
