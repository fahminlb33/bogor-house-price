# Scrapy settings for rumah123 project
#
# For simplicity, this file contains only settings considered important or
# commonly used. You can find more settings consulting the documentation:
#
#     https://docs.scrapy.org/en/latest/topics/settings.html
#     https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
#     https://docs.scrapy.org/en/latest/topics/spider-middleware.html

BOT_NAME = "rumah123"

SPIDER_MODULES = ["rumah123.spiders"]
NEWSPIDER_MODULE = "rumah123.spiders"

# Common settings
# LOG_LEVEL = "INFO"
# TELNETCONSOLE_ENABLED = False
TARGET_DOWNLOAD_DIR = "/home/fahmi/projects/project-rumah-regresi/data/rumah123"

# Jobs to support resumable download
# JOBDIR = f"{TARGET_DOWNLOAD_DIR}/crawl-job"

# Crawling rule
ROBOTSTXT_OBEY = True
COOKIES_ENABLED = True

# Concurrency
CONCURRENT_REQUESTS = 2
# CONCURRENT_REQUESTS_PER_DOMAIN = 16
# CONCURRENT_REQUESTS_PER_IP = 2

# Retry and delay
RETRY_TIMES = 5
DOWNLOAD_DELAY = 3

# Enable or disable spider middlewares
# See https://docs.scrapy.org/en/latest/topics/spider-middleware.html
SPIDER_MIDDLEWARES = {
    # "rumah123.middlewares.RumahSpiderMiddleware": 543,
}

# Enable or disable downloader middlewares
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html
DOWNLOADER_MIDDLEWARES = {
    # "rumah123.middlewares.RumahDownloaderMiddleware": 543,
}

# Enable or disable extensions
# See https://docs.scrapy.org/en/latest/topics/extensions.html
EXTENSIONS = {
    "scrapy.extensions.closespider.CloseSpider": 500,
}

CLOSESPIDER_ERRORCOUNT = 5

# Configure item pipelines
# See https://docs.scrapy.org/en/latest/topics/item-pipeline.html
ITEM_PIPELINES = {
    "rumah123.pipelines.RumahImagePipeline": 300,
}

# Image pipeline
IMAGES_STORE = f"{TARGET_DOWNLOAD_DIR}/images"
IMAGES_URLS_FIELD = "images"
IMAGES_RESULT_FIELD = "image_paths"
MEDIA_ALLOW_REDIRECTS = True

# Enable and configure the AutoThrottle extension (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/autothrottle.html
AUTOTHROTTLE_ENABLED = True
AUTOTHROTTLE_START_DELAY = 5
AUTOTHROTTLE_MAX_DELAY = 120
AUTOTHROTTLE_TARGET_CONCURRENCY = 1
# AUTOTHROTTLE_DEBUG = False

# HTTP config
USER_AGENT = "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
# DEFAULT_REQUEST_HEADERS = {
#    "Accept": "text/html,application/xhtml+xml,application/xml;q=0.9,*/*;q=0.8",
#    "Accept-Language": "en",
# }

# Enable and configure HTTP caching (disabled by default)
# See https://docs.scrapy.org/en/latest/topics/downloader-middleware.html#httpcache-middleware-settings
# HTTPCACHE_ENABLED = True
# HTTPCACHE_EXPIRATION_SECS = 0
# HTTPCACHE_DIR = "httpcache"
# HTTPCACHE_IGNORE_HTTP_CODES = []
# HTTPCACHE_STORAGE = "scrapy.extensions.httpcache.FilesystemCacheStorage"

# Request Fingerprinting
REQUEST_FINGERPRINTER_IMPLEMENTATION = "2.7"
TWISTED_REACTOR = "twisted.internet.asyncioreactor.AsyncioSelectorReactor"

# Feed export settings
FEED_EXPORT_ENCODING = "utf-8"
FEEDS = {
    f"{TARGET_DOWNLOAD_DIR}/rumah123.json": {
        "format": "jsonlines"
    },
}
