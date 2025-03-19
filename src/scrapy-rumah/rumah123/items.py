# Define here the models for your scraped items
#
# See documentation in:
# https://docs.scrapy.org/en/latest/topics/items.html

import scrapy


class RumahItem(scrapy.Item):
    id = scrapy.Field()
    price = scrapy.Field()
    installment = scrapy.Field()

    address = scrapy.Field()
    description = scrapy.Field()
    tags = scrapy.Field()
    specs = scrapy.Field()
    facilities = scrapy.Field()
    agent = scrapy.Field()

    images = scrapy.Field()
    image_paths = scrapy.Field()

    url = scrapy.Field()
    last_modified = scrapy.Field()
    scraped_at = scrapy.Field()
    