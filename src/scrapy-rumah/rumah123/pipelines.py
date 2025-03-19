# Define your item pipelines here
#
# Don't forget to add your pipeline to the ITEM_PIPELINES setting
# See: https://docs.scrapy.org/en/latest/topics/item-pipeline.html

# useful for handling different item types with a single interface
from pathlib import PurePosixPath

from scrapy.pipelines.images import ImagesPipeline
from scrapy.utils.httpobj import urlparse_cached


class RumahImagePipeline(ImagesPipeline):
    def file_path(self, request, response=None, info=None, *, item=None):
        item_id = item["id"]
        filename = PurePosixPath(urlparse_cached(request).path).name
        
        return f"{item_id}/{filename}"
