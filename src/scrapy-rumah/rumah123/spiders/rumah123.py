import os
import re
import datetime
from urllib.parse import urlparse, urljoin, unquote

import scrapy

from ..items import RumahItem

MONTHS = [
    "Januari",
    "Februari",
    "Maret",
    "April",
    "Mei",
    "Juni",
    "Juli",
    "Agustus",
    "September",
    "Oktober",
    "November",
    "Desember",
]

def parse_price(s):
    if (value := re.search(r"\d*[.,]?\d+", s)):
        return float(value.group(0).replace(",", "."))
    
    return None

#
# Rumah123.com Spider
#
# Listing page: https://www.rumah123.com/jual/bogor/rumah/?page=1
# Detail page:  https://www.rumah123.com/properti/bogor/hos15772001/
#
class Rumah123Spider(scrapy.Spider):
    name = "rumah123"
    allowed_domains = ["rumah123.com"]
    start_urls = [
        f"https://www.rumah123.com/jual/bogor/rumah/?page={x}" for x in range(1, 500)
    ]

    # ----- DRIVER METHODS

    def is_property_page(self, url):
        return "/properti/" in url and "/perumahan-baru" not in url

    def parse(self, response):
        # this is detail page, extract the data
        if self.is_property_page(response.url):
            yield RumahItem(
                # base info
                id=self.extract_id(response),
                price=self.extract_price(response),
                installment=self.extract_installment(response),
                # descriptions
                address=self.extract_address(response),
                description=self.extract_description(response),
                tags=self.extract_tags(response),
                specs=self.extract_specs(response),
                facilities=self.extract_facilities(response),
                agent=self.extract_property_agent(response),
                # images
                images=list(self.extract_images(response)),
                # meta
                url=response.url,
                last_modified=self.extract_last_modified(response).isoformat(),
                scraped_at=datetime.datetime.now().isoformat(),
            )

        # extract all url
        for property_url in response.xpath("//a/@href").getall():
            # check if the url is a property url
            if self.is_property_page(property_url):
                yield response.follow(property_url)

    # ----- EXTRACTOR METHODS

    

    def extract_id(self, response):
        return os.path.basename(os.path.normpath(response.url))

    def extract_price(self, response):
        # price in million rupiah
        # extract price
        price_text = response.xpath('//div[contains(@class, "ui-container")]//div[contains(.//span, "Rp ")]//span/text()').get().lower()

        # get the raw price
        price = parse_price(price_text)

        # check unit
        if "miliar" in price_text:
            return price * 1000
        elif "juta" in price_text:
            return price
        else:
            # if not Miliar or Juta, then return None
            return None

    def extract_installment(self, response):
        # installment per month in million rupiah
        # extract installment per month
        installment_elem = response.xpath('//div[contains(@class, "ui-container")]//p[contains(., "Cicilan")]')

        # if there is no installment, then return None
        if installment_elem is None:
            return None

        value_text = installment_elem.xpath("string(.)").get()
        return parse_price(value_text)

    def extract_address(self, response):
        # extract address
        return response.xpath('//div[contains(@class, "ui-container")]//div/h1/following-sibling::p[1]/text()').get()
    
    def extract_last_modified(self, response):
        # get the last modified div
        last_modified = response.xpath('string(//p[contains(., "Diperbarui")])').get()
        last_modified_text = re.search(r"\d{1,2} \w+ \d{4}", last_modified).group(0)

        # parse as date
        date_parts = last_modified_text.split(" ")
        date_parts[1] = str(MONTHS.index(date_parts[1]) + 1)

        return datetime.datetime.strptime(" ".join(date_parts), "%d %m %Y")

    def extract_tags(self, response):
        # extract tags
        tags = response.xpath(
            '//div[contains(@class, "ui-container")]//p/following-sibling::ul/li/text()'
        ).getall()

        # return all tags
        return [f.strip() for f in tags]

    def extract_description(self, response):
        # get the description div
        desc = response.xpath(
            '//div[contains(@class, "ui-container")]//p[contains(., "ripsi")]/following-sibling::p/text()'
        ).get()

        return desc

    def extract_specs(self, response):
        # get all specs
        specs_div = response.xpath('//div[contains(@class, "ui-container")]//p[contains(., "pesifikasi")]/following-sibling::div//div[contains(@class, "mb-4")]')

        # extract all
        specs = {}
        for spec in specs_div:
            # get the row
            row = spec.xpath("./p/text()").getall()
            if len(row) != 2:
                continue

            # get the key and value
            specs[row[0]] = row[1]

        # return all facilities
        return specs

    def extract_facilities(self, response):
        # get all facilities
        facilities = response.xpath('//div[contains(@id, "facility")]//p/span/text()').getall()

        # return all facilities
        return [f.strip() for f in facilities]

    def extract_property_agent(self, response):
        # get host URL
        base_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(response.url))

        # to store the agent data
        agent = {}

        # extract property agent
        agent_elem = response.xpath('//div[contains(@class, "ui-container")]/div[2]//div[contains(@class, "flex-grow")]/a[@title]')

        if len(agent_elem) == 0:
            return agent
        
        if len(agent_elem) > 0:
            agent["name"] = agent_elem[0].xpath("@title").get().strip()
            agent["url"] = urljoin(base_url, agent_elem[0].xpath("@href").get().strip())
            agent["phone"] = response.xpath('//button[@name="phone" and contains(@title, "62")]/@title').get()

        if len(agent_elem) > 1:
            agent["company"] = {
                "name": agent_elem[1].xpath("@title").get().strip(),
                "url": urljoin(base_url, agent_elem[1].xpath("@href").get().strip()),
            }

        return agent

    def extract_images(self, response):
        # for each image, check if the image is customer uploaded image
        images = response.xpath('//img[contains(@srcset, "1x") and not(contains(@alt, "map"))]/@srcset').getall()
        for srcset in images:
            result = re.search(r"url=(.+) \dx,", srcset)
            if result is None:
                continue

            url = unquote(result.group(1))

            # check if the src attribute contains "/customer/"
            if "/customer/" in url:
                yield url[:url.find('&')]
