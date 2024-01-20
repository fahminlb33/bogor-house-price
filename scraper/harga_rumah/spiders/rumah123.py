import os
import re
import datetime
from urllib.parse import urlparse, urljoin

import scrapy

MONTHS = [
    "Januari", "Februari", "Maret", "April", "Mei",
    "Juni", "Juli", "Agustus", "September", "Oktober",
    "November", "Desember"
]

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
        f"https://www.rumah123.com/jual/bogor/rumah/?page={x}"
        for x in range(1, 500)
    ]

    # ----- DRIVER METHODS

    def is_property_page(self, url):
        return "/properti/" in url and "/perumahan-baru" not in url

    def parse(self, response):
        # this is detail page, extract the data
        if self.is_property_page(response.url):
            yield {
                'id': self.extract_id(response),
                'price': self.extract_price(response),
                'installment': self.extract_installment(response),
                'address': self.extract_address(response),
                'tags': self.extract_tags(response),
                'description': self.extract_description(response),
                'specs': self.extract_specs(response),
                'facilities': self.extract_facilities(response),
                'agent': self.extract_property_agent(response),
                'images': list(self.extract_images(response)),
                'url': response.url,
                'last_modified': self.extract_last_modified(response).isoformat(),
                'scraped_at': datetime.datetime.now().isoformat(),
            }

        # extract all url
        for property_url in response.xpath("//a/@href").getall():
            # check if the url is a property url
            if self.is_property_page(property_url):
                yield response.follow(property_url)

    # ----- EXTRACTOR METHODS

    def extract_id(self, response):
        return os.path.basename(os.path.normpath(response.url))

    # price in million rupiah
    def extract_price(self, response):
        # extract price
        price = response.xpath("//div[@class='r123-listing-summary__price']/span/text()").get()

        # split to get the price components
        components = price.split(" ")

        # get the raw price
        unit = components[2].lower()
        price = float(components[1].replace(",", "."))

        # check unit
        if "miliar" in unit:
            return price * 1000
        elif "juta" in unit:
            return price
        else:
            # if not Miliar or Juta, then return None
            return None

    # installment per month in million rupiah
    def extract_installment(self, response):
        # extract installment per month
        installment_per_month = response.xpath("//div[@class='r123-listing-summary__installment']/text()").get()

        # if there is no installment, then return None
        if len(installment_per_month) == 0:
            return None

        # extract number using regex
        return float(re.findall(r"\d+", installment_per_month)[0])

    def extract_images(self, response):
        # for each image, check if the image is customer uploaded image
        for current_src in response.xpath("//img/@src").getall():
            # check if the src attribute contains "/customer/"
            if "/customer/" in current_src:
                # if yes, then return the src attribute
                yield current_src

    def extract_address(self, response):
        # extract address
        return response.xpath(
            "//div[@class='r123-listing-summary__header-container-address']/text()"
        ).get()

    def extract_tags(self, response):
        # extract tags
        tags = response.xpath("//div[@class='ui-listing-overview__badge-wrapper']/div/div/text()").getall()

        # return all tags
        return [f.strip() for f in tags]

    def extract_description(self, response):
        # get the description div
        desc_div = response.xpath(
            "//p[@class='listing-description-v2__title']/following-sibling::div/div/div/div/text()"
        ).getall()

        # if there is no description, then return None
        if len(desc_div) == 0:
            return None

        # return description
        return "\n".join(desc_div)

    def extract_specs(self, response):
        # get all specs
        specs_div = response.xpath("//div[@class='listing-specification-v2__item']")

        # extract all facilities
        specs = {}
        for spec in specs_div:
            # get the row
            row = spec.xpath("./span/text()").getall()
            if len(row) != 2:
                continue

            # get the key and value
            specs[row[0]] = row[1]

        # return all facilities
        return specs

    def extract_facilities(self, response):
        # get all facilities
        facilities = response.xpath("//div[@class='ui-facilities-portal-dialog__item']/span/text()").getall() + \
                        response.xpath("//div[@class='ui-facilities-portal__item']/span/text()").getall()

        # return all facilities
        return [f.strip() for f in facilities]

    def extract_property_agent(self, response):
        # get host URL
        base_url = '{uri.scheme}://{uri.netloc}'.format(uri=urlparse(response.url))

        # to store the agent data
        agent = {}

        # extract property agent
        agent_elem = response.xpath(
            "//a[@class='r123o-m-listing-inquiry__wrapper-agent']"
        )

        if len(agent_elem) != 0:
            agent["name"] = agent_elem.xpath("./@title").get().strip()
            agent["url"] = urljoin(base_url, agent_elem.xpath("./@href").get().strip())

        # extract phone
        agent_phone = response.xpath(
            "//a[contains(@class, 'ui-organism-listing-inquiry-r123__phone-button')]/@value"
        ).get()

        if agent_phone is not None:
            agent["phone"] = agent_phone.strip()

        # extract agent company
        company_elem = response.xpath(
            "//a[@class='r123o-m-listing-inquiry__wrapper-organization']"
        )

        if len(company_elem) != 0:
            agent["company"] = {
                "name": company_elem.xpath("./@title").get().strip(),
                "url": urljoin(base_url, company_elem.xpath("./@href").get().strip()),
            }

        # return agent
        return agent

    def extract_last_modified(self, response):
        # get the last modified div
        last_modified = response.xpath(
            '//div[@class="r123-listing-summary__header-container-updated"]/text()[2]'
        ).get()

        # parse as date
        date_parts = last_modified.split(" ")
        date_parts[1] = str(MONTHS.index(date_parts[1]) + 1)

        return datetime.datetime.strptime(" ".join(date_parts), "%d %m %Y")
