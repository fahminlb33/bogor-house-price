import time
import json
import argparse
import logging
import itertools
from urllib.parse import quote, urljoin

import requests

from etl_base import ProcessorBase, MultipleArgsParser
from etl_constants import (OVERPASS_DEFAULT_BASE_URL, OVERPASS_DEFAULT_TIMEOUT,
                           OVERPASS_DEFAULT_DOWNLOAD_SLEEP,
                           OVERPASS_DEFAULT_RELATIONS,
                           OVERPASS_DEFAULT_AMENITIES)


class OverpassL0DownloadPipeline(ProcessorBase):

    def __init__(self,
                 rels: list[str],
                 amenities: list[str],
                 output_dir: str,
                 base_url: str = OVERPASS_DEFAULT_BASE_URL,
                 timeout: int = OVERPASS_DEFAULT_TIMEOUT,
                 sleep: int = OVERPASS_DEFAULT_DOWNLOAD_SLEEP) -> None:
        super().__init__()

        # set attributes
        self.rels = rels
        self.amenities = amenities
        self.output_dir = output_dir
        self.base_url = base_url
        self.timeout = timeout
        self.sleep = sleep

    def extract(self, root_area, tags={}) -> dict:
        # build query
        tags_query = ";".join([f"{k}=\"{v}\"" for k, v in tags.items()])
        overpassQL = "data=" + quote("[out:json]"
                                     f"[timeout:{self.timeout}];"
                                     f"rel({root_area});map_to_area;"
                                     f"nwr(area)[{tags_query}];"
                                     "out center;")

        # send request
        response = requests.post(
            urljoin(self.base_url, f"/api/interpreter"),
            timeout=self.timeout,
            data=overpassQL,
            headers={"Content-Type": "application/x-www-form-urlencoded"})

        # check response
        if response.status_code != 200:
            raise Exception('Download failed')

        return response.json()

    def load(self):
        # iterate over all combinations of root areas and amenities
        for root_area, tag in itertools.product(self.rels, self.amenities):
            logging.info(
                f"Downloading data for relation '{root_area}' with tag '{tag}'")
            data = self.extract(root_area, {"amenity": tag})

            # save to file
            with open(f"{self.output_dir}/{root_area}_{tag}.json", "w") as f:
                json.dump({
                    "rel": root_area,
                    "amenity": tag,
                    "data": data,
                }, f)

            logging.info(
                f"Saved {root_area}_{tag}.json with {len(data['elements'])} elements"
            )
            time.sleep(self.sleep)

    def run(self):
        logging.info("Starting Overpass API download pipeline")
        self.load()


if __name__ == '__main__':
    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--rels",
                        nargs=MultipleArgsParser,
                        help="Root areas to fetch",
                        default=OVERPASS_DEFAULT_RELATIONS)
    parser.add_argument("--amenities",
                        nargs=MultipleArgsParser,
                        help="Amenities to fetch",
                        default=OVERPASS_DEFAULT_AMENITIES)
    parser.add_argument("--output-dir",
                        help="Output directory for Parquet files",
                        default="./dataset/osm")
    parser.add_argument("--base-url",
                        help="Base URL for Overpass API",
                        default=OVERPASS_DEFAULT_BASE_URL)
    parser.add_argument("--timeout",
                        help="Timeout for requests",
                        default=OVERPASS_DEFAULT_TIMEOUT,
                        type=int)
    parser.add_argument("--sleep",
                        help="Timeout for requests",
                        default=OVERPASS_DEFAULT_DOWNLOAD_SLEEP,
                        type=int)

    args = parser.parse_args()

    # run pipeline
    pipeline = OverpassL0DownloadPipeline(args.rels, args.amenities,
                                          args.output_dir, args.base_url,
                                          args.timeout, args.sleep)
    pipeline.run()
