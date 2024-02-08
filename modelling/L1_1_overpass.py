import logging
import argparse

import duckdb

from etl_base import ETLMixin
from etl_constants import OVERPASS_AMENITIES_CATEGORY


class OverpassL1Pipeline(ETLMixin):

    def __init__(self, input_dir: str, output_dir: str) -> None:
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir

        # connect to database
        logging.info("Connecting to database")
        self.conn = duckdb.connect()

    def extract(self):
        # load data from JSON files
        self.conn.sql(
            f"CREATE TABLE raw AS SELECT * FROM read_json_auto('{self.input_dir}')"
        )

        # count rows
        result = self.conn.sql("SELECT COUNT(*) FROM raw").fetchone()[0]
        logging.info(f"Number of files extracted: {result}")

    def transform(self):
        # select appropiate columns
        self.conn.sql(
            "CREATE TABLE places AS "
            "SELECT element.id, rel, amenity, element.type, "
            "coalesce(element.lat, element.center.lat) AS lat, "
            "coalesce(element.lon, element.center.lon) AS lon "
            "FROM (SELECT UNNEST(data.elements) AS element, rel, amenity FROM raw)"
        )

        # fetch all rows as pandas dataframe
        self.df = self.conn.sql("SELECT * FROM places").df()

        # add amenity category
        for category, amenities in OVERPASS_AMENITIES_CATEGORY.items():
            # add new column
            self.df.loc[self.df['amenity'].isin(amenities),
                        'category'] = category

        # count total rows
        self.processed_count = len(self.df)

    def load(self):
        logging.info(f"Processed {self.processed_count} records")
        self.df.to_parquet(f"{self.output_dir}/L1.overpass.parquet")


if __name__ == '__main__':
    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        help="Input directory containing JSON files",
                        default="./dataset/osm")
    parser.add_argument("--output-dir",
                        help="Output directory for Parquet files",
                        default="./dataset/etl")

    args = parser.parse_args()

    # run pipeline
    pipeline = OverpassL1Pipeline(args.input_dir, args.output_dir)
    pipeline.run()
