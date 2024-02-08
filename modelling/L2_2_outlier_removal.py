import argparse
import logging

import pandas as pd

from etl_base import ETLMixin
from etl_constants import (OUTLIERS_DEFAULT_IQR_THRESHOLD,
                           OUTLIERS_MAX_BEDROOMS, OUTLIERS_MAX_LAND_AREA)


class HouseL2OutliersPipeline(ETLMixin):

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 iqr_threshold: float = OUTLIERS_DEFAULT_IQR_THRESHOLD):
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.iqr_threshold = iqr_threshold

    def extract(self):
        self.df = pd.read_parquet(f"{self.input_dir}/L2.raw_features.parquet")

    def transform(self):
        # rows before removing outliers
        rows_before = self.df.shape[0]
        logging.info(f"Rows before removing outliers: {rows_before}")

        # calculate quantiles and IQR
        q1 = self.df["price"].quantile(0.25)
        q3 = self.df["price"].quantile(0.75)

        iqr = q3 - q1
        lower_bound = q1 - (self.iqr_threshold * iqr)
        upper_bound = q3 + (self.iqr_threshold * iqr)

        # remove outliers
        self.df = self.df[(self.df["price"] > lower_bound) &
                          (self.df["price"] < upper_bound)]

        # remove more outlier
        self.df = self.df[self.df["kamar_tidur"] < OUTLIERS_MAX_BEDROOMS]
        self.df = self.df[self.df["luas_tanah"] < OUTLIERS_MAX_LAND_AREA]

        # rows after removing outliers
        rows_after = self.df.shape[0]
        logging.info(
            f"Rows after removing outliers: {rows_after} (removed {rows_before - rows_after} outliers)"
        )

    def load(self):
        self.df.to_parquet(f"{self.output_dir}/L2.regression_inliers.parquet")


if __name__ == "__main__":
    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        help="Input directory containing Parquet files",
                        default="./dataset/etl")
    parser.add_argument("--output-dir",
                        help="Output directory for Parquet files",
                        default="./dataset/etl")
    parser.add_argument("--iqr-threshold",
                        help="IQR threshold for outlier detection",
                        default=OUTLIERS_DEFAULT_IQR_THRESHOLD,
                        type=float)

    args = parser.parse_args()

    # run pipeline
    pipeline = HouseL2OutliersPipeline(args.input_dir, args.output_dir)
    pipeline.run()
