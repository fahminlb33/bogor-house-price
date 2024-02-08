import argparse
import logging

import numpy as np
import pandas as pd

from etl_base import ETLMixin
from etl_constants import IMPUTE_RULES


class HouseL2RegressionPipeline(ETLMixin):

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir

    def extract(self):
        self.df_houses = pd.read_parquet(f"{self.input_dir}/L1.houses.parquet")
        self.df_tags = pd.read_parquet(
            f"{self.input_dir}/L1.house_tags.parquet")
        self.df_specs = pd.read_parquet(
            f"{self.input_dir}/L1.house_specs.parquet")
        self.df_facility = pd.read_parquet(
            f"{self.input_dir}/L1.house_facilities.parquet")
        self.df_material = pd.read_parquet(
            f"{self.input_dir}/L1.house_material.parquet")
        self.df_floor_material = pd.read_parquet(
            f"{self.input_dir}/L1.house_floor_material.parquet")

    def transform_merge(self):
        # transform using crosstab
        df_tags = pd.crosstab(self.df_tags["reference_id"], self.df_tags["tag"]) \
            .reset_index() \
            .add_prefix("tags_")
        df_material = pd.crosstab(self.df_material["reference_id"], self.df_material["material"]) \
            .reset_index() \
            .add_prefix("house_mat_")
        df_floor_material = pd.crosstab(self.df_floor_material["reference_id"], self.df_floor_material["material"]) \
            .reset_index() \
            .add_prefix("floor_mat_")
        df_facility = pd.crosstab(self.df_facility["reference_id"], self.df_facility["facility"]) \
            .reset_index() \
            .add_prefix("facility_")

        # merge all the dataframes
        df = self.df_houses.merge(self.df_specs,
                                  left_on="id",
                                  right_on="reference_id",
                                  how="left")
        df = df.merge(df_tags,
                      left_on="id",
                      right_on="tags_reference_id",
                      how="left")
        df = df.merge(df_material,
                      left_on="id",
                      right_on="house_mat_reference_id",
                      how="left")
        df = df.merge(df_floor_material,
                      left_on="id",
                      right_on="floor_mat_reference_id",
                      how="left")
        df = df.merge(df_facility,
                      left_on="id",
                      right_on="facility_reference_id",
                      how="left")
        df = df.drop(columns=[
            "reference_id", "tags_reference_id", "house_mat_reference_id",
            "floor_mat_reference_id", "facility_reference_id"
        ])

        self.df = df

    def transform(self):
        # merge dataframes
        self.transform_merge()

        # derive district and city
        self.df["district"] = self.df["address"].str.split(",").str[0]
        self.df["city"] = self.df["address"].str.split(",").str[1]
        self.df = self.df.drop(columns=["address"])

        # replace NaN with NA
        self.df = self.df.replace(np.nan, pd.NA)

        # encode boolean features
        bool_map = {False: 0, True: 1}
        self.df["hook"] = self.df["hook"].map(bool_map)
        self.df["ruang_tamu"] = self.df["ruang_tamu"].map(bool_map)
        self.df["ruang_makan"] = self.df["ruang_makan"].map(bool_map)
        self.df["terjangkau_internet"] = self.df["terjangkau_internet"].map(
            bool_map)

        # impute missing values
        for rule in IMPUTE_RULES:
            if rule["method"] == "mode":
                self.df[rule["col"]] = self.df[rule["col"]].fillna(
                    self.df[rule["col"]].mode()[0])
            elif rule["method"] == "mean":
                self.df[rule["col"]] = self.df[rule["col"]].fillna(
                    self.df[rule["col"]].mean())
            elif rule["method"] == "median":
                self.df[rule["col"]] = self.df[rule["col"]].fillna(
                    self.df[rule["col"]].median())
            elif rule["method"] == "constant":
                self.df[rule["col"]] = self.df[rule["col"]].fillna(
                    rule["value"])

    def load(self):
        logging.info("Dataframe info:\n" + self.df.info())
        self.df.to_parquet(f"{self.output_dir}/L2.raw_features.parquet")


if __name__ == "__main__":
    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        help="Input directory containing Parquet files",
                        default="./dataset/etl")
    parser.add_argument("--output-dir",
                        help="Output directory for Parquet files",
                        default="./dataset/etl")

    args = parser.parse_args()

    # run pipeline
    pipeline = HouseL2RegressionPipeline(args.input_dir, args.output_dir)
    pipeline.run()
