import argparse
import logging

import numpy as np
import pandas as pd
import seaborn as sns
import geopandas as gpd
import scipy.stats as stats
import matplotlib.pyplot as plt

from etl_base import ETLMixin
from etl_constants import (SPATIAL_SHP_DROP_COLUMNS, SPATIAL_OVERPASS_CRS,
                           SPATIAL_GEODETIC_CRS, SPATIAL_PLACE_NORM_RULES)


def pointbiser_corr(dfp: pd.DataFrame, prefix: str) -> pd.DataFrame:
    # select proper column
    cols = [
        x for x in dfp.select_dtypes(include=["number", "bool"]).columns
        if prefix in x
    ]

    # calculate point biserial correlation for each column
    corrs = []
    for variable in cols:
        corr = stats.pointbiserialr(dfp[variable], dfp["price"])
        corrs.append({
            "variable": variable,
            "method": "pointbiser",
            "r": corr[0],
            "pvalue": corr[1]
        })

    return pd.DataFrame(corrs)


def pearsonr_corr(dfp: pd.DataFrame) -> pd.DataFrame:
    # select proper column
    cols = [x for x in dfp.select_dtypes(include="number").columns]
    cols = [
        x for x in cols if "price" not in x and "tags_" not in x and
        "facility_" not in x and "_mat_" not in x
    ]

    # calculate Pearson correlation for each column
    corrs = []
    for variable in cols:
        corr = stats.pearsonr(dfp[variable], dfp["price"])
        corrs.append({
            "variable": variable,
            "method": "pearson",
            "r": corr[0],
            "pvalue": corr[1]
        })

    return pd.DataFrame(corrs)


class HouseL3CorrelationsPipeline(ETLMixin):

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.corrs = []

    def extract(self):
        # load houses dataset
        self.df = pd.read_parquet(
            f"{self.input_dir}/etl/L2.regression_inliers.parquet")

        # load shapefile dataset
        self.df_bogor = pd.concat([
            gpd.read_file(
                f"{self.input_dir}/shp/kota/ADMINISTRASIDESA_AR_25K.shp"),
            gpd.read_file(
                f"{self.input_dir}/shp/kab/ADMINISTRASIDESA_AR_25K.shp")
        ])

        self.df_bogor = self.df_bogor.drop(columns=SPATIAL_SHP_DROP_COLUMNS)

        # load overpass dataset
        df_overpass_raw = pd.read_parquet(
            f"{self.input_dir}/etl/L1.overpass.parquet")
        df_overpass = gpd.GeoDataFrame(df_overpass_raw,
                                       geometry=gpd.points_from_xy(
                                           df_overpass_raw.lon,
                                           df_overpass_raw.lat),
                                       crs=SPATIAL_OVERPASS_CRS)
        df_overpass = df_overpass.drop(columns=["lat", "lon"])

        self.df_overpass = df_overpass

    def transform_houses_corr(self):
        # correlations between attributes
        self.corrs.append(pointbiser_corr(self.df, "tags_"))
        self.corrs.append(pointbiser_corr(self.df, "facility_"))
        self.corrs.append(pointbiser_corr(self.df, "house_mat"))
        self.corrs.append(pointbiser_corr(self.df, "floor_mat"))
        self.corrs.append(pointbiser_corr(self.df, "makan"))
        self.corrs.append(pointbiser_corr(self.df, "tamu"))
        self.corrs.append(pointbiser_corr(self.df, "internet"))
        self.corrs.append(pointbiser_corr(self.df, "hook"))
        self.corrs.append(pearsonr_corr(self.df))

    def load_houses_corr(self):
        # calculate matrix correlation
        cols = [x for x in self.df.select_dtypes(include="number").columns]
        cols = [
            x for x in cols if "tags_" not in x and "facility_" not in x and
            "_mat_" not in x and "makan" not in x and "tamu" not in x and
            "internet" not in x and "hook" not in x
        ]

        corr = self.df[cols].corr(numeric_only=True).round(2)

        # save image
        fig, ax = plt.subplots(figsize=(12, 8))
        sns.heatmap(corr, annot=True, ax=ax, cmap="coolwarm")
        fig.tight_layout()
        fig.savefig(f"{self.output_dir}/L3.corr_features.png")

    def transform_spatial_corr(self):
        # normalize place names and price
        self.df["place"] = self.df["district"].replace(SPATIAL_PLACE_NORM_RULES)
        self.df["price"] = np.log(self.df["price"])

        avg_house_prices = self.df.groupby("place")["price"].mean()

        # join spatial datasets (points from overpass and polygons from shapefile)
        df_places = gpd.sjoin_nearest(self.df_bogor.to_crs(SPATIAL_GEODETIC_CRS), self.df_overpass.to_crs(SPATIAL_GEODETIC_CRS), distance_col="distance") \
            .drop(columns=["index_right"])

        # convert to pandas DataFrame
        df_places = pd.DataFrame(
            df_places.drop(columns=["geometry", "SHAPE_Leng", "SHAPE_Area"]))

        # calculate number of amenities per place
        df_amenities = pd.concat([
            # df_places.pivot_table(index="WADMKC", columns="category", values="id", aggfunc="count", fill_value=0) \
            # .reset_index() \
            # .rename(columns={"WADMKC": "place"}),

            df_places.pivot_table(index="NAMOBJ", columns="category", values="id", aggfunc="count", fill_value=0) \
                .reset_index() \
                .rename(columns={"NAMOBJ": "place"}),
        ])

        # normalize place names
        df_amenities["place"] = df_amenities["place"].replace(
            SPATIAL_PLACE_NORM_RULES)

        # join with average house prices
        self.df_spatial = df_amenities.join(avg_house_prices, on="place") \
            .reset_index(drop=True) \
            .dropna()

        # calculate pearson correlation
        df_corr = self.df_spatial.add_prefix("spatial_").rename(
            {"spatial_price": "price"}, axis=1)
        self.corrs.append(pearsonr_corr(df_corr))

    def load_spatial_corr(self):
        corr = self.df_spatial.corr(numeric_only=True).round(2)

        # save image
        fig, ax = plt.subplots()
        sns.heatmap(corr, annot=True, ax=ax, cmap="coolwarm")
        fig.tight_layout()
        fig.savefig(f"{self.output_dir}/L3.corr_spatial.png")

    def transform(self):
        logging.info("Processing correlations for houses dataset")
        self.transform_houses_corr()

        logging.info("Processing correlations for spatial dataset")
        self.transform_spatial_corr()

    def load(self):
        # save correlations to file
        df_house_corr = pd.concat(self.corrs, ignore_index=True)
        df_house_corr.to_csv(f"{self.output_dir}/L3.correlations_features.csv",
                             index=None)
        self.df_spatial.to_csv(
            f"{self.output_dir}/L3.correlations_spatial_features.csv",
            index=None)

        # save images
        self.load_houses_corr()
        self.load_spatial_corr()


if __name__ == "__main__":
    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        help="Input directory containing Parquet files",
                        default="./dataset")
    parser.add_argument("--output-dir",
                        help="Output directory for Parquet files",
                        default="./dataset/etl")

    args = parser.parse_args()

    # run pipeline
    pipeline = HouseL3CorrelationsPipeline(args.input_dir, args.output_dir)
    pipeline.run()
