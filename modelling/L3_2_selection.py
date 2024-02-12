import logging
import argparse
from typing import Literal

import numpy as np
import pandas as pd

from etl_base import ETLMixin, MultipleArgsParser
from etl_constants import (SELECTION_DEFAULT_MODE, SELECTION_DEFAULT_MIN_R,
                           SELECTION_DEFAULT_CRIT_PVALUE,
                           SELECTION_DROP_COLUMNS)

SELECTION_MODE = Literal["r", "pvalue", "manual", "all"]


class HouseL3SelectionPipeline(ETLMixin):

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 selection_mode: SELECTION_MODE = SELECTION_DEFAULT_MODE,
                 min_r: float = SELECTION_DEFAULT_MIN_R,
                 crit_pvalue: float = SELECTION_DEFAULT_CRIT_PVALUE,
                 manual_cols: list = []):
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.selection_mode = selection_mode
        self.min_r = min_r
        self.crit_pvalue = crit_pvalue
        self.manual_cols = manual_cols

    def extract(self):
        self.df_corr = pd.read_csv(
            f"{self.input_dir}/L3.correlations_features.csv")
        self.df = pd.read_parquet(
            f"{self.input_dir}/L2.regression_inliers.parquet")

    def transform(self):
        # columns with low correlation
        significant_cols = []
        if self.selection_mode == "r":
            significant_cols = self.df_corr[np.abs(self.df_corr["r"]) >
                                            self.min_r]["variable"].tolist()
        elif self.selection_mode == "pvalue":
            significant_cols = self.df_corr[
                self.df_corr["pvalue"] < self.crit_pvalue]["variable"].tolist()
        elif self.selection_mode == "manual":
            significant_cols = self.manual_cols
            print(significant_cols)
        else:
            # all columns
            significant_cols = list(self.df.columns)

        logging.info(f"Significant columns: {significant_cols}")

        # column candidates
        drop_cols = [
            col for col in self.df.columns.tolist()
            if col not in significant_cols
        ] + SELECTION_DROP_COLUMNS + [
            col for col in self.df.columns if col.startswith("spatial_")
        ]

        # drop unused columns
        self.df = self.df.drop(columns=list(set(drop_cols) - set(["price"])),
                               errors="ignore")

    def load(self):
        self.df.info()
        self.df.to_parquet(f"{self.output_dir}/L3.regression_train.parquet")


if __name__ == "__main__":
    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        help="Input directory containing Parquet files",
                        default="./dataset/etl")
    parser.add_argument("--output-dir",
                        help="Output directory for Parquet files",
                        default="./dataset/etl")
    parser.add_argument("--selection-mode",
                        help="Selection mode (r, pvalue, manual, or all)",
                        default=SELECTION_DEFAULT_MODE)
    parser.add_argument("--min-r",
                        help="Minimum correlation coefficient",
                        default=SELECTION_DEFAULT_MIN_R,
                        type=float)
    parser.add_argument("--crit-pvalue",
                        help="Critical p-value for significance",
                        default=SELECTION_DEFAULT_CRIT_PVALUE,
                        type=float)
    parser.add_argument("--manual-cols",
                        help="List of manually selected columns",
                        type=MultipleArgsParser)

    args = parser.parse_args()

    # run pipeline
    pipeline = HouseL3SelectionPipeline(args.input_dir, args.output_dir,
                                        args.selection_mode, args.min_r,
                                        args.crit_pvalue, args.manual_cols)
    pipeline.run()
