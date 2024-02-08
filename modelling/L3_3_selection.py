import argparse
import logging
from typing import Literal

import pandas as pd

from etl_base import ETLMixin
from etl_constants import (SELECTION_DEFAULT_MODE, SELECTION_DEFAULT_MIN_R,
                           SELECTION_DEFAULT_CRIT_PVALUE,
                           SELECTION_DROP_COLUMNS)

SELECTION_MODE = Literal["r", "pvalue", "manual"]


class HouseL3SelectionPipeline(ETLMixin):

    def __init__(self,
                 input_dir: str,
                 output_dir: str,
                 selection_mode: SELECTION_MODE = SELECTION_DEFAULT_MODE,
                 min_r: float = SELECTION_DEFAULT_MIN_R,
                 crit_pvalue: float = SELECTION_DEFAULT_CRIT_PVALUE):
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir
        self.selection_mode = selection_mode
        self.min_r = min_r
        self.crit_pvalue = crit_pvalue

    def extract(self):
        self.df_corr = pd.read_parquet(
            f"{self.input_dir}/L3.correlations.parquet")
        self.df = pd.read_parquet(
            f"{self.input_dir}/L2.regression_inliers.parquet")

    def transform(self):
        # columns with low correlation
        significant_cols = []
        if self.selection_mode == "r":
            significant_cols = self.df_corr[self.df_corr["r"] >
                                            self.min_r]["variable"].tolist()
        elif self.selection_mode == "pvalue":
            significant_cols = self.df_corr[
                self.df_corr["pvalue"] < self.crit_pvalue]["variable"].tolist()

        # drop unused columns
        self.df = self.df.drop(columns=SELECTION_DROP_COLUMNS +
                               significant_cols)

    def load(self):
        print(self.df.info())
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
                        help="Selection mode (r, pvalue, or manual)",
                        default=SELECTION_DEFAULT_MODE)
    parser.add_argument("--min-r",
                        help="Minimum correlation coefficient",
                        default=SELECTION_DEFAULT_MIN_R,
                        type=float)
    parser.add_argument("--crit-pvalue",
                        help="Critical p-value for significance",
                        default=SELECTION_DEFAULT_CRIT_PVALUE,
                        type=float)

    args = parser.parse_args()

    # run pipeline
    pipeline = HouseL3SelectionPipeline(args.input_dir, args.output_dir,
                                        args.selection_mode, args.min_r,
                                        args.crit_pvalue)
    pipeline.run()
