import os
import argparse

import pandas as pd
import scipy.stats as stats
from pandas.api.types import is_numeric_dtype
from sklearn.model_selection import train_test_split

from utils.ml_base import TrainerMixin

# define configuration
TEST_PERCENTAGE = 0.2
RANDOM_STATE = 21
MANUAL_SUBSET_COLS = [
    "price",
    "luas_tanah",
    "luas_bangunan",
    "kamar_tidur",
    "kamar_mandi",
    "kamar_pembantu",
    "kamar_mandi_pembantu",
    "daya_listrik",
    "jumlah_lantai",
    "lebar_jalan",
    "carport",
    "dapur",
    "ruang_makan",
    "ruang_tamu",
    "facility_ac",
    "facility_keamanan",
    "facility_laundry",
    "facility_masjid",
    "house_mat_bata_hebel",
    "house_mat_bata_merah",
    "tag_cash_bertahap",
    "tag_komplek",
    "tag_kpr",
    "tag_perumahan",
]


class DatasetSplitter(TrainerMixin):

  def __init__(self,
               dataset_path: str,
               output_dir: str,
               test_percentage=TEST_PERCENTAGE,
               subset_mode="all",
               random_state=RANDOM_STATE) -> None:
    super().__init__()

    self.dataset_path = dataset_path
    self.output_dir = output_dir
    self.test_percentage = test_percentage
    self.subset_mode = subset_mode
    self.random_state = random_state

  def load_data(self):
    self.df = pd.read_parquet(self.dataset_path)

  @staticmethod
  def should_use_point_biser_corr(column):
    return "tag_" in column or "facility_" in column or "house_" in column

  @staticmethod
  def impute(df_train: pd.DataFrame, df_test: pd.DataFrame, column: str, method: str, constant_value=""):
    if method == "mean":
      df_train[column] = df_train[column].fillna(df_train[column].mean())
      df_test[column] = df_test[column].fillna(df_train[column].mean())
    elif method == "median":
      df_train[column] = df_train[column].fillna(df_train[column].median())
      df_test[column] = df_test[column].fillna(df_train[column].median())
    elif method == "mode":
      df_train[column] = df_train[column].fillna(
          df_train[column].mode(dropna=True).iloc[0])
      df_test[column] = df_test[column].fillna(
          df_train[column].mode(dropna=True).iloc[0])
    elif method == "constant":
      df_train[column] = df_train[column].fillna(constant_value)
      df_test[column] = df_test[column].fillna(constant_value)

  def train(self):
    # split train and test
    df_train, df_test = train_test_split(
        self.df, test_size=self.test_percentage, random_state=self.random_state)

    # impute missing values
    DatasetSplitter.impute(df_train, df_test, "luas_tanah", "mean")
    DatasetSplitter.impute(df_train, df_test, "luas_bangunan", "mean")

    DatasetSplitter.impute(df_train, df_test, "daya_listrik", "mode")
    DatasetSplitter.impute(df_train, df_test, "hadap", "mode")
    DatasetSplitter.impute(df_train, df_test, "sertifikat", "mode")
    DatasetSplitter.impute(df_train, df_test, "sumber_air", "mode")
    DatasetSplitter.impute(df_train, df_test, "pemandangan", "mode")
    DatasetSplitter.impute(df_train, df_test, "tipe_properti", "mode")
    DatasetSplitter.impute(df_train, df_test, "konsep_dan_gaya_rumah", "mode")

    DatasetSplitter.impute(df_train, df_test, "kondisi_properti", "constant",
                           "unfurnished")
    DatasetSplitter.impute(df_train, df_test, "kondisi_perabotan", "constant",
                           "unfurnished")

    # fill the rest with zeros
    for column in self.df.columns:
      if column.startswith('facility_') or column.startswith(
          'tag_') or column.startswith('floor_mat_') or column.startswith(
              'house_mat_'):
        DatasetSplitter.impute(df_train, df_test, column, "constant", 0)

    # subset
    if self.subset_mode == "pvalue" or self.subset_mode == "r":
      # columns to keep
      keep_cols = ["price"]

      # calculate correlations
      for column in df_train.columns:
        # skip price column
        if column == "price":
          continue

        # check if column is numeric
        if not is_numeric_dtype(df_train[column]):
          # add to keep_cols
          keep_cols.append(column)
          continue

        # calculate correlation
        if DatasetSplitter.should_use_point_biser_corr(column):
          corr = stats.pointbiserialr(df_train[column], df_train["price"])
        else:
          corr = stats.pearsonr(df_train[column], df_train["price"])

        # should we add this to keep_cols?
        if self.subset_mode == "pvalue" and corr[1] < 0.05:
          # add to keep_cols
          keep_cols.append(column)
        elif self.subset_mode == "r" and abs(corr[0]) > 0.1:
          # add to keep_cols
          keep_cols.append(column)

      # subset
      df_train = df_train[keep_cols]
      df_test = df_test[keep_cols]

    elif self.subset_mode == "manual":
      df_train = df_train[MANUAL_SUBSET_COLS]
      df_test = df_test[MANUAL_SUBSET_COLS]

    # print info
    print(df_train.info())

    # save data
    df_train.to_parquet(os.path.join(self.output_dir, f"{self.subset_mode}_train.parquet"))
    df_test.to_parquet(os.path.join(self.output_dir, f"{self.subset_mode}_test.parquet"))


if __name__ == "__main__":
  # setup command-line arguments
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--dataset",
      help="Input dataset from L3",
      default="./dataset/curated/marts_ml_train_sel_all.parquet")
  parser.add_argument(
      "--output-dir",
      help="Output directory for training metrics",
      default="./ml_models/baseline")
  parser.add_argument(
      "--test-percentage",
      help="Amount of data to use for testing in percent",
      default=TEST_PERCENTAGE)
  parser.add_argument(
      "--subset", help="Subset method", default="all", choices=["all", "pvalue", "r", "manual"])
  parser.add_argument(
      "--random_state", help="Random state", default=RANDOM_STATE)

  args = parser.parse_args()

  # train model
  processor = DatasetSplitter(args.dataset, args.output_dir, args.test_percentage,
                            args.subset, args.random_state)
  processor.run()
