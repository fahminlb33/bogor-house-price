import os
import argparse
import datetime
from pprint import pprint


# patch sklearn with Intel Extension for Scikit-learn
from sklearnex import patch_sklearn

patch_sklearn()


import yaml
import joblib
import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from utils.ml_base import TrainerMixin


class TrainPipeline(TrainerMixin):

  def __init__(self,
               train_dataset: str,
               test_dataset: str,
               params_path: str,
               output_path: str,
               algorithm: str = "catboost",
               random_state: int = 21):
    super().__init__()

    self.train_dataset_path = train_dataset
    self.test_dataset_path = test_dataset
    self.params_path = params_path
    self.output_path = output_path
    self.algorithm = algorithm
    self.random_state = random_state

  def load_data(self):
    # load dataset
    self.df_train = pd.read_parquet(self.train_dataset_path)
    self.df_test = pd.read_parquet(self.test_dataset_path)

    # create hyperparameters
    self.params = yaml.safe_load(open(self.params_path, "r"))

  def get_data_sklearn(self):
    # identify columns
    self.multihot_cols = []
    self.multihot_cols.extend(
        [col for col in self.df_train.columns if col.startswith("floor_mat_")])
    self.multihot_cols.extend(
        [col for col in self.df_train.columns if col.startswith("house_mat_")])
    self.multihot_cols.extend(
        [col for col in self.df_train.columns if col.startswith("facility_")])
    self.multihot_cols.extend(
        [col for col in self.df_train.columns if col.startswith("tag_")])

    # extra features not included in tags_
    extra_tags = ["ruang_tamu", "ruang_makan", "terjangkau_internet", "hook"]

    for tag in extra_tags:
      if tag in self.df_train.columns:
        self.multihot_cols.append(tag)

    # categorical columns
    self.cat_cols = [
        col for col in self.df_train.select_dtypes(include=["object"]).columns
    ]

    # numerical columns
    self.num_cols = list(
        set(self.df_train.columns) -
        set(self.multihot_cols + self.cat_cols + ["price"]))

    # create preprocessing pipeline
    catl_encoder = Pipeline(steps=[
        ("encoder", OneHotEncoder(handle_unknown="ignore")),
    ])

    num_encoder = Pipeline(steps=[
        ("scaler", MinMaxScaler()),
    ])

    self.compose_transformers = ColumnTransformer(transformers=[
        ("passthrough", "passthrough", self.multihot_cols),
        ("catergorical_encoder", catl_encoder, self.cat_cols),
        ("numerical_encoder", num_encoder, self.num_cols),
    ])

    # preprocess using sklearn pipeline
    X_train = self.df_train.drop(columns=["price"])
    X_test = self.df_test.drop(columns=["price"])

    y_train = self.df_train["price"]
    y_test = self.df_test["price"]

    return (X_train, y_train), (X_test, y_test)
  
  def get_data_catboost(self):
    # split data
    X_train, X_test = self.df_train.drop(columns=["price"]), self.df_test.drop(columns=["price"])
    y_train, y_test = self.df_train["price"], self.df_test["price"]

    # categorical columns
    self.cat_cols = [
        col for col in self.df_train.select_dtypes(include=["object"]).columns
    ]

    # create pool
    train_pool = Pool(data=X_train, label=y_train, cat_features=self.cat_cols)
    test_pool = Pool(data=X_test, label=y_test, cat_features=self.cat_cols)

    return (train_pool, test_pool)
  
  def save_importances(self, importance_series: pd.Series):
      # save importance
      importance_series.to_csv(os.path.join(self.output_path, f"{self.algorithm}_importance.csv"))

      # plot importance
      fig, ax = plt.subplots(figsize=(20, 10))
      importance_series.plot.bar(ax=ax)
      ax.set_title("Feature importances using Gini ratio")
      ax.set_ylabel("Mean decrease in impurity")
      fig.tight_layout()

      # save plot
      fig.savefig(os.path.join(self.output_path, f"{self.algorithm}_importance.png"))
  
  def train_catboost(self):
    # get dataset
    (train_pool, test_pool) = self.get_data_catboost()

    # create model
    model = CatBoostRegressor(**self.params)

    # fit model
    self.logger.info("Training model...")
    model.fit(train_pool)

    # evaluate model
    self.logger.info("Evaluating model...")
    y_pred = model.predict(test_pool)
    y_true = test_pool.get_label()

    pprint({
      "r2": r2_score(y_true, y_pred),
      "mse": mean_squared_error(y_true, y_pred),
      "mae": mean_absolute_error(y_true, y_pred),
      "mape": mean_absolute_percentage_error(y_true, y_pred),
    })

    # save model
    self.logger.info("Saving model...")
    model.save_model(os.path.join(self.output_path, f"{self.algorithm}_model.cbm"))

    # model importance
    self.logger.info("Saving model importance...")
    importance_cols = list(self.df_train.columns)
    importance_cols.remove("price")
    
    forest_importances = pd.Series(model.feature_importances_, index=importance_cols) \
        .sort_values(ascending=False)
    self.save_importances(forest_importances)

  def train_random_forest(self):
    # get dataset
    (X_train, y_train), (X_test, y_true) = self.get_data_sklearn()

    # create model
    model =Pipeline(steps=[
      ("preprocessor", self.compose_transformers),
      ("regressor",  RandomForestRegressor(**self.params))
    ]) 

    # fit model
    self.logger.info("Training model...")
    model.fit(X_train, y_train)

    # evaluate model
    self.logger.info("Evaluating model...")
    y_pred = model.predict(X_test)

    pprint({
      "r2": r2_score(y_true, y_pred),
      "mse": mean_squared_error(y_true, y_pred),
      "mae": mean_absolute_error(y_true, y_pred),
      "mape": mean_absolute_percentage_error(y_true, y_pred),
    })

    # save model
    self.logger.info("Saving model...")
    joblib.dump(model, os.path.join(self.output_path, f"{self.algorithm}_model.joblib"))

    # --- model importance
    # load model
    forest = model.named_steps["regressor"]

    # get importance and calculate standard deviation
    importances = forest.feature_importances_

    # normalize feature names
    feature_names = self.compose_transformers.get_feature_names_out()
    feature_names = [
        name.replace("catergorical_encoder__", "") for name in feature_names
    ]
    feature_names = [
        name.replace("numerical_encoder__", "") for name in feature_names
    ]
    feature_names = [
        name.replace("passthrough__", "") for name in feature_names
    ]

    # create importance series
    forest_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)
    
    self.save_importances(forest_importances)

  def train(self):
    # create model
    if self.algorithm == "catboost":
      self.train_catboost()
    elif self.algorithm == "random_forest":
      self.train_random_forest()
    else:
      raise ValueError(f"Invalid algorithm: {self.algorithm}")

  def run(self):
    self.logger.info("Creating output directory...")
    os.makedirs(self.output_path, exist_ok=True)

    super().run()


if __name__ == "__main__":
  # set matplotlib backend
  matplotlib.use("Agg")

  # today
  today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

  # setup command-line arguments
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--train-dataset",
      help="Input train dataset from L3",
      default="./dataset/curated/marts_ml_train_sel_manual.parquet")
  parser.add_argument(
      "--test-dataset",
      help="Input test dataset from L3",
      default="./dataset/curated/marts_ml_train_sel_manual.parquet")
  parser.add_argument(
      "--params-path",
      help="Input parameters for training",
      default=f"./ml_models/params.yaml")
  parser.add_argument(
      "--output-path",
      help="Output path for model and metrics",
      default=f"./ml_models/catboost-{today}")
  parser.add_argument("--algorithm", help="Algorithm", default="catboost", choices=["catboost", "random_forest"])
  parser.add_argument("--random_state", help="Random state", default=21)
  

  args = parser.parse_args()

  # train model
  trainer = TrainPipeline(args.train_dataset, args.test_dataset, args.params_path, args.output_path, args.algorithm, args.random_state)
  trainer.run()
