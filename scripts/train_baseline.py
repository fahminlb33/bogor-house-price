import os
import time
import argparse
import warnings

warnings.filterwarnings("ignore")
os.environ['PYTHONWARNINGS'] = 'ignore'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
os.environ["CUDA_VISIBLE_DEVICES"] = "-1"

# patch sklearn with Intel Extension for Scikit-learn
from sklearnex import patch_sklearn

patch_sklearn()

import numpy as np
import pandas as pd

from catboost import CatBoostRegressor, Pool

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from utils.ml_base import TrainerMixin
from utils.ml_algorithms import CustomTensorFlowRegressor

# define configuration
TRAIN_BASELINE_CV_SPLIT = 10
TRAIN_BASELINE_BATCH_SIZE = 256
TRAIN_BASELINE_N_JOBS = 4
TRAIN_BASELINE_RANDOM_STATE = 21


class TrainRandomForest(TrainerMixin):

  def __init__(self,
               dataset_path: str,
               output_dir: str,
               batch_size=TRAIN_BASELINE_BATCH_SIZE,
               cv_split=TRAIN_BASELINE_CV_SPLIT,
               n_jobs: int = TRAIN_BASELINE_N_JOBS,
               random_state=TRAIN_BASELINE_RANDOM_STATE,
               run_name="all") -> None:
    super().__init__()

    self.dataset_path = dataset_path
    self.output_dir = output_dir
    self.batch_size = batch_size
    self.cv_split = cv_split
    self.n_jobs = n_jobs
    self.random_state = random_state
    self.run_name = run_name

    # to hold CV results
    self.cv_results = []

  def load_data(self):
    # load dataset
    df = pd.read_parquet(self.dataset_path)

    # create X and y
    self.X = df.drop(columns=["price"])
    self.y = df["price"]

    # identify columns
    self.multihot_cols = []
    self.multihot_cols.extend(
        [col for col in df.columns if col.startswith("floor_mat_")])
    self.multihot_cols.extend(
        [col for col in df.columns if col.startswith("house_mat_")])
    self.multihot_cols.extend(
        [col for col in df.columns if col.startswith("facility_")])
    self.multihot_cols.extend(
        [col for col in df.columns if col.startswith("tag_")])

    # extra features not included in tags_
    extra_tags = ["ruang_tamu", "ruang_makan", "terjangkau_internet", "hook"]

    for tag in extra_tags:
      if tag in df.columns:
        self.multihot_cols.append(tag)

    # categorical columns
    self.cat_cols = [
        col for col in df.select_dtypes(include=["object"]).columns
    ]

    # numerical columns
    self.num_cols = list(
        set(df.columns) - set(self.multihot_cols + self.cat_cols + ["price"]))

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

  def get_tensorboard_logdir(self, prefix: str, fold_i: int) -> str:
    return os.path.join(self.output_dir, f"{self.run_name}_logs",
                        f"{prefix}_{fold_i + 1}")

  def get_data(self, train_indices, test_indices, mode: str):
    # split data
    X_train, X_test = self.X.iloc[train_indices], self.X.iloc[test_indices]
    y_train, y_test = self.y.iloc[train_indices], self.y.iloc[test_indices]

    # return correct dataset
    if mode == "CatBoostRegressor":
      cat_cols = list(X_train.select_dtypes(include=["object"]).columns)
      pool_train = Pool(data=X_train, label=y_train, cat_features=cat_cols)
      pool_test = Pool(data=X_test, label=y_test, cat_features=cat_cols)

      return (pool_train, pool_test)

    # preprocess using sklearn pipeline
    X_train = self.compose_transformers.fit_transform(X_train)
    X_test = self.compose_transformers.transform(X_test)

    return (X_train, y_train), (X_test, y_test)

  def cross_validate_ex(self, category, name):
    # create K-fold
    cv = KFold(
        n_splits=self.cv_split, shuffle=True, random_state=self.random_state)

    # run training
    for fold_i, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y)):
      self.logger.debug(f"{fold_i + 1}")

      # get the dataset
      (ds_train, ds_test) = self.get_data(train_idx, test_idx, name)

      # get the tensorboard logdir
      log_dir = self.get_tensorboard_logdir(name, fold_i)

      # create model
      model = None
      if name == "LinearRegression":
        model = LinearRegression()
      elif name == "Lasso":
        model = Lasso(random_state=self.random_state)
      elif name == "Ridge":
        model = Ridge(random_state=self.random_state)
      elif name == "SVR":
        model = SVR()
      elif name == "LinearSVR":
        model = LinearSVR(random_state=self.random_state)
      elif name == "DecisionTreeRegressor":
        model = DecisionTreeRegressor(random_state=self.random_state, max_depth=110)
      elif name == "RandomForestRegressor":
        model = RandomForestRegressor(random_state=self.random_state, max_depth=110)
      elif name == "GradientBoostingRegressor":
        model = GradientBoostingRegressor(random_state=self.random_state)
      elif name == "CatBoostRegressor":
        model = CatBoostRegressor(
            random_seed=self.random_state, task_type="CPU", train_dir=log_dir)
      elif name == "TensorFlowV1Regressor":
        model = CustomTensorFlowRegressor(
            model_config="v1", tensorboard_dir=log_dir)
      elif name == "TensorFlowV2Regressor":
        model = CustomTensorFlowRegressor(
            model_config="v2", tensorboard_dir=log_dir)
      elif name == "TensorFlowV3Regressor":
        model = CustomTensorFlowRegressor(
            model_config="v3", tensorboard_dir=log_dir)

      # train model
      fit_time_start = time.time()

      if name == "CatBoostRegressor":
        model.fit(ds_train, eval_set=ds_test, verbose=0)
      else:
        model.fit(ds_train[0], ds_train[1])

      fit_time_end = time.time()

      # run predictions
      score_time_start = time.time()

      if name == "CatBoostRegressor":
        y_pred = model.predict(ds_test).reshape(-1)
      else:
        y_pred = model.predict(ds_test[0])

      score_time_end = time.time()

      # get the true predictions
      y_true = ds_test[1] if name != "CatBoostRegressor" else ds_test.get_label()

      # store metrics
      self.cv_results.append({
          "fit_time": fit_time_end - fit_time_start,
          "score_time": score_time_end - score_time_start,
          "r2": r2_score(y_true, y_pred),
          "mse": mean_squared_error(y_true, y_pred),
          "mae": mean_absolute_error(y_true, y_pred),
          "mape": mean_absolute_percentage_error(y_true, y_pred),
          "category": category,
          "name": name,
          "fold": fold_i,
          "run_name": self.run_name,
          "features_count": self.X.shape[1],
      })

      del ds_train
      del ds_test

  # --- MAIN METHODS

  def train(self):
    # create tensorboard logs directory
    os.makedirs(
        os.path.join(self.output_dir, f"{self.run_name}_logs"), exist_ok=True)
    
    # define models
    models = [
        ("Linear", "LinearRegression"),
        ("Linear", "Lasso"),
        ("Linear", "Ridge"),
        ("SVM", "SVR"),
        ("SVM", "LinearSVR"),
        ("TreeEnsemble", "DecisionTreeRegressor"),
        ("TreeEnsemble", "RandomForestRegressor"),
        ("TreeEnsemble", "GradientBoostingRegressor"),
        ("TreeEnsemble", "CatBoostRegressor"),
        ("DeepLearning", "TensorFlowV1Regressor"),
        ("DeepLearning", "TensorFlowV2Regressor"),
        ("DeepLearning", "TensorFlowV3Regressor"),
    ]

    # evaluate each model
    for category, name in models:
      self.logger.info(f"Evaluating {category}/{name} model")

      # run cross validation
      self.cross_validate_ex(category, name)

    # save metrics
    df_scores = pd.DataFrame(self.cv_results)
    df_scores.to_csv(
        f"{self.output_dir}/{self.run_name}-raw_scores.csv", index=False)

    # summarize metrics
    rdf = df_scores.pivot_table(index=["category", "name"],
           values=["fit_time", "score_time", "r2", "mse", "mae", "mape"]) \
        .reset_index() \
        .sort_values(by="mse", ascending=True)

    rdf.to_csv(f"{self.output_dir}/{self.run_name}-summary.csv", index=False)


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
      "--batch-size",
      help="Batch size for TensorFlow",
      default=TRAIN_BASELINE_BATCH_SIZE)
  parser.add_argument(
      "--cv-split",
      help="Number of cross-validation splits",
      default=TRAIN_BASELINE_CV_SPLIT)
  parser.add_argument("--n_jobs", help="N jobs", default=TRAIN_BASELINE_N_JOBS)
  parser.add_argument(
      "--random_state",
      help="Random state",
      default=TRAIN_BASELINE_RANDOM_STATE)
  parser.add_argument("--verbose", help="Verbose", default=1, type=int)
  parser.add_argument("--run-name", help="Run name", default="all")

  args = parser.parse_args()

  # train model
  trainer = TrainRandomForest(args.dataset, args.output_dir, args.batch_size,
                              args.cv_split, args.n_jobs, args.random_state,
                              args.run_name)
  trainer.run()
