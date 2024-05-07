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

import tensorflow as tf

from catboost import CatBoostRegressor, Pool

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import cross_validate, KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler, FunctionTransformer
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from sklearn.svm import SVR, LinearSVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.neighbors import KNeighborsRegressor
from sklearn.linear_model import LinearRegression, Lasso, Ridge, BayesianRidge
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor

from utils.ml_base import TrainerMixin

# define configuration
TRAIN_BASELINE_CV_SPLIT = 10
TRAIN_BASELINE_BATCH_SIZE = 256
TRAIN_BASELINE_N_JOBS = 4
TRAIN_BASELINE_RANDOM_STATE = 21
TRAIN_BASELINE_CROSS_VAL_SCORING = [
    "r2", "neg_mean_squared_error", "neg_mean_absolute_error",
    "neg_mean_absolute_percentage_error"
]


class TrainRandomForest(TrainerMixin):

  def __init__(self,
               dataset_path: str,
               output_dir: str,
               batch_size=TRAIN_BASELINE_BATCH_SIZE,
               cv_split=TRAIN_BASELINE_CV_SPLIT,
               n_jobs: int = TRAIN_BASELINE_N_JOBS,
               random_state=TRAIN_BASELINE_RANDOM_STATE,
               verbose=1,
               sparse_fix=False,
               run_name="all") -> None:
    super().__init__()

    self.dataset_path = dataset_path
    self.output_dir = output_dir
    self.batch_size = batch_size
    self.cv_split = cv_split
    self.n_jobs = n_jobs
    self.random_state = random_state
    self.verbose = verbose
    self.sparse_fix = sparse_fix
    self.run_name = run_name

    # to hold CV results
    self.cv_results = []

  def get_tensorboard_logdir(self, prefix: str, fold_i: int) -> str:
    return os.path.join(self.output_dir, f"{self.run_name}_logs",
                        f"{prefix}_{fold_i + 1}")

  def load_data(self):
    # load dataset
    df = pd.read_parquet(self.dataset_path)
    print(df.info())

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

    # create X and y
    self.X = df.drop(columns=["price"])
    self.y = df["price"].values

    # dataset for TensorFlow
    self.X_trans = self.compose_transformers.fit_transform(self.X)

  # ---- SKLEARN METHODS

  def cross_validate_ex(self, model, X, y, category, name):
    # create cross-validation param
    cv = KFold(
        n_splits=self.cv_split, shuffle=True, random_state=self.random_state)

    # cross-validate
    scores = cross_validate(
        model,
        X,
        y,
        cv=cv,
        scoring=TRAIN_BASELINE_CROSS_VAL_SCORING,
        n_jobs=self.n_jobs,
        verbose=self.verbose)

    # change into record-wise
    score_records = []
    for i in range(len(scores["fit_time"])):
      score_records.append({
          "fit_time": scores["fit_time"][i],
          "score_time": scores["score_time"][i],
          "r2": scores["test_r2"][i],
          "mse": -scores["test_neg_mean_squared_error"][i],
          "mae": -scores["test_neg_mean_absolute_error"][i],
          "mape": -scores["test_neg_mean_absolute_percentage_error"][i],
          "category": category,
          "name": name,
          "fold": i,
          "run_name": self.run_name,
          "features_count": X.shape[1],
      })

    return score_records

  def train_sklearn(self):
    # define models
    models = [
        ("Linear", "LinearRegression", LinearRegression()),
        ("Linear", "Lasso", Lasso()),
        ("Linear", "Ridge", Ridge()),
        ("Linear", "BayesianRidge", BayesianRidge(verbose=self.verbose)),
        ("Tree", "DecisionTreeRegressor", DecisionTreeRegressor()),
        ("KNN", "KNeighborsRegressor", KNeighborsRegressor()),
        ("SVM", "SVR", SVR(verbose=self.verbose)),
        ("SVM", "LinearSVR", LinearSVR(dual="auto", verbose=self.verbose)),
        ("Neural Network", "MLPRegressor", MLPRegressor(verbose=self.verbose)),
        ("Ensemble", "RandomForestRegressor",
         RandomForestRegressor(verbose=self.verbose)),
        ("Ensemble", "GradientBoostingRegressor",
         GradientBoostingRegressor(verbose=self.verbose)),
    ]

    # evaluate each model
    for category, name, model in models:
      self.logger.info(f"Evaluating {category}/{name} model")

      # create classifier pipeline
      clf = Pipeline(steps=[
          ("preprocessor", self.compose_transformers),
          ("regressor", model),
      ] if not self.sparse_fix else [
          ("preprocessor", self.compose_transformers),
          ("to_dense",
           FunctionTransformer(lambda x: x.toarray(), accept_sparse=True)),
          ("regressor", model),
      ])

      # run cross validation
      self.cv_results.extend(
          self.cross_validate_ex(clf, self.X, self.y, category, name))

  # ---- CATBOOST METHODS

  def train_catboost(self):
    # run training
    cv = KFold(
        n_splits=self.cv_split, shuffle=True, random_state=self.random_state)

    for fold_i, (train_idx, test_idx) in enumerate(cv.split(self.X, self.y)):
      self.logger.debug(f"Training fold {fold_i + 1}")

      # split data
      X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
      y_train, y_test = self.y[train_idx], self.y[test_idx]

      # create pool
      train_pool = Pool(data=X_train, label=y_train, cat_features=self.cat_cols)
      test_pool = Pool(data=X_test, label=y_test, cat_features=self.cat_cols)

      # create model
      log_dir = self.get_tensorboard_logdir("catboost", fold_i)
      model = CatBoostRegressor(
          verbose=self.verbose,
          random_seed=self.random_state,
          task_type="CPU",
          train_dir=log_dir)

      # fit model
      fit_time_start = time.time()
      model.fit(train_pool, eval_set=test_pool, verbose=self.verbose)
      fit_time_end = time.time()

      # run predictions
      score_time_start = time.time()
      y_pred = model.predict(test_pool)
      score_time_end = time.time()

      # store metrics
      self.cv_results.append({
          "fit_time": fit_time_end - fit_time_start,
          "score_time": score_time_end - score_time_start,
          "r2": r2_score(y_test, y_pred),
          "mse": mean_squared_error(y_test, y_pred),
          "mae": mean_absolute_error(y_test, y_pred),
          "mape": mean_absolute_percentage_error(y_test, y_pred),
          "category": "CatBoost",
          "name": f"CatBoostRegressor",
          "fold": fold_i,
          "run_name": self.run_name,
          "features_count": self.X.shape[1],
      })

      del model
      del train_pool
      del test_pool
      del X_train
      del X_test
      del y_train
      del y_test

  # ---- TENSORFLOW METHODS

  def construct_tf_dataset(self, X: np.ndarray,
                           y: np.ndarray) -> tf.data.Dataset:
    # create dataset
    ds_labels = tf.data.Dataset.from_tensor_slices(y)
    ds_features = tf.data.Dataset.from_tensor_slices(X)

    return tf.data.Dataset.zip((ds_features, ds_labels))\
     .batch(self.batch_size) \
     .cache() \
     .prefetch(tf.data.AUTOTUNE)

  def create_tf_model(self, ds: tf.data.Dataset) -> tf.keras.Model:
    # create model
    inputs = tf.keras.layers.Input((ds.element_spec[0].shape[1],))
    x = tf.keras.layers.Dense(512, activation="gelu")(inputs)
    x = tf.keras.layers.Dense(256, activation="gelu")(x)
    x = tf.keras.layers.Dense(128, activation="gelu")(x)
    x = tf.keras.layers.Dense(64, activation="gelu")(x)
    outputs = tf.keras.layers.Dense(1, name="price")(x)

    # create model
    model = tf.keras.Model(inputs=inputs, outputs=outputs)
    return model

  def train_tensorflow(self):
    # run training
    cv = KFold(
        n_splits=self.cv_split, shuffle=True, random_state=self.random_state)

    for fold_i, (train_idx,
                 test_idx) in enumerate(cv.split(self.X_trans, self.y)):
      self.logger.debug(f"{fold_i + 1}")

      # split data
      X_train, X_test = self.X_trans[train_idx], self.X_trans[test_idx]
      y_train, y_test = self.y[train_idx], self.y[test_idx]

      # create dataset
      train_ds = self.construct_tf_dataset(X_train, y_train)
      test_ds = self.construct_tf_dataset(X_test, y_test)

      # create model
      model = self.create_tf_model(train_ds)

      # compile model
      model.compile(
          optimizer="adam", loss="mean_squared_error", metrics=["mae", "mse"])

      # create early stopping callback
      early_stopping_callback = tf.keras.callbacks.EarlyStopping(
          monitor='val_loss', patience=3)

      # create tensorboard callback
      log_dir = self.get_tensorboard_logdir("tf", fold_i)
      tensorboard_callback = tf.keras.callbacks.TensorBoard(
          log_dir=log_dir, histogram_freq=1)

      # train model
      fit_time_start = time.time()
      model.fit(
          train_ds,
          epochs=200,
          validation_data=test_ds,
          verbose=self.verbose,
          callbacks=[early_stopping_callback, tensorboard_callback])
      fit_time_end = time.time()

      # run predictions
      score_time_start = time.time()
      y_pred = model.predict(test_ds).reshape(-1)
      score_time_end = time.time()

      # store metrics
      self.cv_results.append({
          "fit_time": fit_time_end - fit_time_start,
          "score_time": score_time_end - score_time_start,
          "r2": r2_score(y_test, y_pred),
          "mse": mean_squared_error(y_test, y_pred),
          "mae": mean_absolute_error(y_test, y_pred),
          "mape": mean_absolute_percentage_error(y_test, y_pred),
          "category": "TensorFlow",
          "name": "DNNRegressor",
          "fold": fold_i,
          "run_name": self.run_name,
          "features_count": self.X.shape[1],
      })

      del model
      del train_ds
      del test_ds
      del X_train
      del X_test
      del y_train
      del y_test

  # --- MAIN METHODS

  def train(self):
    # print tensorflow devices
    for device in tf.config.list_physical_devices():
      self.logger.info("TensorFlow device: %s (%s)", device.name,
                       device.device_type)

    # create tensorboard logs directory
    os.makedirs(
        os.path.join(self.output_dir, f"{self.run_name}_logs"), exist_ok=True)

    # run training
    self.logger.info("Training sklearn models")
    self.train_sklearn()

    self.logger.info("Training CatBoost models")
    self.train_catboost()

    self.logger.info("Training TensorFlow models")
    self.train_tensorflow()

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
  parser.add_argument(
      "--sparse-fix", help="Sparse fix", default=False, action="store_true")
  parser.add_argument("--run-name", help="Run name", default="all")

  args = parser.parse_args()

  # train model
  trainer = TrainRandomForest(args.dataset, args.output_dir, args.batch_size,
                              args.cv_split, args.n_jobs, args.random_state,
                              args.verbose, args.sparse_fix, args.run_name)
  trainer.run()
