import time
import argparse

import optuna
import mlflow

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.figure

import ydf
import xgboost as xgb
import catboost as cb
import lightgbm as lgb

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)


def symmetric_mean_absolute_percentage_error(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred) / ((np.abs(y_true) + np.abs(y_pred)) / 2))


def plot_predictions(y_test, y_pred):
    fig = matplotlib.figure.Figure(figsize=(10, 5))
    ax = fig.subplots(1, 2)

    ax[0].scatter(y_test, y_pred, color="blue", alpha=0.5)
    ax[0].set_xlabel("Actual")
    ax[0].set_ylabel("Predicted")
    ax[0].set_title("Predictions")

    ax[1].scatter(y_test, y_test - y_pred, color="blue", alpha=0.5)
    ax[1].axhline(y=0, color="r", linestyle="-")
    ax[1].set_xlabel("Actual")
    ax[1].set_ylabel("Residuals")
    ax[1].set_title("Residuals")

    fig.tight_layout()

    return fig


def plot_distributions(y_test, y_pred):
    fig = matplotlib.figure.Figure(figsize=(10, 5))
    ax1, ax2 = fig.subplots(1, 2)

    ax1.hist(y_test, bins=50)
    ax1.set_title("Actual")

    ax2.hist(y_pred, bins=50)
    ax2.set_title("Predicted")

    fig.tight_layout()

    return fig


def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


def evaluate_model(y_true, y_pred):
    return {
        "mean": np.mean(y_pred),
        "std": np.std(y_pred),
        "var": np.var(y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
        "r2": r2_score(y_true, y_pred),
    }


class OptunaObjective:
    def __init__(self, algorithm: str, dataset_path: str):
        self.algorithm = algorithm
        self.dataset_path = dataset_path

    def load_data(self):
        # load dataset
        df = pd.read_parquet(self.dataset_path)

        # --- preprocessing
        multichoice_cols = ["house_material", "floor_material", "tags", "facilities"]
        drop_col = [
            "id",
            "installment",
            "city",
            "description",
            "url",
            "tipe_properti",
            "main_image_url",
            "hadap",
        ] + multichoice_cols

        # copy data
        df_data = df.drop(columns=drop_col).copy()

        # IQR-based outlier removal
        q1 = df_data["price"].quantile(0.25)
        q3 = df_data["price"].quantile(0.75)
        iqr = q3 - q1

        df_data = df_data[
            (df_data["price"] > (q1 - 1.5 * iqr))
            & (df_data["price"] < (q3 + 1.5 * iqr))
        ]

        # rooms-based outlier removal
        df_data = df_data[df_data["kamar_tidur"] <= 5]

        # log-tranform price
        df_data["price"] = np.log(df_data["price"])

        # convert categorical columns
        cat_cols = df_data.select_dtypes("object").columns.tolist()
        df_data = df_data.fillna(value={k: "" for k in cat_cols})
        df_data[cat_cols] = df_data[cat_cols].astype("category")

        # split data
        train_df, test_df = train_test_split(df_data, test_size=0.33, random_state=22)

        self.cat_cols = cat_cols
        self.train_df = train_df
        self.test_df = test_df

    def generate_params(self, trial: optuna.Trial):
        monotone_up = ["luas_tanah", "luas_bangunan"]

        input_features = self.train_df.columns.tolist()
        input_features.remove("price")

        if self.algorithm == "ydf":
            # https://ydf.readthedocs.io/en/latest/tutorial/tuning/#local-tuning-with-manually-set-hyper-parameters
            return {
                # common hyperparameters
                "num_trees": trial.suggest_int("num_trees", 100, 500),
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                # specific hyperparameters
                "subsample": trial.suggest_float("subsample", 0.1, 1.0),
                "l1_regularization": trial.suggest_float("l1_regularization", 0, 100),
                "l2_regularization": trial.suggest_float("l2_regularization", 0, 100),
                # fixed hyperparameters
                "random_seed": 123456,
                "use_hessian_gain": True,
                "include_all_columns": True,
                "features": [
                    ydf.Feature(x, ydf.Semantic.NUMERICAL, monotonic=1)
                    for x in monotone_up
                ],
            }

        if self.algorithm == "lightgbm":
            # https://www.kaggle.com/code/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761
            # https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
            return {
                # common hyperparameters
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "num_iterations": trial.suggest_int("num_iterations", 100, 500),
                "objective": trial.suggest_categorical(
                    "objective", ["regression", "huber"]
                ),
                # specific hyperparameters
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 100),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 100),
                "num_leaves": trial.suggest_int("num_leaves", 6, 50),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
                # fixed hyperparameters
                "seed": 22,
                "metric": "rmse",
                "verbosity": -1,
                "monotone_constraints": [
                    1 if x in monotone_up else 0 for x in input_features
                ],
            }

        if self.algorithm == "catboost":
            # https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_pruning.py
            # https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_simple.py
            # https://github.com/catboost/tutorials/blob/master/hyperparameters_tuning/hyperparameters_tuning_using_optuna_and_hyperopt.ipynb
            params = {
                # common hyperparameters
                "depth": trial.suggest_int("depth", 3, 10),
                "iterations": trial.suggest_int("iterations", 100, 500),
                "objective": trial.suggest_categorical("objective", ["RMSE", "Huber"]),
                # specific hyperparameters
                "boosting_type": trial.suggest_categorical(
                    "boosting_type", ["Ordered", "Plain"]
                ),
                "bootstrap_type": trial.suggest_categorical(
                    "bootstrap_type", ["Bayesian", "Bernoulli", "MVS"]
                ),
                "colsample_bylevel": trial.suggest_float(
                    "colsample_bylevel", 0.01, 0.1, log=True
                ),
                # fixed hyperparameters
                "verbose": 0,
                "random_seed": 21,
                "eval_metric": "RMSE",
                "monotone_constraints": {k: 1 for k in monotone_up},
            }

            if params["bootstrap_type"] == "Bayesian":
                params["bagging_temperature"] = trial.suggest_float(
                    "bagging_temperature", 0, 10
                )

            elif params["bootstrap_type"] == "Bernoulli":
                params["subsample"] = trial.suggest_float("subsample", 0.1, 1, log=True)

            return params

        if self.algorithm == "xgboost":
            # https://www.kaggle.com/code/prashant111/a-guide-on-xgboost-hyperparameters-tuning#4.1-What-is-HYPEROPT-
            return {
                # common hyperparameters
                "objective": "reg:squaredlogerror",
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                # specific hyperparameters
                "alpha": trial.suggest_float("alpha", 0, 40),
                "gamma": trial.suggest_float("gamma", 1, 9),
                "lambda": trial.suggest_float("lambda", 0, 1),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1),
                # fixed hyperparameters
                "eval_metric": "rmse",
                "random_seed": 0,
                "enable_categorical": True,
                "monotone_constraints": {k: 1 for k in monotone_up},
            }

        raise ValueError("Invalid model name")

    def train_model(self, params: dict, train_data: pd.DataFrame):
        X_train, y_train = train_data.drop(columns=["price"]), train_data["price"]

        if self.algorithm == "ydf":
            return ydf.GradientBoostedTreesLearner(
                label="price", task=ydf.Task.REGRESSION, **params
            ).train(train_data)

        if self.algorithm == "lightgbm":
            train_ds = lgb.Dataset(X_train, y_train, categorical_feature=self.cat_cols)
            return lgb.train(params, train_set=train_ds)

        if self.algorithm == "catboost":
            return cb.CatBoostRegressor(**params).fit(
                X_train, y_train, cat_features=self.cat_cols
            )

        if self.algorithm == "xgboost":
            return xgb.XGBRegressor(**params).fit(X_train, y_train)

        raise ValueError("Invalid model name")

    def __call__(self, trial: optuna.Trial):
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            # create hyperparameters
            params = self.generate_params(trial)

            # create scores
            val_scores = []

            # perform cross validation
            cv = KFold(n_splits=10, shuffle=True, random_state=21)
            for fold_i, (train_idx, test_idx) in enumerate(cv.split(self.train_df)):
                print(f"{fold_i + 1}...", end="")

                # split data
                train_data = self.train_df.iloc[train_idx, :]
                val_data = self.train_df.iloc[test_idx, :]

                # create model
                model = self.train_model(params, train_data)

                # run prediction
                X_val, y_val = (
                    val_data.drop(columns=["price"]),
                    np.exp(val_data["price"]),
                )
                y_val_pred = np.exp(model.predict(X_val))

                # collect metrics
                val_scores.append(evaluate_model(y_val, y_val_pred))

            print("")

            # log params
            if "features" in params:
                params.pop("features")
            if "monotone_constraints" in params:
                params.pop("monotone_constraints")

            mlflow.log_params(params)

            # log metrics
            for name in val_scores[0].keys():
                mlflow.log_metric(
                    f"val_{name}", np.mean([score[name] for score in val_scores])
                )

            # run prediction on test data
            X_test, y_test = (
                self.test_df.drop(columns=["price"]),
                np.exp(self.test_df["price"]),
            )
            y_test_pred = np.exp(model.predict(X_test))

            test_score = evaluate_model(y_test, y_test_pred)

            # log test metrics
            for k, v in test_score.items():
                mlflow.log_metric(f"test_{k}", v)

            mlflow.log_figure(plot_predictions(y_test, y_test_pred), "predictions.png")
            mlflow.log_figure(
                plot_distributions(y_test, y_test_pred), "distributions.png"
            )

            return np.mean([score["mse"] for score in val_scores])


if __name__ == "__main__":
    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        help="Input dataset from L3",
        default="../data/curated/marts_houses_downstream.parquet",
    )
    parser.add_argument(
        "--algorithm",
        help="Algorithm to tune",
        choices=["ydf", "lightgbm", "catboost", "xgboost"],
    )
    parser.add_argument(
        "--experiment-name", required=True, help="MLflow experiment name"
    )
    parser.add_argument("--tracking-url", help="MLflow tracking server URL")

    args = parser.parse_args()

    # change matplotlib backend
    matplotlib.use("Agg")

    # set mlflow tracking server
    if args.tracking_url:
        mlflow.set_tracking_uri(args.tracking_url)

    # create objective
    objective = OptunaObjective(args.algorithm, args.dataset)

    # load dataset
    objective.load_data()

    # create mlflow experiment
    experiment_id = get_or_create_experiment(args.experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    # create study
    study = optuna.create_study(
        direction="minimize",
        study_name=args.algorithm,
        storage="sqlite:///bogor_houses_v3.db",
        load_if_exists=True,
    )
    study.optimize(objective, n_trials=100)
