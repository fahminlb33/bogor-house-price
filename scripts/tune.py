import time
import argparse

import mlflow
import optuna
import optunahub

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.figure

import lightgbm as lgb

from sklearn.model_selection import KFold, train_test_split
from sklearn.metrics import (
    mean_squared_error,
    mean_absolute_error,
    mean_absolute_percentage_error,
    r2_score,
)

FEATS_MONOTONE_UP = [
    "luas_tanah",
    "luas_bangunan",
    "kamar_tidur",
    "kamar_mandi",
    "jumlah_lantai",
]

FEATS = [
    "price",
    "subdistrict",
    "kamar_mandi",
    "kamar_pembantu",
    "luas_tanah",
    "luas_bangunan",
    "jumlah_lantai",
    "tahun_dibangun",
    "daya_listrik",
    "sumber_air",
    "land_building_ratio",
    "total_beds",
    "total_baths",
    "building_area_floor_ratio",
    "vehicle_accessibility",
]


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
        "r2": r2_score(y_true, y_pred),
        "mse": mean_squared_error(y_true, y_pred),
        "mae": mean_absolute_error(y_true, y_pred),
        "mape": mean_absolute_percentage_error(y_true, y_pred),
        "smape": symmetric_mean_absolute_percentage_error(y_true, y_pred),
    }


class OptunaObjective:
    def __init__(self, dataset_path: str, n_splits=5):
        self.dataset_path = dataset_path
        self.n_splits = n_splits

    def load_data(self):
        # load dataset
        df_data = pd.read_parquet(self.dataset_path)

        # log-tranform price
        df_data["price"] = np.log(df_data["price"])

        # derive new features
        df_data["land_building_ratio"] = df_data["luas_tanah"] / df_data["luas_bangunan"]
        df_data["total_beds"] = df_data["kamar_tidur"] + df_data["kamar_pembantu"]
        df_data["total_baths"] = df_data["kamar_mandi"] + df_data["kamar_mandi_pembantu"]
        df_data["bed_bath_ratio"] = df_data["kamar_tidur"] / df_data["kamar_mandi"]
        df_data["total_bed_bath_ratio"] = df_data["total_beds"] / df_data["total_baths"]
        df_data["rennovated_built_diff"] = df_data["tahun_di_renovasi"] - df_data["tahun_dibangun"]
        df_data["building_area_floor_ratio"] = df_data["luas_bangunan"] / df_data["jumlah_lantai"]
        df_data["vehicle_accessibility"] = df_data["garasi"] + df_data["carport"] / df_data["lebar_jalan"]

        # feature selection
        df_data = df_data[FEATS]

        # convert categorical columns
        cat_cols = df_data.select_dtypes("object").columns.tolist()
        df_data = df_data.fillna(value={k: "" for k in cat_cols})
        df_data[cat_cols] = df_data[cat_cols].astype("category")

        # split data
        train_df, test_df = train_test_split(df_data, test_size=0.33, random_state=22)

        self.cat_cols = cat_cols
        self.train_df = train_df
        self.test_df = test_df

        # create monotone constraint
        self.monotone_constraints = [1 if x in FEATS_MONOTONE_UP else 0 for x in train_df.drop(columns=["price"]).columns]


    def __call__(self, trial: optuna.Trial):
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            # create hyperparameters
            # https://www.kaggle.com/code/mlisovyi/lightgbm-hyperparameter-optimisation-lb-0-761
            # https://lightgbm.readthedocs.io/en/stable/Parameters-Tuning.html
            # https://lightgbm.readthedocs.io/en/latest/Parameters.html
            params = {
                # tunable hyperparameters
                "max_depth": trial.suggest_int("max_depth", 3, 10),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 100),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 100),
                "num_leaves": trial.suggest_int("num_leaves", 6, 50),
                'learning_rate': trial.suggest_float('learning_rate', 0.001, 0.1, log=True),
                "num_iterations": trial.suggest_int("num_iterations", 100, 500),
                'min_data_in_leaf': trial.suggest_int('min_data_in_leaf', 10, 50),
                "bagging_fraction": trial.suggest_float("bagging_fraction", 0.5, 1),
                "feature_fraction": trial.suggest_float("feature_fraction", 0.5, 1),
                "extra_trees": trial.suggest_categorical("extra_trees", [True, False]),
                "objective": trial.suggest_categorical("objective", ["regression", "huber", "fair"]),
                # fixed hyperparameters
                "seed": 22,
                "metric": "rmse",
                "verbosity": -1,
                "monotone_constraints": self.monotone_constraints
            }

            # create scores
            val_scores = []

            # perform cross validation
            cv = KFold(n_splits=self.n_splits, shuffle=True, random_state=21)
            for (train_idx, val_idx) in cv.split(self.train_df):

                # split data
                train_data = self.train_df.iloc[train_idx, :]
                X_train, y_train = (
                    train_data.drop(columns=["price"]),
                    train_data["price"],
                )

                # fit model
                model = lgb.LGBMRegressor(**params)
                model.fit(X_train, y_train, categorical_feature=self.cat_cols)

                # run prediction
                val_data = self.train_df.iloc[val_idx, :]
                X_val, y_val = val_data.drop(columns=["price"]), val_data["price"]

                y_val_pred = model.predict(X_val)

                # collect metrics
                val_scores.append(evaluate_model(y_val, y_val_pred))

            # log params
            if "monotone_constraints" in params:
                params.pop("monotone_constraints")

            mlflow.log_params(params)

            # log metrics
            for name in val_scores[0].keys():
                mlflow.log_metric(
                    f"val_{name}", np.mean([score[name] for score in val_scores])
                )

            # run prediction on test data
            X_test, y_test = self.test_df.drop(columns=["price"]), self.test_df["price"]
            y_test_pred = model.predict(X_test)

            test_score = evaluate_model(y_test, y_test_pred)

            # log test metrics
            for k, v in test_score.items():
                mlflow.log_metric(f"test_{k}", v)

            mlflow.log_figure(plot_predictions(y_test, y_test_pred), "predictions.png")
            mlflow.log_figure(plot_distributions(y_test, y_test_pred), "distributions.png")

            return np.mean([score["mse"] for score in val_scores])


if __name__ == "__main__":
    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--experiment-name", 
        type=str, 
        required=True, 
        help="MLflow experiment name"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        help="Input dataset from L3",
        default="../data/curated/marts_downstream_houses.parquet",
    )
    parser.add_argument(
        "--tracking-url", 
        type=str, 
        help="MLflow tracking server URL",
        default="http://10.20.20.102:8009/",
    )
    parser.add_argument(
        "--n-trials", 
        type=int, 
        help="Number of tuning iterations", 
        default=100
    )

    args = parser.parse_args()

    # change matplotlib backend
    matplotlib.use("Agg")

    # set mlflow tracking server
    mlflow.set_tracking_uri(args.tracking_url)

    # create objective
    objective = OptunaObjective(args.dataset)

    # load dataset
    objective.load_data()

    # create mlflow experiment
    experiment_id = get_or_create_experiment(args.experiment_name)
    mlflow.set_experiment(experiment_id=experiment_id)

    # get sampler
    sampler = optunahub.load_module("samplers/auto_sampler").AutoSampler()

    # create study
    study = optuna.create_study(
        sampler=sampler,
        storage="sqlite:///bogor_houses_v3.db",
        direction="minimize",
        study_name=args.experiment_name,
        load_if_exists=True,
    )

    study.optimize(objective, n_trials=100)
