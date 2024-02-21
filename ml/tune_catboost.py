import argparse
import mlflow
import optuna

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from ml_plot import plot_distributions, plot_residuals, plot_predictions

mlflow.set_tracking_uri("http://10.20.20.102:8009")


def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


class Objective():

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path

    def load_data(self):
        # load dataset
        self.df = pd.read_parquet(self.dataset_path)

        # create X and y
        self.X = self.df.drop(columns=["price"])
        self.y = self.df["price"]

        # identify columns
        self.cat_cols = [
            col for col in self.df.select_dtypes(include=["object"]).columns
        ]

    def __call__(self, trial: optuna.Trial):
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            # create hyperparameters
            # from: https://github.com/optuna/optuna-examples/blob/main/catboost/catboost_pruning.py
            params = {
                "iterations":
                    trial.suggest_int("iterations", 10, 1000),
                "depth":
                    trial.suggest_int("depth", 1, 12),
                # "colsample_bylevel": trial.suggest_float("colsample_bylevel", 0.01, 1.0, log=True),
                "subsample":
                    trial.suggest_float("subsample", 0.1, 1, log=True),
                "grow_policy":
                    trial.suggest_categorical(
                        "grow_policy",
                        ["SymmetricTree", "Depthwise", "Lossguide"]),
                "learning_rate":
                    trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "min_data_in_leaf":
                    trial.suggest_int("min_data_in_leaf", 1, 100),

                # fixed hyperparameters
                "task_type":
                    "GPU",
                "objective":
                    "RMSE",
                "bootstrap_type":
                    "Bernoulli",
                "random_seed":
                    21,
                "verbose":
                    0,
            }

            # create scores
            scores = {
                "mean": [],
                "std": [],
                "var": [],
                "mse": [],
                "mae": [],
                "mape": [],
                "r2": []
            }

            # perform cross validation
            cv = KFold(n_splits=10, shuffle=True, random_state=21)
            for fold_i, (train_idx,
                         test_idx) in enumerate(cv.split(self.X, self.y)):
                print(f"Training fold {fold_i + 1}")

                # split data
                X_train, X_test = self.X.iloc[train_idx], self.X.iloc[test_idx]
                y_train, y_test = self.y.iloc[train_idx], self.y.iloc[test_idx]

                # create pool
                train_pool = Pool(data=X_train,
                                  label=y_train,
                                  cat_features=self.cat_cols)
                test_pool = Pool(data=X_test,
                                 label=y_test,
                                 cat_features=self.cat_cols)

                # create model
                model = CatBoostRegressor(**params)

                # fit model
                model.fit(train_pool,
                          eval_set=test_pool,
                          verbose=0,
                          early_stopping_rounds=100)

                # run prediction
                y_pred = model.predict(X_test)

                # log metrics
                scores["mean"].append(np.mean(y_pred))
                scores["std"].append(np.std(y_pred))
                scores["var"].append(np.var(y_pred))
                scores["mse"].append(mean_squared_error(y_test, y_pred))
                scores["mae"].append(mean_absolute_error(y_test, y_pred))
                scores["mape"].append(
                    mean_absolute_percentage_error(y_test, y_pred))
                scores["r2"].append(r2_score(y_test, y_pred))

            # log params and metrics
            mlflow.log_params(params)
            mlflow.log_metric("mean", np.mean(scores["mean"]))
            mlflow.log_metric("std", np.mean(scores["std"]))
            mlflow.log_metric("var", np.mean(scores["var"]))
            mlflow.log_metric("mse", np.mean(scores["mse"]))
            mlflow.log_metric("mae", np.mean(scores["mae"]))
            mlflow.log_metric("mape", np.mean(scores["mape"]))
            mlflow.log_metric("r2", np.mean(scores["r2"]))
            mlflow.log_figure(plot_residuals(y_test, y_pred), "residuals.png")
            mlflow.log_figure(plot_predictions(y_test, y_pred),
                              "predictions.png")
            mlflow.log_figure(plot_distributions(y_test, y_pred),
                              "distributions.png")

            plt.close("all")

            return np.mean(scores["mse"])


if __name__ == "__main__":
    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        help="Input dataset from L3",
        default="./dataset/curated/marts_ml_train_sel_all.parquet")

    args = parser.parse_args()

    # change matplotlib backend
    matplotlib.use("Agg")

    # create objective
    objective = Objective(args.dataset)

    # load dataset
    objective.load_data()

    # create mlflow experiment
    experiment_id = get_or_create_experiment("Project House Price: CatBoost")
    mlflow.set_experiment(experiment_id=experiment_id)

    # create study
    study = optuna.create_study(direction="minimize",
                                study_name="catboost",
                                storage="sqlite:///bogor_houses.db",
                                load_if_exists=True)
    study.optimize(objective, n_trials=100)
