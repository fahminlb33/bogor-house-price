import optuna
import mlflow

import numpy as np
import pandas as pd
import matplotlib
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import KFold
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_squared_error, mean_absolute_error, mean_absolute_percentage_error, r2_score

from ml_plot import plot_distributions, plot_residuals, plot_predictions

DATASET_PATH = "./dataset/etl/L2.regression_inliers.parquet"

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

        # select columns
        floor_mat_cols = [
            col for col in self.df.columns if col.startswith("floor_mat_")
        ]
        house_mat_cols = [
            col for col in self.df.columns if col.startswith("house_mat_")
        ]
        tags_cols = [col for col in self.df.columns if col.startswith("tags_")
                    ] + [
                        "hook_available", "ruang_tamu_available",
                        "ruang_makan_available", "terjangkau_internet_available"
                    ]

        cat_cols = [
            "kondisi_perabotan_norm", "kondisi_properti_norm",
            "konsep_dan_gaya_rumah", "sumber_air", "pemandangan", "sertifikat"
        ]
        num_cols = [
            "lebar_jalan_num", "daya_listrik_num", "luas_bangunan_num",
            "luas_tanah_num", "carport", "garasi", "dapur", "jumlah_lantai",
            "kamar_mandi_pembantu", "kamar_pembantu", "kamar_mandi",
            "kamar_tidur"
        ]

        # create encoders
        categorical_encoder = Pipeline(steps=[
            ("encoder", OneHotEncoder(handle_unknown="ignore")),
        ])

        numerical_encoder = Pipeline(steps=[
            ("scaler", MinMaxScaler()),
        ])

        # create transformer
        self.compose_transformers = ColumnTransformer(transformers=[
            ("passthrough", "passthrough",
             tags_cols + floor_mat_cols + house_mat_cols),
            ("catergorical_encoder", categorical_encoder, cat_cols),
            ("numerical_encoder", numerical_encoder, num_cols),
        ])

    def __call__(self, trial: optuna.Trial):
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            # create hyperparameters
            params = {
                "n_estimators":
                    trial.suggest_int("n_estimators", 200, 2000, step=10),
                "max_depth":
                    trial.suggest_int("max_depth", 10, 110, step=10),
                "min_samples_split":
                    trial.suggest_categorical("min_samples_split", [2, 5, 10]),
                "min_samples_leaf":
                    trial.suggest_categorical("min_samples_leaf", [1, 2, 4]),
                "bootstrap":
                    trial.suggest_categorical("bootstrap", [True, False]),
                "max_features":
                    trial.suggest_categorical("max_features_2", [None, "sqrt"]),
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

                # create pipeline
                clf = Pipeline(steps=[
                    ("preprocessor", self.compose_transformers),
                    ("regressor",
                     RandomForestRegressor(**params, n_jobs=4, random_state=21)
                    ),
                ])

                # fit model
                clf.fit(X_train, y_train)

                # run prediction
                y_pred = clf.predict(X_test)

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
    # change matplotlib backend
    matplotlib.use("Agg")

    # create objective
    objective = Objective(DATASET_PATH)

    # load dataset
    objective.load_data()

    # create mlflow experiment
    experiment_id = get_or_create_experiment(
        "Project House Price: Random Forest")
    mlflow.set_experiment(experiment_id=experiment_id)

    # create study
    study = optuna.create_study(direction="minimize",
                                study_name="random_forest",
                                storage="sqlite:///bogor_houses.db",
                                load_if_exists=True)
    study.optimize(objective, n_trials=100)
