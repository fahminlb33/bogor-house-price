import optuna
import mlflow
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

import tensorflow as tf
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import TensorflowLoader

DATASET_PATH = "./dataset/houses-9k.json"

tf.random.set_seed(42)
mlflow.set_tracking_uri("http://10.20.20.102:8009")


def get_or_create_experiment(experiment_name):
    if experiment := mlflow.get_experiment_by_name(experiment_name):
        return experiment.experiment_id
    else:
        return mlflow.create_experiment(experiment_name)


class Objective():

    def __init__(self, dataset_path):
        self.dataset_path = dataset_path
        self.clean_params = {
            "log_price": True,
            "train_size": 0.75,
            "drop_cols": None,
            "dropna_thresh": 0.7
        }

    def load_data(self):
        # load dataset
        self.loader = TensorflowLoader()
        self.loader.load(self.dataset_path)

        # perform cleaning
        self.loader.prepare(**self.clean_params)

    def __call__(self, trial: optuna.Trial):
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            # create hyperparameters
            params = {
                "iterations":
                    trial.suggest_int("iterations", 50, 200, step=50),
                "learning_rate":
                    trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "depth":
                    trial.suggest_int("depth", 1, 10),
                "units":
                    trial.suggest_categorical("units", [32, 64, 128, 256]),
                "dropout_enabled":
                    trial.suggest_categorical("dropout_enabled", [True, False]),
                "dropout_rate":
                    trial.suggest_float("dropout_rate", 0.1, 0.5),
            }

            # concatenate all features
            x = tf.keras.layers.concatenate(self.loader.features)

            # create hidden layers
            for _ in range(params["depth"]):
                x = tf.keras.layers.Dense(params["units"], activation="relu")(x)

            # create dropout layer
            if params["dropout_enabled"]:
                x = tf.keras.layers.Dropout(params["dropout_rate"])(x)

            # create output layer
            output = tf.keras.layers.Dense(1)(x)

            # create model
            model = tf.keras.Model(self.loader.inputs, output)

            # compile model
            model.compile(optimizer=tf.keras.optimizers.Adam(
                params["learning_rate"]),
                          loss=tf.keras.losses.MeanSquaredError(),
                          metrics=["mse", "mae"])

            # run training
            history = model.fit(self.loader.ds_train,
                                epochs=params["iterations"],
                                validation_data=self.loader.ds_test,
                                verbose=0)

            # run prediction
            y_pred = model.predict(self.loader.ds_test)

            # log params and metrics
            mlflow.log_params(params)
            mlflow.log_params(self.clean_params)
            mlflow.log_param("feature_names", self.loader.feature_names)

            mlflow.log_metrics(self.loader.metrics(y_pred))

            mlflow.log_figure(self.loader.plot_residuals(y_pred),
                              "residuals.png")
            mlflow.log_figure(self.loader.plot_predictions(y_pred),
                              "predictions.png")
            mlflow.log_figure(self.loader.plot_distributions(y_pred),
                              "distributions.png")
            mlflow.log_figure(self.loader.plot_loss(history), "loss.png")

            plt.close("all")

            return mean_squared_error(self.loader.y_test, y_pred), r2_score(self.loader.y_test, y_pred)


if __name__ == "__main__":
    # change matplotlib backend
    matplotlib.use("Agg")

    # create objective
    objective = Objective(DATASET_PATH)

    # load dataset
    objective.load_data()

    # create mlflow experiment
    experiment_id = get_or_create_experiment("Bogor House Price: TensorFlow")
    mlflow.set_experiment(experiment_id=experiment_id)

    # create study
    study = optuna.create_study(directions=["minimize", "maximize"],
                                study_name="tensorflow",
                                storage="sqlite:///bogor_houses.db",
                                load_if_exists=True)
    study.optimize(objective, n_trials=100)
