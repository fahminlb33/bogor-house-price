import optuna
import mlflow
import matplotlib
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor
from sklearn.metrics import mean_squared_error, r2_score

from preprocessing import CatboostLoader

DATASET_PATH = "./dataset/houses-9k.json"

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
        self.loader = CatboostLoader()
        self.loader.load(self.dataset_path)

        # perform cleaning
        self.loader.prepare(**self.clean_params)

    def __call__(self, trial: optuna.Trial):
        with mlflow.start_run(run_name=f"trial-{trial.number}"):
            # create hyperparameters
            params = {
                "iterations":
                    trial.suggest_int("iterations", 100, 500, step=50),
                "learning_rate":
                    trial.suggest_float("learning_rate", 1e-3, 0.1, log=True),
                "depth":
                    trial.suggest_int("depth", 1, 10),
                "subsample":
                    trial.suggest_float("subsample", 0.05, 1.0),
                "colsample_bylevel":
                    trial.suggest_float("colsample_bylevel", 0.05, 1.0),
                "min_data_in_leaf":
                    trial.suggest_int("min_data_in_leaf", 1, 100),
            }

            # create regressor
            model = CatBoostRegressor(**params, silent=True, random_seed=42)

            # run training
            model.fit(self.loader.train_data, eval_set=self.loader.test_data)

            # run prediction
            y_pred = model.predict(self.loader.test_data)

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
    experiment_id = get_or_create_experiment("Bogor House Price: CatBoost")
    mlflow.set_experiment(experiment_id=experiment_id)

    # create study
    study = optuna.create_study(directions=["minimize", "maximize"],
                                study_name="catboost",
                                storage="sqlite:///bogor_houses.db",
                                load_if_exists=True)
    study.optimize(objective, n_trials=100)
