import os
import argparse
import datetime

import pandas as pd
import matplotlib.pyplot as plt

from catboost import CatBoostRegressor, Pool

from ml_base import TrainerMixin


class TrainCatBoost(TrainerMixin):

    def __init__(self,
                 dataset_path: str,
                 output_path: str,
                 random_state: int = 21,
                 verbose: int = 0):
        super().__init__()

        self.dataset_path = dataset_path
        self.output_path = output_path
        self.random_state = random_state
        self.verbose = verbose

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

    def train(self):
        # create log dir
        log_dir = os.path.join(self.output_path, "logs")

        # create hyperparameters
        params = {
            "verbose": self.verbose,
            "task_type": "GPU",
            "train_dir": log_dir,
            "random_seed": self.random_state,
        }

        # create pool
        train_pool = Pool(data=self.X, label=self.y, cat_features=self.cat_cols)

        # create model
        model = CatBoostRegressor(**params)

        # fit model
        model.fit(train_pool)

        # --- save model
        self.logger.info("Saving model...")
        model.save_model(os.path.join(self.output_path, "model.cbm"))

        # --- save model importance
        importances = model.feature_importances_
        feature_names = list(self.X.columns)

        # create importance series
        forest_importances = pd.Series(importances, index=feature_names) \
            .sort_values(ascending=False)

        forest_importances.to_csv(
            os.path.join(self.output_path, "importance.csv"))

        # plot importance
        fig, ax = plt.subplots(figsize=(20, 10))
        forest_importances.plot.bar(ax=ax)
        ax.set_title("Feature importances using Gini ratio")
        ax.set_ylabel("Mean decrease in impurity")
        fig.tight_layout()

        # save plot
        fig.savefig(os.path.join(self.output_path, "importance.png"))

    def run(self):
        self.logger.info("Creating output directory...")
        os.makedirs(self.output_path, exist_ok=True)

        self.logger.info("Loading dataset...")
        self.load_data()

        self.logger.info("Training model...")
        self.train()


if __name__ == "__main__":
    # today
    today = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")

    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--dataset",
        help="Input dataset from L3",
        default="./dataset/curated/marts_ml_train_sel_all.parquet")
    parser.add_argument("--output-path",
                        help="Output path for model and metrics",
                        default=f"./models/catboost-{today}")
    parser.add_argument("--random_state", help="Random state", default=21)
    parser.add_argument("--verbose", help="Verbose", default=0)

    args = parser.parse_args()

    # train model
    trainer = TrainCatBoost(args.dataset, args.output_path, args.random_state,
                            args.verbose)
    trainer.run()
