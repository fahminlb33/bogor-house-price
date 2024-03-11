import os
import timeit
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
                 bootstrap_type: str,
                 depth: int,
                 grow_policy: str,
                 learning_rate: float,
                 min_data_in_leaf: int,
                 subsample: float,
                 random_state: int = 21,
                 verbose: int = 0):
        super().__init__()

        self.dataset_path = dataset_path
        self.output_path = output_path
        self.bootstrap_type = bootstrap_type
        self.depth = depth
        self.grow_policy = grow_policy
        self.learning_rate = learning_rate
        self.min_data_in_leaf = min_data_in_leaf
        self.subsample = subsample
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

        # create pool
        self.train_pool = Pool(data=self.X,
                               label=self.y,
                               cat_features=self.cat_cols)

    def train(self):
        # create log dir
        log_dir = os.path.join(self.output_path, "logs")

        # create hyperparameters
        params = {
            "bootstrap_type": self.bootstrap_type,
            "depth": self.depth,
            "grow_policy": self.grow_policy,
            "learning_rate": self.learning_rate,
            "min_data_in_leaf": self.min_data_in_leaf,
            "subsample": self.subsample,
            "verbose": self.verbose,
            "task_type": "GPU",
            # "task_type": "CPU",
            "train_dir": log_dir,
            "random_seed": self.random_state,
        }

        # create model
        model = CatBoostRegressor(**params)

        # fit model
        start_time = timeit.default_timer()
        model.fit(self.train_pool)
        elapsed = timeit.default_timer() - start_time
        self.logger.info(f"Fit completed in {elapsed:.2f} seconds")

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

        super().run()


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
                        default=f"./ml_models/catboost-{today}")
    parser.add_argument("--bootstrap", help="Bootstrap", default="Bernoulli")
    parser.add_argument("--depth", help="Depth", default=12)
    parser.add_argument("--grow_policy",
                        help="Grow policy",
                        default="Depthwise")
    parser.add_argument("--learning_rate",
                        help="Learning rate",
                        default=0.05683273851486543)
    parser.add_argument("--min_data_in_leaf",
                        help="Min data in leaf",
                        default=89)
    parser.add_argument("--subsample",
                        help="Subsample",
                        default=0.851934076287287)
    parser.add_argument("--random_state", help="Random state", default=21)
    parser.add_argument("--verbose", help="Verbose", default=0)

    args = parser.parse_args()

    # train model
    trainer = TrainCatBoost(args.dataset, args.output_path, args.bootstrap,
                            args.depth, args.grow_policy, args.learning_rate,
                            args.min_data_in_leaf, args.subsample,
                            args.random_state, args.verbose)
    trainer.run()
