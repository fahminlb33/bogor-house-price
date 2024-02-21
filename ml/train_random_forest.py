import os
import argparse
import datetime

import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import OneHotEncoder, MinMaxScaler
from sklearn.ensemble import RandomForestRegressor

from ml_base import TrainerMixin


class TrainRandomForest(TrainerMixin):

    def __init__(self,
                 dataset_path: str,
                 output_path: str,
                 bootstrap: bool = True,
                 max_depth: int = 30,
                 max_features: str = None,
                 min_samples_leaf: int = 1,
                 min_samples_split: int = 2,
                 n_estimators: int = 1900,
                 n_jobs: int = -1,
                 random_state: int = 21):
        super().__init__()

        self.dataset_path = dataset_path
        self.output_path = output_path
        self.bootstrap = bootstrap
        self.max_depth = max_depth
        self.max_features = max_features
        self.min_samples_leaf = min_samples_leaf
        self.min_samples_split = min_samples_split
        self.n_estimators = n_estimators
        self.n_jobs = n_jobs
        self.random_state = random_state

    def load_data(self):
        # load dataset
        self.df = pd.read_parquet(self.dataset_path)

        # create X and y
        self.X = self.df.drop(columns=["price"])
        self.y = self.df["price"]

        # identify columns
        self.multihot_cols = []
        self.multihot_cols.extend(
            [col for col in self.df.columns if col.startswith("floor_mat_")])
        self.multihot_cols.extend(
            [col for col in self.df.columns if col.startswith("house_mat_")])
        self.multihot_cols.extend(
            [col for col in self.df.columns if col.startswith("facility_")])
        self.multihot_cols.extend(
            [col for col in self.df.columns if col.startswith("tag_")])

        # extra features not included in tags_
        extra_tags = [
            "ruang_tamu", "ruang_makan", "terjangkau_internet", "hook"
        ]

        for tag in extra_tags:
            if tag in self.df.columns:
                self.multihot_cols.append(tag)

        # categorical columns
        self.cat_cols = [
            col for col in self.df.select_dtypes(include=["object"]).columns
        ]

        # numerical columns
        self.num_cols = list(
            set(self.df.columns) -
            set(self.multihot_cols + self.cat_cols + ["price"]))

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

    def train(self):
        # create hyperparameters
        params = {
            "bootstrap": self.bootstrap,
            "max_depth": self.max_depth,
            "max_features": self.max_features,
            "min_samples_leaf": self.min_samples_leaf,
            "min_samples_split": self.min_samples_split,
            "n_estimators": self.n_estimators,
        }

        # create pipeline
        clf = Pipeline(steps=[
            ("preprocessor", self.compose_transformers),
            ("regressor",
             RandomForestRegressor(**params,
                                   n_jobs=self.n_jobs,
                                   random_state=self.random_state,
                                   verbose=1)),
        ])

        # fit model
        clf.fit(self.X, self.y)

        # --- save model
        self.logger.info("Saving model...")
        joblib.dump(clf, os.path.join(self.output_path, "model.joblib"))

        # --- save model importance
        # load model
        forest = clf.named_steps["regressor"]

        # get importance and calculate standard deviation
        importances = forest.feature_importances_
        std = np.std([tree.feature_importances_ for tree in forest.estimators_],
                     axis=0)

        # normalize feature names
        feature_names = self.compose_transformers.get_feature_names_out()
        feature_names = [
            name.replace("catergorical_encoder__", "") for name in feature_names
        ]
        feature_names = [
            name.replace("numerical_encoder__", "") for name in feature_names
        ]
        feature_names = [
            name.replace("passthrough__", "") for name in feature_names
        ]

        # create importance series
        forest_importances = pd.Series(
            importances, index=feature_names).sort_values(ascending=False)
        forest_importances.to_csv(
            os.path.join(self.output_path, "importance.csv"))

        # plot importance
        fig, ax = plt.subplots(figsize=(20, 10))
        forest_importances.plot.bar(yerr=std, ax=ax)
        ax.set_title("Feature importances using MDI")
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
                        default=f"./models/random_forest-{today}")
    parser.add_argument("--bootstrap", help="Bootstrap", default=True)
    parser.add_argument("--max_depth", help="Max depth", default=30)
    parser.add_argument("--max_features", help="Max features", default=None)
    parser.add_argument("--min_samples_leaf",
                        help="Min samples leaf",
                        default=1)
    parser.add_argument("--min_samples_split",
                        help="Min samples split",
                        default=2)
    parser.add_argument("--n_estimators", help="N estimators", default=1900)
    parser.add_argument("--n_jobs", help="N jobs", default=-1)
    parser.add_argument("--random_state", help="Random state", default=21)

    args = parser.parse_args()

    # train model
    trainer = TrainRandomForest(args.dataset, args.output_path, args.bootstrap,
                                args.max_depth, args.max_features,
                                args.min_samples_leaf, args.min_samples_split,
                                args.n_estimators, args.n_jobs,
                                args.random_state)
    trainer.run()
