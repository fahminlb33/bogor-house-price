import re
import itertools

import nltk
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

import tensorflow as tf
from catboost import Pool
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.metrics import mean_squared_error, mean_absolute_percentage_error, mean_absolute_error, r2_score

# --- cleaning methods


class PreprocessorBase(object):

    def __init__(self) -> None:
        nltk.download('punkt')

    def derive_kecamatan(self, df):
        # extract kecamatan
        return df["address"].str.split(", ").str[0]

    def dedupe_agency_company(self, name):
        # return empty if null
        if pd.isnull(name):
            return name

        # return common property names
        COMMON_COMPANY = [
            "BRIGHTON", "CENTURY21", "CENTURY 21", "RAY WHITE", "LJ HOOKER",
            "XAVIER MARKS", "MR REALTY", "ERA FIESTA", "ASIA ONE", "RE/MAX",
            "ERA PROJECT", "ERA VICTORIA", "MPRO"
        ]
        for company in COMMON_COMPANY:
            if company in name:
                return company

        # remove common city names
        # COMMON_WORDS = ["PROPERTY", "PROPERTI", "BANDUNG", "JAKARTA", "GARDEN CITY", "GADING SERPONG", "KELAPA GADING", "CENGKARENG", "BINTARO", "BINTARO 3 BRANCH", "MALANG", "TANGERANG", "PALEMBANG"]
        # for word in COMMON_WORDS:
        #   if word in name:
        #     name = name.replace(word, "")

        # normalize common words
        if "PROPERTY" in name:
            name = name.replace("PROPERTY", "PROPERTI")
        if "NOT IDENTIFIED" in name:
            name = name.replace("NOT IDENTIFIED", "INDEPENDENT")
        if "INDEPDENDENT" in name:
            name = name.replace("INDEPDENDENT", "INDEPENDENT")

        # remove double spaces
        name = re.sub(' +', ' ', name).strip()

        # emtpty is independent
        if name == "":
            name = "INDEPENDENT"

        return name

    def dedupe_facilities(self, facilities):
        # inner function to clean facility info
        def clean_facility(name):
            # set uppercase
            name = name.upper()

            # remove outlier
            OUTLIERS = [
                "SAMPING & BELAKANG", "LISTRIK", "ATAP BAJA", "ROW JALAN",
                'BUILT IN ROBES', "R.TAMU", "STUDY", "BATH", "R. MAKAN",
                "SERTIFIKAT", "HADAP"
            ]
            for outlier in OUTLIERS:
                if outlier in name:
                    return None

            # replace rules to summarize facility
            RULESET = {
                "LAPANGAN BERMAIN": "PLAYGROUND",
                "PARKIR": "PARKIR",
                "JOGGING TRACK": "TRACK LARI",
                "JET PUMP": "AIR TANAH",
                "PAM": "AIR PAM",
                "CARPORT": "CARPORT",
                "GYM": "GYM",
                'SWIMMING POOL': "KOLAM RENANG",
                "AC": "AC",
                "AIR COND": "AC",
                "AIRCON": "AC",
                "SECURITY": "KEAMANAN",
                "KEAMANAN": "KEAMANAN",
                "COURTYARDS": "HALAMAN",
                "TERAS": "HALAMAN",
                "KITCHEN": "DAPUR",
                "KOMPOR": "DAPUR",
                "KULKAS": "DAPUR",
                "JEMURAN": "LAUNDRY",
                "LAUNDRY": "LAUNDRY",
                "CUCI": "LAUNDRY",
                "PEMANAS AIR": "WATER HEATER",
                "HOT WATER": "WATER HEATER",
                "TAMAN": "TAMAN",
                "GARDEN": "TAMAN",
                "ENTERTAINING": "TAMAN",
                "INTERNET": "INTERNET",
                "WIFI": "INTERNET",
                "WI-FI": "INTERNET",
            }

            # return rule if found
            for rule in RULESET:
                if rule in name:
                    return RULESET[rule]

            return name

        # return common property names
        return [
            x for x in set([clean_facility(tag) for tag in facilities])
            if x is not None
        ]

    def transform_dict_col(self, df, column_name, prefix="t_"):
        # dict to dataframe
        df_dict = pd.json_normalize(df[column_name])
        df_dict.columns = prefix + df_dict.columns.str.replace(' ',
                                                               '_').str.lower()
        df = df.join(df_dict)

        return df

    def transform_list_col(self,
                           df,
                           column_name,
                           prefix="t_",
                           return_embeddings=False,
                           binarize=True,
                           split_fun=None):
        TEMP_COL = "temp"

        # extract tags (possibly multiple tags per house and also compound tags)
        if split_fun is not None:
            df[TEMP_COL] = df[column_name].apply(split_fun)
        else:
            df[TEMP_COL] = df[column_name].apply(lambda values: list(
                set(
                    itertools.chain.from_iterable(
                        [x.lower().split("/") for x in values]))))

        # create one-hot encoded columns
        mlb = MultiLabelBinarizer(sparse_output=not return_embeddings)
        mlb.fit_transform(df[TEMP_COL])

        # create embeddings
        if not return_embeddings:
            # create sparse matrix with one-hot encoded columns
            df_spm = pd.DataFrame.sparse.from_spmatrix(mlb.transform(
                df[TEMP_COL]),
                                                       columns=mlb.classes_)
            df_spm.columns = prefix + df_spm.columns.str.replace(' ', '_')

            # join sparse matrix with original dataframe
            df = df.join(df_spm)
        else:
            # change original column to embeddings
            if binarize:
                df[column_name] = df[TEMP_COL].apply(
                    lambda value: np.array(mlb.transform([value])[0]))

        # drop temporary column
        df = df.drop(TEMP_COL, axis=1)

        return df

    def infer_spec_cols(self, df, prefix="t_"):
        # get the first part
        df[f"{prefix}lebar_jalan"] = df[f"{prefix}lebar_jalan"].str.split(
            " ").str[0]

        # transform to number
        df[f"{prefix}luas_tanah"] = df[f"{prefix}luas_tanah"].str.replace(
            "m²", "")
        df[f"{prefix}luas_bangunan"] = df[f"{prefix}luas_bangunan"].str.replace(
            "m²", "")
        df[f"{prefix}daya_listrik"] = df[f"{prefix}daya_listrik"].str.lower(
        ).str.replace("watt", "").str.replace("lainnya", "0")
        df[f"{prefix}lebar_jalan"] = df[f"{prefix}lebar_jalan"].str.split(
            " ").str[0]

        # Change column type
        df = df.astype({
            f"{prefix}luas_tanah": 'float64',
            f"{prefix}luas_bangunan": 'float64',
            f"{prefix}daya_listrik": 'float64',
            f"{prefix}lebar_jalan": 'float64',
            f"{prefix}kamar_tidur": 'float64',
            f"{prefix}carport": 'float64',
            f"{prefix}kamar_mandi": 'float64',
            f"{prefix}kamar_pembantu": 'float64',
            f"{prefix}dapur": 'float64',
            f"{prefix}jumlah_lantai": 'float64',
            f"{prefix}garasi": 'float64',
            f"{prefix}tahun_di_renovasi": 'float64',
            f"{prefix}kamar_mandi_pembantu": 'float64',
            f"{prefix}tahun_dibangun": 'float64',
        })

        return df

    def drop_features_by_nan(self, df: pd.DataFrame, threshold=0.5):
        # drop features with nan more than threshold
        return df.dropna(axis=1, thresh=threshold * len(df))


class DataLoaderMixin(PreprocessorBase):
    df: pd.DataFrame

    def __init__(self):
        self.df = None

    def load(self, dataset_path):
        # load dataset
        self.df = pd.read_json(dataset_path, lines=True)

    def prepare(self, log_price=True, dropna_thresh=0.0, drop_cols=None):
        # check if df is None
        if self.df is None:
            raise Exception("Please load dataset first")

        # --- clean data
        # extract kecamatan
        self.df["kecamatan"] = self.derive_kecamatan(self.df)

        # transform dictionary to columns
        self.df = super().transform_dict_col(self.df, "specs", "spec_")
        self.df = self.transform_list_col(self.df,
                                          "tags",
                                          return_embeddings=True,
                                          binarize=True)
        self.df = self.transform_list_col(self.df,
                                          "facilities",
                                          return_embeddings=True,
                                          binarize=True,
                                          split_fun=self.dedupe_facilities)

        # infer column
        self.df = self.infer_spec_cols(self.df, prefix="spec_")

        # drop unused columns
        self.df = self.df.drop(columns=[
            'id', 'images', 'installment', 'address', 'description', 'specs',
            'agent', 'url', 'spec_id_iklan'
        ])

        # drop additional columns
        if drop_cols is not None:
            self.df = self.df.drop(columns=drop_cols)
        elif dropna_thresh > 0:
            self.df = self.drop_features_by_nan(self.df,
                                                threshold=dropna_thresh)

        # fill na
        FILLNA_COLS = self.df.select_dtypes(include=['object']).columns.tolist()
        self.df = self.df.fillna({k: "[UNK]" for k in FILLNA_COLS})

        FILLNA_NUM_COLS = self.df.select_dtypes(
            exclude=['object']).columns.tolist()
        self.df = self.df.fillna({k: self.df[k].mean() for k in FILLNA_NUM_COLS})

        # log price
        if log_price:
            self.df["price"] = np.log(self.df["price"])


class MetricsPlottingMixin(object):

    def plot_predictions(self, y_test, y_pred):
        fig = plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_pred)
        plt.xlabel("Actual")
        plt.ylabel("Predicted")
        plt.axis("tight")

        return fig

    def plot_residuals(self, y_test, y_pred):
        fig = plt.figure(figsize=(10, 5))
        plt.scatter(y_test, y_test - y_pred, color="blue", alpha=0.5)
        plt.axhline(y=0, color="r", linestyle="-")
        plt.xlabel("Actual")
        plt.ylabel("Residuals")
        plt.axis("tight")

        return fig

    def plot_distributions(self, y_test, y_pred):
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 5))

        ax1.hist(y_test, bins=50)
        ax1.set_title("Actual")

        ax2.hist(y_pred, bins=50)
        ax2.set_title("Predicted")

        return fig

    def metrics(self, y_test, y_pred):
        return {
            "Mean": np.mean(y_pred),
            "StdDev": np.std(y_pred),
            "Var": np.var(y_pred),
            "R2": r2_score(y_test, y_pred),
            "MSE": mean_squared_error(y_test, y_pred),
            "MAE": mean_absolute_error(y_test, y_pred),
            "MAPE": mean_absolute_percentage_error(y_test, y_pred)
        }


# --- library specific cleaning


class CatboostLoader(DataLoaderMixin, MetricsPlottingMixin):

    def prepare(self,
                log_price=True,
                dropna_thresh=0.0,
                drop_cols=None,
                train_size=0.75):
        # prepare
        super().prepare(log_price=log_price,
                        dropna_thresh=dropna_thresh,
                        drop_cols=drop_cols)

        # split data
        X = self.df.drop(columns=["price"])
        y = self.df["price"]

        self.feature_names = X.columns.tolist()

        # split train and test
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X, y, train_size=train_size, random_state=42)

        # select categorical columns
        cat_cols = list(
            set(self.df.select_dtypes(include=['object']).columns) -
            set(["tags", "facilities"]))
        embed_cols = ["tags", "facilities"]

        # create pool
        self.train_data = Pool(self.X_train,
                               label=self.y_train,
                               cat_features=cat_cols,
                               embedding_features=embed_cols)
        self.test_data = Pool(self.X_test,
                              label=self.y_test,
                              cat_features=cat_cols,
                              embedding_features=embed_cols)

    def plot_predictions(self, y_pred):
        return super().plot_predictions(self.y_test, y_pred)

    def plot_residuals(self, y_pred):
        return super().plot_residuals(self.y_test, y_pred)

    def plot_distributions(self, y_pred):
        return super().plot_distributions(self.y_test, y_pred)

    def metrics(self, y_pred):
        return super().metrics(self.y_test, y_pred)


class TensorflowLoader(DataLoaderMixin, MetricsPlottingMixin):
    MULTIHOT_COLS = set(["tags", "facilities"])

    def prepare(self,
                log_price=True,
                dropna_thresh=0.0,
                drop_cols=None,
                train_size=0.75):
        # prepare
        super().prepare(log_price=log_price,
                        dropna_thresh=dropna_thresh,
                        drop_cols=drop_cols)

        # create tf dataset
        df_train, df_test = train_test_split(self.df,
                                             train_size=train_size,
                                             random_state=42)

        self.ds_train = self._df_to_ds(df_train)
        self.ds_test = self._df_to_ds(df_test)

        self.feature_names = list(
            set(self.df.columns.tolist()) - set(["price"]))
        self.y_test = df_test["price"].values.reshape(-1, 1)

        # create input and outputs
        self.inputs = []
        self.features = []

        # encode numerical features
        num_cols = set(
            self.df.select_dtypes(include=['float64']).columns) - set(["price"])
        for col in num_cols:
            input_col = tf.keras.Input(shape=(1,), name=col)
            normalization_layer = self._get_normalization_layer(
                col, self.ds_train)
            encoded_col = normalization_layer(input_col)

            self.inputs.append(input_col)
            self.features.append(encoded_col)

        # encode categorical features
        cat_cols = set(self.df.select_dtypes(include=['object']).columns) - set(
            self.MULTIHOT_COLS)
        for col in cat_cols:
            input_col = tf.keras.Input(shape=(1,), name=col, dtype="string")
            encoding_layer = self._get_category_encoding_layer(col,
                                                               self.ds_train,
                                                               max_tokens=100)
            encoded_col = encoding_layer(input_col)

            self.inputs.append(input_col)
            self.features.append(encoded_col)

        # input for multi-hot categorical features
        for col in self.MULTIHOT_COLS:
            input_tags = tf.keras.Input(shape=(len(self.df[col][0]),),
                                        name=col,
                                        dtype="int32")
            encoded_tags = tf.cast(input_tags, tf.float32)

            self.inputs.append(input_tags)
            self.features.append(encoded_tags)

    def plot_predictions(self, y_pred):
        return super().plot_predictions(self.y_test, y_pred)

    def plot_residuals(self, y_pred):
        return super().plot_residuals(self.y_test, y_pred)

    def plot_distributions(self, y_pred):
        return super().plot_distributions(self.y_test, y_pred)

    def metrics(self, y_pred):
        return super().metrics(self.y_test, y_pred)

    def plot_loss(self, history):
        epochs = range(1, len(history.history["mse"]) + 1)

        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(10, 4))
        ax1.plot(epochs, history.history["mse"], label='Training MSE')
        ax1.plot(epochs, history.history["val_mse"], label='Validation MSE')
        ax1.set_title('Training and validation MSE')
        ax1.legend()

        ax2.plot(epochs, history.history["loss"], label='Training loss')
        ax2.plot(epochs, history.history["val_loss"], label='Validation loss')
        ax2.set_title('Training and validation loss')
        ax2.legend()

        fig.tight_layout()

        return fig

    def _df_to_ds(self, dataframe, shuffle=True, batch_size=128):
        # copy dataframe
        df = dataframe.copy()

        # extract labels
        labels = df.pop('price')

        # convert multihot columns to array
        for col in self.MULTIHOT_COLS:
            df[col] = df[col].apply(lambda x: np.array(x))

        # create dictionary
        dc = {}
        for k, v in df.items():
            if k in self.MULTIHOT_COLS:
                dc[k] = np.array([np.array(x) for x in v.values])
            else:
                dc[k] = v.values

        # create tf dataset
        ds = tf.data.Dataset.from_tensor_slices((dc, labels))

        # shuffle
        if shuffle:
            ds = ds.shuffle(buffer_size=len(dataframe))

        # batch and prefetch
        ds = ds.batch(batch_size)
        ds = ds.prefetch(batch_size)

        return ds

    def _get_normalization_layer(self, name, dataset):
        # Create a Normalization layer for the feature.
        normalizer = tf.keras.layers.Normalization(axis=None)

        # Prepare a Dataset that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the statistics of the data.
        normalizer.adapt(feature_ds)

        return normalizer

    def _get_category_encoding_layer(self, name, dataset, max_tokens=None):
        # Create a layer that turns strings into integer indices.
        index = tf.keras.layers.StringLookup(max_tokens=max_tokens)

        # Prepare a `tf.data.Dataset` that only yields the feature.
        feature_ds = dataset.map(lambda x, y: x[name])

        # Learn the set of possible values and assign them a fixed integer index.
        index.adapt(feature_ds)

        # Encode the integer indices.
        encoder = tf.keras.layers.CategoryEncoding(
            num_tokens=index.vocabulary_size())

        # Apply multi-hot encoding to the indices. The lambda function captures the
        # layer, so you can use them, or include them in the Keras Functional model later.
        return lambda feature: encoder(index(feature))
