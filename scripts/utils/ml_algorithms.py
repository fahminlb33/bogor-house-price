import numpy as np
import tensorflow as tf

from sklearn.model_selection import train_test_split


class CustomTensorFlowRegressor:

  def __init__(self,
               batch_size=256,
               epochs=200,
               random_seed=42,
               model_config="v1",
               val_size=0.2,
               tensorboard_dir="") -> None:
    self.batch_size = batch_size
    self.epochs = epochs
    self.random_seed = random_seed
    self.model_config = model_config
    self.val_size = val_size
    self.tensorboard_dir = tensorboard_dir

  @staticmethod
  def construct_tf_dataset(X: np.ndarray,
                           y: np.ndarray,
                           batch_size: int = 32) -> tf.data.Dataset:
    # create dataset
    ds_labels = tf.data.Dataset.from_tensor_slices(y)
    ds_features = tf.data.Dataset.from_tensor_slices(X)

    return tf.data.Dataset.zip((ds_features, ds_labels)) \
        .batch(batch_size) \
        .cache() \
        .prefetch(tf.data.AUTOTUNE)

  def create_model(self, input_shape):
    # get activation
    fact = "relu"
    if self.model_config == "v2":
      fact = "silu"
    elif self.model_config == "v3":
      fact = "gelu"

    # create model
    inputs = tf.keras.layers.Input(input_shape)
    x = tf.keras.layers.Dense(256, activation=fact)(inputs)
    x = tf.keras.layers.Dense(256, activation=fact)(x)
    x = tf.keras.layers.Dense(128, activation=fact)(x)
    x = tf.keras.layers.Dense(64, activation=fact)(x)
    outputs = tf.keras.layers.Dense(1)(x)

    # create model
    self.model = tf.keras.Model(inputs=inputs, outputs=outputs)

    # compile model
    self.model.compile(
        optimizer="adam", loss="mean_squared_error", metrics=["mae", "mse"])

  def predict(self, X):
    return self.model.predict(X)

  def fit(self, X, y):
    # split train into train and validation
    X_train, X_val, y_train, y_val = train_test_split(
        X, y, test_size=self.val_size, random_state=self.random_seed)

    # create dataset
    train_ds = CustomTensorFlowRegressor.construct_tf_dataset(
        X_train, y_train, self.batch_size)
    val_ds = CustomTensorFlowRegressor.construct_tf_dataset(
        X_val, y_val, self.batch_size)

    # create model
    self.create_model((X.shape[1],))

    # create early stopping callback
    callbacks = [
        tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=3)
    ]

    # check if tensorboard dir is set
    if self.tensorboard_dir:
      callbacks.append(
          tf.keras.callbacks.TensorBoard(log_dir=self.tensorboard_dir))

    # train model
    return self.model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=self.epochs,
        callbacks=callbacks,
        verbose=0)
