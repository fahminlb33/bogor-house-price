import joblib
from sklearn.pipeline import Pipeline

model: Pipeline = None


def load_model(model_path: str):
    global model
    model = joblib.load(model_path)


def predict(data):
    return model.predict(data)
