import json

from catboost import CatBoostRegressor
from haystack import Document, component

from utils.regression import construct_features


@component
class ReturnDocumentsFromRetriever:

    @component.output_types(documents=list[dict])
    def run(self, docs: list[Document]):
        return {"documents": [{"id": doc.id, **doc.meta} for doc in docs]}


@component
class PredictHousePrice:
    # load model
    clf = CatBoostRegressor()
    clf.load_model("assets/model/house_price_reg.cbm")

    @component.output_types(prediction=float, features=dict)
    def run(self, replies: list[str]):
        try:
            # parse the features
            features = json.loads(replies[0])

            # construct features
            X_pred = construct_features({
                "luas_tanah": features.get("land_area", 0),
                "luas_bangunan": features.get("house_size", 0),
                "kamar_tidur": features.get("bedrooms", 0),
                "kamar_mandi": features.get("bathrooms", 0),
            })

            # predict
            y_pred = self.clf.predict(X_pred)

            # return prediction
            return {
                "prediction": y_pred[0] * 1_000_000,
                "features": {
                    "land_area": features.get("land_area", 0),
                    "house_size": features.get("house_size", 0),
                    "bedrooms": features.get("bedrooms", 0),
                    "bathrooms": features.get("bathrooms", 0),
                }
            }
        except:
            return -1
