import pandas as pd

from catboost import CatBoostRegressor

from appv2.utils.shared import get_settings, get_logger

AVAILABLE_FACILITIES = [
    "AC", "Keamanan", "Laundry", "Masjid", 'Ruang Makan', 'Ruang Tamu'
]

AVAILABLE_HOUSE_MATERIAL = ["Bata Merah", "Bata Hebel"]

AVAILABLE_TAGS = ["Cash Bertahap", "KPR", "Komplek", "Perumahan"]


class PricePredictor:

  def __init__(self) -> None:
    # get config
    self.settings = get_settings()
    self.logger = get_logger("PricePredictor")

    # initialize model
    self.logger.info("Loading model...")
    self.clf = CatBoostRegressor()
    self.clf.load_model(self.settings.CATBOOST_PREDICTION_MODEL)

  @staticmethod
  def safe_get(d: dict, col: str, default_value):
    val = d.get(col, default_value)
    if type(val) is str and len(val) == 0:
      return default_value

    return val

  def predict(self, X):
    self.logger.debug("Running inference...")
    return self.clf.predict(X)

  def construct_features(self, input_features: dict) -> pd.DataFrame:
    self.logger.debug("Constructing features...")

    features = {
        "luas_tanah":
            PricePredictor.safe_get(input_features, "luas_tanah", 0),
        "luas_bangunan":
            PricePredictor.safe_get(input_features, "luas_bangunan", 0),
        "kamar_tidur":
            PricePredictor.safe_get(input_features, "kamar_tidur", 0),
        "kamar_mandi":
            PricePredictor.safe_get(input_features, "kamar_mandi", 0),
        "kamar_pembantu":
            PricePredictor.safe_get(input_features, "kamar_pembantu", 0),
        "kamar_mandi_pembantu":
            PricePredictor.safe_get(input_features, "kamar_mandi_pembantu", 0),
        "daya_listrik":
            PricePredictor.safe_get(input_features, "daya_listrik", 0),
        "jumlah_lantai":
            PricePredictor.safe_get(input_features, "jumlah_lantai", 0),
        "lebar_jalan":
            PricePredictor.safe_get(input_features, "lebar_jalan", 0),
        "carport":
            PricePredictor.safe_get(input_features, "carport", 0),
        "dapur":
            PricePredictor.safe_get(input_features, "dapur", 0),
        "ruang_makan":
            1 if "Ruang Makan" in PricePredictor.safe_get(
                input_features, "fasilitas", []) else 0,
        "ruang_tamu":
            1 if "Ruang Tamu" in PricePredictor.safe_get(
                input_features, "fasilitas", []) else 0,
    }

    for facility in AVAILABLE_FACILITIES:
      key = "facility_" + facility.replace(" ", "_").lower()
      features[key] = 1 if facility in PricePredictor.safe_get(
          input_features, "fasilitas", []) else 0

    for material in AVAILABLE_HOUSE_MATERIAL:
      key = "house_mat_" + material.replace(" ", "_").lower()
      features[key] = 1 if material in PricePredictor.safe_get(
          input_features, "house_material", []) else 0

    for tag in AVAILABLE_TAGS:
      key = "tag_" + tag.replace(" ", "_").lower()
      features[key] = 1 if tag in PricePredictor.safe_get(
          input_features, "tags", []) else 0

    features["tahun_dibangun"] = PricePredictor.safe_get(
        input_features, "tahun_dibangun", 0)

    self.logger.debug("Preprocess data finished")
    return pd.DataFrame([features])


predictor = PricePredictor()
