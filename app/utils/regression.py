import pandas as pd
import streamlit as st

from catboost import CatBoostRegressor

AVAILABLE_FACILITIES = [
    "AC", "Keamanan", "Laundry", "Masjid", 'Ruang Makan', 'Ruang Tamu'
]

AVAILABLE_HOUSE_MATERIAL = ["Bata Merah", "Bata Hebel"]

AVAILABLE_TAGS = ["Cash Bertahap", "KPR", "Komplek", "Perumahan"]


@st.cache_resource
def load_model():
    clf = CatBoostRegressor()
    clf.load_model("assets/model/house_price_reg.cbm")

    return clf


def construct_features(input_features: dict) -> pd.DataFrame:
    features = {
        "carport": input_features.get("carport", 0),
        "dapur": input_features.get("dapur", 0),
        "daya_listrik": input_features.get("daya_listrik", 0),
    }

    for facility in AVAILABLE_FACILITIES:
        key = "facility_" + facility.replace(" ", "_").lower()
        features[key] = 1 if facility in input_features.get("fasilitas",
                                                            0) else 0

    for material in AVAILABLE_HOUSE_MATERIAL:
        key = "house_mat_" + material.replace(" ", "_").lower()
        features[key] = 1 if material in input_features.get(
            "house_material", 0) else 0

    features = {
        **features,
        "jumlah_lantai":
            input_features.get("jumlah_lantai", 0),
        "kamar_mandi":
            input_features.get("kamar_mandi", 0),
        "kamar_mandi_pembantu":
            input_features.get("kamar_mandi_pembantu", 0),
        "kamar_pembantu":
            input_features.get("kamar_pembantu", 0),
        "kamar_tidur":
            input_features.get("kamar_tidur", 0),
        "lebar_jalan":
            input_features.get("lebar_jalan", 0),
        "luas_bangunan":
            input_features.get("luas_bangunan", 0),
        "luas_tanah":
            input_features.get("luas_tanah", 0),
        "ruang_makan":
            1 if "Ruang Makan" in input_features.get("fasilitas", 0) else 0,
        "ruang_tamu":
            1 if "Ruang Tamu" in input_features.get("fasilitas", 0) else 0,
    }

    for tag in AVAILABLE_TAGS:
        key = "tag_" + tag.replace(" ", "_").lower()
        features[key] = 1 if tag in input_features.get("tags", 0) else 0

    features["tahun_dibangun"] = input_features.get("tahun_dibangun")

    return pd.DataFrame([features])
