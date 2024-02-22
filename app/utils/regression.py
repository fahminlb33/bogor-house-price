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
    clf.load_model("assets/model.cbm")

    return clf


def construct_features(input_features: dict) -> pd.DataFrame:
    features = {
        "carport": input_features["carport"],
        "dapur": input_features["dapur"],
        "daya_listrik": input_features["daya_listrik"],
    }

    for facility in AVAILABLE_FACILITIES:
        key = "facility_" + facility.replace(" ", "_").lower()
        features[key] = 1 if facility in input_features["fasilitas"] else 0

    for material in AVAILABLE_HOUSE_MATERIAL:
        key = "house_mat_" + material.replace(" ", "_").lower()
        features[key] = 1 if material in input_features["house_material"] else 0

    features = {
        **features,
        "jumlah_lantai": input_features["jumlah_lantai"],
        "kamar_mandi": input_features["kamar_mandi"],
        "kamar_mandi_pembantu": input_features["kamar_mandi_pembantu"],
        "kamar_pembantu": input_features["kamar_pembantu"],
        "kamar_tidur": input_features["kamar_tidur"],
        "lebar_jalan": input_features["lebar_jalan"],
        "luas_bangunan": input_features["luas_bangunan"],
        "luas_tanah": input_features["luas_tanah"],
        "ruang_makan": 1 if "Ruang Makan" in input_features["fasilitas"] else 0,
        "ruang_tamu": 1 if "Ruang Tamu" in input_features["fasilitas"] else 0,
    }

    for tag in AVAILABLE_TAGS:
        key = "tag_" + tag.replace(" ", "_").lower()
        features[key] = 1 if tag in input_features["tags"] else 0

    features["tahun_dibangun"] = input_features["tahun_dibangun"]

    return pd.DataFrame([features])
