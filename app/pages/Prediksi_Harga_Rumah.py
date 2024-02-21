import pandas as pd
import streamlit as st

from catboost import CatBoostRegressor

AVAILABLE_FACILITIES = ["AC", "Keamanan", "Laundry", "Masjid", 'Ruang Makan', 'Ruang Tamu']
AVAILABLE_HOUSE_MATERIAL = ["Bata Merah", "Bata Hebel"]
AVAILABLE_TAGS = ["Cash Bertahap", "KPR", "Komplek", "Perumahan"]

@st.cache_resource()
def load_css():
    with open("assets/style.css") as f:
        return f.read()

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
        features["facility_" + facility.replace(" ", "_").lower()] = 1 if facility in input_features["fasilitas"] else 0

    for material in AVAILABLE_HOUSE_MATERIAL:
        features["house_mat_" + material.replace(" ", "_").lower()] = 1 if material in input_features["house_material"] else 0

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
        features["tag_" + tag.replace(" ", "_").lower()] = 1 if tag in input_features["tags"] else 0

    features["tahun_dibangun"] = input_features["tahun_dibangun"]

    return pd.DataFrame([features])

def main():
    st.set_page_config(
        page_title="Prediksi Harga Rumah - NyariRumah",
        page_icon="👋",
    )

    st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)


    st.title("🏡Prediksi Harga Rumah")
    st.markdown("""
                Selamat datang ke layanan prediksi harga rumah!
                """)

    # st.image("https://source.unsplash.com/brown-and-black-wooden-house-TiVPTYCG_3E", caption="Sumber: brown and black wooden house oleh vu anh dari Unsplash")

    input_features = {}

    with st.form("predict_form", border=False):
        st.subheader("Spesifikasi Rumah")

        col1, col2, col3 = st.columns(3)
        with col1:
            input_features["luas_tanah"] = st.number_input('Luas tanah', step=1)
            input_features["luas_bangunan"] = st.number_input('Luas bangunan', step=1)
            input_features["daya_listrik"] = st.number_input('Daya listrik', step=1)
            input_features["tahun_dibangun"] = st.number_input('Tahun dibangun', step=1)

        with col2:
            input_features["kamar_mandi"] = st.number_input('Kamar mandi', step=1)
            input_features["kamar_tidur"] = st.number_input('Kamar tidur', step=1)
            input_features["kamar_pembantu"] = st.number_input('Kamar pembantu', step=1)
            input_features["kamar_mandi_pembantu"] = st.number_input('Kamar mandi pembantu', step=1)

        with col3:
            input_features["jumlah_lantai"] = st.number_input('Jumlah lantai', step=1)
            input_features["dapur"] = st.number_input('Dapur', step=1)
            input_features["lebar_jalan"] = st.number_input('Lebar jalan (mobil)', step=1)
            input_features["carport"] = st.number_input('Carport', step=1)

        st.subheader("Fasilitas dan Lainnya")
        input_features["fasilitas"] = st.multiselect('Fasilitas', AVAILABLE_FACILITIES)
        input_features["house_material"] = st.multiselect('Material Bangunan', AVAILABLE_HOUSE_MATERIAL)
        input_features["tags"] = st.multiselect('Tags', AVAILABLE_TAGS)

        st.text("")
        submit_res = st.form_submit_button("Prediksi",
                                        type="primary",
                                        use_container_width=True)

    st.info("Ingat, model ini memiliki nilai *Mean Absolute Error* (MAE) sebesar ~1.242.050", icon="💸")

    if submit_res:
        model = load_model()
        X_pred = construct_features(input_features)
        y_pred = model.predict(X_pred)

        price = y_pred[0] * 1_000_000
        price = f"{price:,.0f}".replace(",", ".")

        st.markdown('<div class="text-center">Hasil Prediksi', unsafe_allow_html=True)
        st.markdown(f'<div class="text-center text-prediction">Rp{price}</div>', unsafe_allow_html=True)

if __name__ == "__main__":
    main()
