import streamlit as st
import extra_streamlit_components as stx

from utils.db import track_prediction
from utils.data_loaders import format_price
from utils.cookies import ensure_user_has_session, get_session_id
from utils.regression import (AVAILABLE_FACILITIES, AVAILABLE_HOUSE_MATERIAL,
                              AVAILABLE_TAGS, construct_features, load_model)


@st.cache_resource()
def load_css():
    with open("assets/style.css") as f:
        return f.read()


def main():
    #
    # Page configuration
    #

    # set page config
    st.set_page_config(
        page_title="Prediksi Harga Rumah - NyariRumah",
        page_icon="👋",
    )

    # set cookie manager
    cookie_manager = stx.CookieManager()
    ensure_user_has_session(cookie_manager)

    # load custom styles
    st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

    #
    # Page contents
    #

    st.title("🏡Prediksi Harga Rumah")
    st.markdown("""
                Selamat datang ke layanan prediksi harga rumah! Masukkan informasi spesifikasi rumah yang ingin Anda beli pada form berikut.
                """)

    #
    # Input features
    #

    input_features = {}
    with st.form("predict_form", border=False):
        st.subheader("Spesifikasi Rumah")

        col1, col2, col3 = st.columns(3)
        with col1:
            input_features["luas_tanah"] = st.number_input('Luas tanah', step=1)
            input_features["luas_bangunan"] = st.number_input('Luas bangunan',
                                                              step=1)
            input_features["daya_listrik"] = st.number_input('Daya listrik',
                                                             step=1)
            input_features["tahun_dibangun"] = st.number_input('Tahun dibangun',
                                                               step=1)

        with col2:
            input_features["kamar_mandi"] = st.number_input('Kamar mandi',
                                                            step=1)
            input_features["kamar_tidur"] = st.number_input('Kamar tidur',
                                                            step=1)
            input_features["kamar_pembantu"] = st.number_input('Kamar pembantu',
                                                               step=1)
            input_features["kamar_mandi_pembantu"] = st.number_input(
                'Kamar mandi pembantu', step=1)

        with col3:
            input_features["jumlah_lantai"] = st.number_input('Jumlah lantai',
                                                              step=1)
            input_features["dapur"] = st.number_input('Dapur', step=1)
            input_features["lebar_jalan"] = st.number_input(
                'Lebar jalan (mobil)', step=1)
            input_features["carport"] = st.number_input('Carport', step=1)

        st.subheader("Fasilitas dan Lainnya")
        input_features["fasilitas"] = st.multiselect('Fasilitas',
                                                     AVAILABLE_FACILITIES)
        input_features["house_material"] = st.multiselect(
            'Material Bangunan', AVAILABLE_HOUSE_MATERIAL)
        input_features["tags"] = st.multiselect('Tags', AVAILABLE_TAGS)

        st.text("")
        submit_res = st.form_submit_button("Prediksi",
                                           type="primary",
                                           use_container_width=True)

    #
    # Prediction output
    #

    st.info(
        "Ingat, model ini memiliki nilai *Mean Absolute Error* (MAE) sebesar ~Rp278jt",
        icon="💸")

    if submit_res:
        # load model
        model = load_model()

        # construct features
        X_pred = construct_features(input_features)

        # predict
        y_pred = model.predict(X_pred)
        price = format_price(y_pred[0] * 1_000_000)

        # show prediction
        st.markdown('<div class="text-center">Hasil Prediksi',
                    unsafe_allow_html=True)
        st.markdown(f'<div class="text-center text-prediction">{price}</div>',
                    unsafe_allow_html=True)

        # track prediction
        track_prediction(get_session_id(cookie_manager), input_features,
                         y_pred[0])


if __name__ == "__main__":
    # bootstrap
    main()
