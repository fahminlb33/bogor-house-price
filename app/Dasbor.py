import streamlit as st
import extra_streamlit_components as stx

from utils.cookies import ensure_user_has_session


def main():
    #
    # Page configuration
    #

    # set page config
    st.set_page_config(
        page_title="Dasbor - NyariRumah",
        page_icon="👋",
    )

    # set cookie manager
    cookie_manager = stx.CookieManager()
    ensure_user_has_session(cookie_manager)

    #
    # Page contents
    #

    st.write("Hello world")


if __name__ == "__main__":
    # bootstrap
    main()
