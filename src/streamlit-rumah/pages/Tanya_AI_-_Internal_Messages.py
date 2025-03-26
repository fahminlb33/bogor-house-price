import json

import pandas as pd
import streamlit as st


def main():
    #
    # Page configuration
    #

    st.set_page_config(
        page_title="Tanya AI - NyariRumah",
        page_icon="👋",
    )

    st.sidebar.markdown(
        """
        Data internal riwayat *chat* yang dilakukan oleh Gemini pada menu **Tanya AI**.
        
        Source code: [klik disini.](https://github.com/fahminlb33/bogor-house-price/blob/master/src/streamlit-rumah/pages/Tanya_AI_-_Internal_Messages.py)
        """
    )

    #
    # Page contents
    #

    st.title("🤖Internal Messages")
    st.markdown("""
                Lihat apa saja yang dilakukan oleh Gemini untuk menjawab pertanyaan Anda.
                """)

    # format chat history
    data = []
    history = st.session_state.chat.get_history()

    for item in history:
        for part in item.parts:
            if part.function_call is not None:
                data.append(
                    {
                        "role": item.role,
                        "text": "FUNCTION CALL",
                        "function_name": part.function_call.name,
                        "function_args": json.dumps(part.function_call.args),
                        "function_response": None,
                    }
                )
                continue

            if part.function_response is not None:
                data.append(
                    {
                        "role": item.role,
                        "text": "FUNCTION RESPONSE",
                        "function_name": part.function_response.name,
                        "function_args": None,
                        "function_response": json.dumps(
                            part.function_response.response
                        ),
                    }
                )
                continue

            data.append(
                {
                    "role": item.role,
                    "text": part.text,
                    "function_name": None,
                    "function_args": None,
                    "function_response": None,
                }
            )

    # show data
    df_history = pd.DataFrame(data)
    st.dataframe(df_history, use_container_width=True)
    # st.markdown(df_history.to_html(escape=False), unsafe_allow_html=True)


if __name__ == "__main__":
    # init states
    if "initialized" in st.session_state and st.session_state.initialized:
        # bootstrap
        main()
    else:
        st.markdown(
            "Anda belum mengunjungi halaman Tanya AI.\n\nTidak ada data yang dapat ditampilkan."
        )
