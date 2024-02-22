from dataclasses import dataclass

import streamlit as st
import extra_streamlit_components as stx

from utils.db import track_prompt
from utils.cookies import ensure_user_has_session, get_session_id
from utils.llm import query, get_rag_pipeline, get_document_store


@dataclass
class HouseRecord:
    city: str
    district: str
    price: float
    url: str
    main_image_url: str


@dataclass
class ChatRecord:
    role: str
    content: str
    results: list[HouseRecord]


MAX_MESSAGE_LENGTH = 500


@st.cache_resource()
def load_css():
    with open("assets/style.css") as f:
        return f.read()


def render_chat(message: ChatRecord):
    with st.chat_message(message.role):
        # if the LLM responded with no data, dont' render
        content_lower = message.content.lower()
        if "no_res" in content_lower or "tidak ada hasil" in content_lower:
            st.markdown(
                "Tidak ada hasil yang cocok dengan pencarian Anda. Silakan coba lagi."
            )
            return

        # render content
        st.markdown(message.content)

        if len(message.results) > 0:
            # create columns with a maximum of 5 results per column
            # if there are more than 5 results, the columns will be created in a new row
            for i, result in enumerate(message.results):
                # create a new row of columns every 5 results
                if i % 5 == 0:
                    cols = st.columns(5)

                # format price
                price = f"Rp{result.price:,.0f}jt" if result.price < 1000 else f"Rp{(result.price / 1000):,.0f}m"
                with cols[i % 5]:
                    st.image(result.main_image_url)
                    st.markdown(
                        f'<div class="text-caption">{price}<br><a href="{result.url}">{result.district}, {result.city}</a></div>',
                        unsafe_allow_html=True)


def main():
    #
    # Page configuration
    #

    st.set_page_config(
        page_title="Tanya AI - NyariRumah",
        page_icon="👋",
    )

    # to prevent http referer
    st.markdown("<meta name='referrer' content='no-referrer'>", unsafe_allow_html=True)

    # set cookie manager
    cookie_manager = stx.CookieManager()
    ensure_user_has_session(cookie_manager)

    # load custom styles
    st.markdown(f"<style>{load_css()}</style>", unsafe_allow_html=True)

    #
    # Page contents
    #

    st.title("Tanya AI")

    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        render_chat(message)

    # react to user input
    if prompt := st.chat_input("What is up?"):
        # display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # add user message to chat history
        st.session_state.messages.append(
            ChatRecord(role="user", content=prompt, results=[]))

        # check if prompt is too long
        if len(prompt) > MAX_MESSAGE_LENGTH:
            response = "Pertanyaan terlalu panjang, maksimal 500 karakter. Silakan coba lagi."
            st.session_state.messages.append(
                ChatRecord(role="assistant", content=response, results=[]))

            with st.chat_message("assistant"):
                st.markdown(response)

            return

        # query to LLM
        with st.spinner("Thinking..."):
            # get pipeline and document store
            doc_store = get_document_store()
            pipeline = get_rag_pipeline(doc_store)

            # query LLM
            result = query(pipeline, prompt)
            response = result["llm"]["replies"][0]

            # create record
            house_records = [
                HouseRecord(city=doc["city"],
                            district=doc["district"],
                            price=doc["price"],
                            url=doc["url"],
                            main_image_url=doc["main_image_url"])
                for doc in result["return_docs"]["documents"]
            ]

            chat_record = ChatRecord(role="assistant",
                                     content=response,
                                     results=house_records)

            # display assistant response in chat message container
            render_chat(chat_record)

            # add assistant response to chat history
            st.session_state.messages.append(chat_record)

            # track prompt
            track_prompt(get_session_id(cookie_manager), prompt, result)


if __name__ == "__main__":
    # initialize chat history
    if "messages" not in st.session_state:
        st.session_state.messages = []

    # bootstrap
    main()
