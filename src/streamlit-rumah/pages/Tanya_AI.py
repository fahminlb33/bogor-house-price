from uuid import uuid4
from random import choice
from pathlib import Path
from dataclasses import dataclass

import streamlit as st

from google import genai
from google.genai import types

from utils.llm import (
    SYSTEM_INSTRUCTION,
    top_listing_by_location,
    search_by_keyword,
    search_by_image_id,
    get_house_images,
    get_available_subdistricts,
    predict_house_price,
)


@dataclass
class ChatRecord:
    role: str
    content: str
    image_path: str


MAX_MESSAGE_LENGTH = 500


def generate_response(text: str) -> ChatRecord:
    # send message
    response = st.session_state.chat.send_message(text)

    # check if we're responding with images
    for history in response.automatic_function_calling_history:
        for part in history.parts:
            if part.function_response is None:
                continue

            func_resp = part.function_response
            if func_resp.name == "get_house_images":
                image_paths = func_resp.response["result"]
                if len(image_paths) == 0:
                    return ChatRecord("assistant", response.text, image_path=None)

                return ChatRecord(
                    "assistant",
                    "Here are a sample image for the property",
                    image_path=choice(image_paths),
                )

    return ChatRecord("assistant", response.text, image_path=None)


def render_chat(message: ChatRecord):
    with st.chat_message(message.role):
        # render content
        st.markdown(message.content)

        # render results
        if message.image_path is not None:
            st.image(Path(st.secrets["IMAGE_ROOT"]) / message.image_path)


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
        Chat dilakukan menggunakan Gemini 2.0 Flash dan tool calling.
        
        Source code: [klik disini.](https://github.com/fahminlb33/bogor-house-price/blob/master/src/streamlit-rumah/pages/Tanya_AI.py)
        """
    )

    #
    # Page contents
    #

    # display chat messages from history on app rerun
    for message in st.session_state.messages:
        render_chat(message)

    # react to user input
    if prompt := st.chat_input("Mau cari apa?"):
        # display user message in chat message container
        st.chat_message("user").markdown(prompt)

        # add user message to chat history
        st.session_state.messages.append(
            ChatRecord(role="user", content=prompt, image_path=None)
        )

        # check if prompt is too long
        if len(prompt) > MAX_MESSAGE_LENGTH:
            response = "Pertanyaan terlalu panjang, maksimal 500 karakter."
            st.session_state.messages.append(
                ChatRecord(role="assistant", content=response, results=[])
            )

            with st.chat_message("assistant"):
                st.markdown(response)

            return

        # query to LLM
        with st.spinner("Sedang berpikir..."):
            # query and parse LLM response
            response = generate_response(prompt)

            # display assistant response in chat message container
            render_chat(response)

            # add assistant response to chat history
            st.session_state.messages.append(response)


if __name__ == "__main__":
    # init states
    if "initialized" not in st.session_state or not st.session_state.initialized:
        client = genai.Client(api_key=st.secrets["GEMINI_API_KEY"])
        config = types.GenerateContentConfig(
            system_instruction=SYSTEM_INSTRUCTION,
            tools=[
                top_listing_by_location,
                search_by_keyword,
                search_by_image_id,
                get_house_images,
                get_available_subdistricts,
                predict_house_price,
            ],
        )

        st.session_state.initialized = True
        st.session_state.messages = []
        st.session_state.chat = client.chats.create(
            model=st.secrets["GEMINI_MODEL"], config=config
        )

    # bootstrap
    main()
