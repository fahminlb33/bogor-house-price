import json
import uuid

import streamlit as st

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from utils.config import get_settings
from utils.models import (UserSession, Prediction, OpenAIPrompt, OpenAIResponse,
                          OpenAIUsage, RetrievedDocument)


@st.cache_resource(show_spinner=False)
def get_connection():
    # get settings from env
    settings = get_settings()

    # build dsn
    dsn = f"mysql+pymysql://{settings.db_user}:{settings.db_password}@{settings.db_host}:{settings.db_port}/{settings.db_name}?charset=utf8mb4"

    # create connection
    engine = create_engine(dsn, echo=settings.debug)

    return engine


def track_user_session(_session_token: str) -> str:
    # get connection
    engine = get_connection()

    # start transaction
    with Session(engine) as tx:
        # get user session
        user_session = tx.query(UserSession) \
            .filter_by(session_token=_session_token) \
            .first()

        # create user session if not exists
        if not user_session:
            user_session = UserSession(id=str(uuid.uuid4()),
                                       session_token=_session_token)

            tx.add(user_session)

        tx.commit()

        return user_session.id


def track_prediction(_session_token: str, _request: dict, _predicted: float):
    # get connection
    engine = get_connection()

    # start transaction
    with Session(engine) as tx:
        # get user session
        user_session = track_user_session(_session_token)

        # save prediction
        prediction = Prediction(id=str(uuid.uuid4()),
                                request=json.dumps(_request),
                                predicted=_predicted,
                                session_id=user_session)

        tx.add(prediction)
        tx.commit()


def track_prompt(_session_token: str, _prompt: str, _response: dict):
    # get connection
    engine = get_connection()

    # start transaction
    with Session(engine) as tx:
        # get user session
        user_session = track_user_session(_session_token)

        # save prompt and usage
        openai_prompt = OpenAIPrompt(id=str(uuid.uuid4()),
                                     prompt=_prompt,
                                     session_id=user_session)

        embedder_meta = _response["embedder"]["meta"]
        embed_usage = OpenAIUsage(
            id=str(uuid.uuid4()),
            usage_type="embedding",
            model=embedder_meta["model"],
            prompt_tokens=embedder_meta["usage"]["prompt_tokens"],
            completion_tokens=0,
            total_tokens=embedder_meta["usage"]["total_tokens"],
            prompt_id=openai_prompt.id)

        tx.add_all([openai_prompt, embed_usage])

        # save returned documents from RAG
        for doc in _response["return_docs"]["documents"]:
            # save RAG usage
            rag_usage = RetrievedDocument(id=str(uuid.uuid4()),
                                          city=doc["city"],
                                          district=doc["district"],
                                          price=doc["price"],
                                          document_id=doc["id"],
                                          prompt_id=openai_prompt.id)

            tx.add(rag_usage)

        # save LLM responses
        for reply, meta in zip(_response["llm"]["replies"],
                               _response["llm"]["meta"]):
            # save OpenAI usage
            gen_response = OpenAIResponse(id=str(uuid.uuid4()),
                                          contents=reply,
                                          prompt_id=openai_prompt.id)

            # save OpenAI usage
            gen_usage = OpenAIUsage(
                id=str(uuid.uuid4()),
                usage_type="generation",
                model=meta["model"],
                prompt_tokens=meta["usage"]["prompt_tokens"],
                completion_tokens=meta["usage"]["completion_tokens"],
                total_tokens=meta["usage"]["total_tokens"],
                prompt_id=openai_prompt.id,
                response_id=gen_response.id)

            tx.add_all([gen_response, gen_usage])

        tx.commit()
