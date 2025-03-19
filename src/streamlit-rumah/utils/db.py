import json
import uuid

import streamlit as st

from sqlalchemy import create_engine
from sqlalchemy.orm import Session

from utils.config import get_settings
from utils.models import (
    UserSession,
    Prediction,
    OpenAIPrompt,
    OpenAIResponse,
    OpenAIUsage,
    RetrievedDocument,
)

USAGE_TYPE_ROUTER = "router"
USAGE_TYPE_PREDICTION_EXTRACTION = "llm_prediction_extraction"
USAGE_TYPE_PREDICTION_PARAPHRASE = "llm_prediction_paraphrase"
USAGE_TYPE_RAG_EMBEDDING = "llm_rag_embedding"
USAGE_TYPE_RAG_RESPONSE = "llm_rag_response"


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
        user_session = (
            tx.query(UserSession).filter_by(session_token=_session_token).first()
        )

        # create user session if not exists
        if not user_session:
            user_session = UserSession(
                id=str(uuid.uuid4()), session_token=_session_token
            )

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
        prediction = Prediction(
            id=str(uuid.uuid4()),
            request=json.dumps(_request),
            predicted=_predicted,
            session_id=user_session,
        )

        tx.add(prediction)
        tx.commit()


def track_prompt(_session_token: str, _prompt: str, _response: dict):
    # get connection
    engine = get_connection()

    # start transaction
    with Session(engine) as tx:
        # get user session
        session_id = track_user_session(_session_token)

        # phase 1: track prompt usage
        prompt_id = str(uuid.uuid4())
        tx.add(OpenAIPrompt(id=prompt_id, prompt=_prompt, session_id=session_id))

        # phase 2: track router usage
        tx.add(
            create_openai_usage(
                USAGE_TYPE_ROUTER, _response["router_llm"]["meta"][0], prompt_id
            )
        )

        # phase 3: track prediction or RAG usage
        if "prediction_llm" in _response:
            # phase 3a: track prediction pipeline
            tx.add_all(
                [
                    # phase 3a.1: prediction feature extraction
                    create_openai_usage(
                        USAGE_TYPE_PREDICTION_EXTRACTION,
                        _response["prediction_llm"]["meta"][0],
                        prompt_id,
                    ),
                    # phase 3a.2: paraphrase response
                    create_openai_response(
                        _response["prediction_result_llm"]["replies"][0], prompt_id
                    ),
                    create_openai_usage(
                        USAGE_TYPE_PREDICTION_PARAPHRASE,
                        _response["prediction_result_llm"]["meta"][0],
                        prompt_id,
                    ),
                ]
            )
        else:
            # phase 3b: track RAG usage
            tx.add_all(
                [
                    # phase 3b.1: question embedding
                    create_openai_usage(
                        USAGE_TYPE_RAG_EMBEDDING,
                        _response["rag_embedder"]["meta"],
                        prompt_id,
                    ),
                    # phase 3b.2: returned documents
                    *create_rag_documents(
                        _response["rag_doc_returner"]["documents"], prompt_id
                    ),
                    # phase 3b.3: QNA response
                    create_openai_response(
                        _response["rag_llm"]["replies"][0], prompt_id
                    ),
                    create_openai_usage(
                        USAGE_TYPE_RAG_RESPONSE,
                        _response["rag_llm"]["meta"][0],
                        prompt_id,
                    ),
                ]
            )

        # commit transaction
        tx.commit()


def create_openai_response(contents: str, prompt_id: str):
    return OpenAIResponse(id=str(uuid.uuid4()), contents=contents, prompt_id=prompt_id)


def create_openai_usage(usage_type: str, meta: dict[str, dict], prompt_id: str):
    usage = meta["usage"]
    return OpenAIUsage(
        id=str(uuid.uuid4()),
        usage_type=usage_type,
        model=meta["model"],
        total_tokens=usage.get("total_tokens", 0),
        prompt_tokens=usage.get("prompt_tokens", 0),
        completion_tokens=usage.get("completion_tokens", 0),
        prompt_id=prompt_id,
    )


def create_rag_documents(documents: list[dict], prompt_id: str):
    return [
        RetrievedDocument(
            id=str(uuid.uuid4()),
            city=doc["city"],
            district=doc["district"],
            price=doc["price"],
            document_id=doc["id"],
            prompt_id=prompt_id,
        )
        for doc in documents
    ]
