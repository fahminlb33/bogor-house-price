import streamlit as st

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class AppSettings(BaseSettings):
    model_config = SettingsConfigDict(env_file=".env", env_file_encoding="utf-8")

    # app
    debug: bool = Field(default=False)

    # mariadb
    db_host: str = Field()
    db_port: int = Field()
    db_user: str = Field()
    db_password: str = Field()
    db_name: str = Field()

    # qdrant
    qdrant_host: str = Field()
    qdrant_port: int = Field()
    qdrant_grpc_port: int = Field()
    qdrant_index: str = Field()
    qdrant_embedding_dim: int = Field(default=1536)
    qdrant_hnsw_m: int = Field(default=16)
    qdrant_hnsw_ef: int = Field(default=100)

    # llm
    openai_api_key: str = Field()
    openai_chat_model: str = Field(default="gpt-3.5-turbo")
    openai_embedding_model: str = Field(default="text-embedding-3-small")


@st.cache_resource(show_spinner=False)
def get_settings():
    return AppSettings()
