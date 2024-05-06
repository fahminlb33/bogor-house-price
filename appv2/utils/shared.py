import sys
import logging
from functools import lru_cache

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict
from flask_caching import Cache

__LOGGER = None

cache = Cache()


class AppSettings(BaseSettings):
  model_config = SettingsConfigDict(env_file='.env', env_file_encoding='utf-8')

  # app
  DEBUG: bool = Field(default=False)
  UPLOAD_DIR: str = Field(default="uploads")

  # caching
  CACHE_TYPE: str = Field("FileSystemCache")
  CACHE_DEFAULT_TIMEOUT: int = Field(300)
  CACHE_DIR: str = Field(default="cache")

  # mariadb
  SQLALCHEMY_DATABASE_URI: str = Field()

  # qdrant
  QDRANT_HOST: str = Field()
  QDRANT_PORT: int = Field()
  QDRANT_GRPC_PORT: int = Field()
  QDRANT_TOP_K: int = Field(default=3)
  QDRANT_HNSW_M: int = Field(default=16)
  QDRANT_HNSW_EF: int = Field(default=100)
  
  QDRANT_DOCUMENT_INDEX: str = Field()
  QDRANT_DOCUMENT_EMBEDDING_DIM: int = Field(default=1536)

  QDRANT_IMAGE_INDEX: str = Field()
  QDRANT_IMAGE_EMBEDDING_DIM: int = Field(default=1280)

  # llm
  OPENAI_API_KEY: str = Field()
  OPENAI_CHAT_MODEl: str = Field(default="gpt-3.5-turbo")
  OPENAI_EMBEDDING_MODEL: str = Field(default="text-embedding-3-small")

  # prediction models
  CATBOOST_PREDICTION_MODEL: str = Field(default="assets/house_price_reg.cbm")
  TENSORFLOW_MOBILENET_MODEL: str = Field(default="assets/mobilenet_v3")


def get_logger(name):
  # setup logging
  __LOGGER = logging.getLogger(name)
  __LOGGER.propagate = False

  # create console handler and set level to info
  stdout = logging.StreamHandler(stream=sys.stdout)
  stdout.setLevel(logging.DEBUG)

  # create formatter
  formatter = logging.Formatter(
      "%(name)s: %(asctime)s | %(levelname)s | %(message)s")
  stdout.setFormatter(formatter)

  # add handler
  __LOGGER.addHandler(stdout)

  # set level to info
  __LOGGER.setLevel(logging.DEBUG)

  return __LOGGER


@lru_cache
def get_settings():
  return AppSettings()
