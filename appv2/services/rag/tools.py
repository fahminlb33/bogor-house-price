import os
from dataclasses import dataclass

import logging

logging.basicConfig(
    format="%(levelname)s - %(name)s -  %(message)s", level=logging.WARNING)
logging.getLogger("haystack").setLevel(logging.INFO)

from haystack import Pipeline, Document, component
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from utils.shared import get_settings, get_logger
from services.image_search import searcher
from services.price_predictor import predictor
from services.rag.prompts import PROMPT_FOR_RAG


@component
class ReturnDocumentsFromRetriever:

  @component.output_types(documents=list[dict])
  def run(self, docs: list[Document]):
    return {"documents": [{"id": doc.id, **doc.meta} for doc in docs]}


class BaseTool:

  def __init__(self) -> None:
    self.settings = get_settings()

  def get_tool_schema(self) -> dict:
    pass


@dataclass
class OpenAIUsage:
  prompt_tokens: int
  completion_tokens: int

  @staticmethod
  def from_dict(d: dict) -> "OpenAIUsage":
    return OpenAIUsage(
        prompt_tokens=d["usage"].get("prompt_tokens", 0),
        completion_tokens=d["usage"].get("completion_tokens", 0),
    )


@dataclass
class HouseDocument:
  id: str
  city: str
  district: str
  price: float
  url: str
  image_url: str


@dataclass
class HouseRecommendationResult:
  success: bool
  reply: str
  documents: list[HouseDocument]


class HouseRecommendationTool(BaseTool):

  tool_name = "house_recommendation_by_text"

  def __init__(self) -> None:
    super().__init__()
    self.logger = get_logger("HouseRecommendationTool")

    # create document store
    qdrant = QdrantDocumentStore(
        host=self.settings.QDRANT_HOST,
        port=self.settings.QDRANT_PORT,
        grpc_port=self.settings.QDRANT_GRPC_PORT,
        prefer_grpc=True,
        index=self.settings.QDRANT_DOCUMENT_INDEX,
        embedding_dim=self.settings.QDRANT_DOCUMENT_EMBEDDING_DIM,
        wait_result_from_api=True,
        hnsw_config=dict(
            m=self.settings.QDRANT_HNSW_M,
            ef_construct=self.settings.QDRANT_HNSW_EF),
    )

    # create pipeline
    self.pipeline = Pipeline()

    # create components
    self.pipeline.add_component(
        "embedder",
        OpenAITextEmbedder(model=self.settings.OPENAI_EMBEDDING_MODEL))
    self.pipeline.add_component(
        "retriever",
        QdrantEmbeddingRetriever(
            document_store=qdrant, top_k=self.settings.QDRANT_TOP_K))
    self.pipeline.add_component("prompt",
                                PromptBuilder(template=PROMPT_FOR_RAG))
    self.pipeline.add_component(
        "llm", OpenAIGenerator(model=self.settings.OPENAI_CHAT_MODEl))
    self.pipeline.add_component("docs", ReturnDocumentsFromRetriever())

    # connect the components
    self.pipeline.connect("embedder.embedding", "retriever.query_embedding")
    self.pipeline.connect("retriever", "prompt.documents")
    self.pipeline.connect("retriever", "docs")  # output documents used in RAG
    self.pipeline.connect("prompt", "llm")  # output replies from RAG

  def __call__(self,
               query: str) -> tuple[HouseRecommendationResult, OpenAIUsage]:
    # run the pipeline
    self.logger.debug("Running text RAG pipeline...")
    pipe_res = self.pipeline.run(
        {
            "embedder": {
                "text": query
            },
            "prompt": {
                "question": query
            }
        },
        debug=self.settings.DEBUG)

    # create query documents
    documents = []
    for doc in pipe_res["docs"]["documents"]:
      # insert the document
      documents.append(
          HouseDocument(
              id=doc["id"],
              city=doc["city"],
              district=doc["district"],
              price=doc["price"],
              url=doc["url"],
              image_url=doc["main_image_url"]))

    # sum the usage
    usage_embed = OpenAIUsage.from_dict(pipe_res["embedder"]["meta"])
    usage_generation = OpenAIUsage.from_dict(pipe_res["llm"]["meta"][0])
    total_usage = OpenAIUsage(
        prompt_tokens=usage_embed.prompt_tokens +
        usage_generation.prompt_tokens,
        completion_tokens=usage_embed.completion_tokens +
        usage_generation.completion_tokens)

    # parse the result
    pipe_res = HouseRecommendationResult(
        success=True, reply=pipe_res["llm"]["replies"][0], documents=documents)

    return pipe_res, total_usage

  def get_tool_schema(self) -> dict:
    return {
        "type": "function",
        "function": {
            "name":
                self.tool_name,
            "description":
                "This tool provides house recommendations based on user preferences such as location, number of bedrooms, bathrooms, and maximum price. It searches our database for suitable options matching the provided criteria.",
            "parameters": {
                "type": "object",
                "required": ["query"],
                "properties": {
                    "query": {
                        "type":
                            "string",
                        "description":
                            "The query to use in the house search to the database. Infer this from the user's message. It should be a question or a statement.",
                    }
                },
            },
        },
    }


class HouseImageSearchTool(BaseTool):

  tool_name = "house_recommendation_by_image"

  def __init__(self) -> None:
    super().__init__()
    self.logger = get_logger("HouseImageSearchTool")

    # create pipeline
    self.pipeline = Pipeline()

    # create components
    self.pipeline.add_component("prompt",
                                PromptBuilder(template=PROMPT_FOR_RAG))
    self.pipeline.add_component(
        "llm", OpenAIGenerator(model=self.settings.OPENAI_CHAT_MODEl))

    # connect the components
    self.pipeline.connect("prompt", "llm")

  def __call__(self,
               file_name: str) -> tuple[HouseRecommendationResult, OpenAIUsage]:
    # check if path is safe
    path_save, path_real = self.make_sure_path_safe(file_name)
    if not path_save:
      result = HouseRecommendationResult(
          success=False, reply="Invalid path", documents=[])
      usage = OpenAIUsage(prompt_tokens=0, completion_tokens=0)

      return result, usage

    self.logger.error("Running image RAG pipeline...")

    # find images
    pipe_documents = searcher.search(path_real)

    # run the pipeline
    pipe_result = self.pipeline.run({
        "prompt": {
            "question":
                "I want a house recommendation based on the provided context. You must answer in Bahasa Indonesia.",
            "documents":
                pipe_documents
        }
    })

    # create query documents
    documents = []
    for doc in pipe_documents:
      # insert the document
      documents.append(
          HouseDocument(
              id=doc.meta["id"],
              city=doc.meta["city"],
              district=doc.meta["district"],
              price=doc.meta["price"],
              url=doc.meta["url"],
              image_url=doc.meta["main_image_url"]))

    # parse the result
    result = HouseRecommendationResult(
        success=True,
        reply=pipe_result["llm"]["replies"][0],
        documents=documents)
    usage = OpenAIUsage.from_dict(pipe_result["llm"]["meta"][0])

    return result, usage

  def make_sure_path_safe(self, file_name: str) -> tuple[bool, str]:
    # get real path
    real_path = os.path.realpath(
        os.path.join(self.settings.UPLOAD_DIR, file_name))
    upload_dir = os.path.realpath(self.settings.UPLOAD_DIR)

    # check if the path is inside the upload directory
    if os.path.commonprefix([real_path, upload_dir]) != upload_dir:
      return False, "Invalid path"

    return True, real_path

  def get_tool_schema(self) -> dict:
    return {
        "type": "function",
        "function": {
            "name":
                self.tool_name,
            "description":
                "This tool analyzes an uploaded image of a house specified by its file name and provides recommendations or information about similar properties.",
            "parameters": {
                "type": "object",
                "required": ["file_name"],
                "properties": {
                    "file_name": {
                        "type":
                            "string",
                        "description":
                            "The file name of the image of the house.",
                    }
                },
            },
        },
    }


@dataclass
class HousePricePredictionResult:
  success: bool
  prediction: float
  land_area: float
  house_size: float
  bedrooms: int
  bathrooms: int


class HousePricePredictionTool:

  tool_name = "house_price_prediction"

  def __init__(self) -> None:
    self.logger = get_logger("HousePricePredictionTool")

  def __call__(self,
               **kwargs) -> tuple[HousePricePredictionResult, OpenAIUsage]:
    try:
      # construct features
      X_pred = predictor.construct_features({
          "luas_tanah": kwargs.get("land_area", 0),
          "luas_bangunan": kwargs.get("house_size", 0),
          "kamar_tidur": kwargs.get("bedrooms", 0),
          "kamar_mandi": kwargs.get("bathrooms", 0),
      })

      # predict
      y_pred = predictor.predict(X_pred)

      # return prediction
      result = HousePricePredictionResult(
          success=True,
          prediction=y_pred[0] * 1_000_000,
          land_area=kwargs.get("land_area", 0),
          house_size=kwargs.get("house_size", 0),
          bedrooms=kwargs.get("bedrooms", 0),
          bathrooms=kwargs.get("bathrooms", 0))
      usage = OpenAIUsage(prompt_tokens=0, completion_tokens=0)

      return result, usage
    except Exception as e:
      self.logger.error(
          "Error when predicting house price in LLM component", exc_info=e)
      result = HousePricePredictionResult(
          success=False,
          prediction=0,
          land_area=kwargs.get("land_area", 0),
          house_size=kwargs.get("house_size", 0),
          bedrooms=kwargs.get("bedrooms", 0),
          bathrooms=kwargs.get("bathrooms", 0))
      usage = OpenAIUsage(prompt_tokens=0, completion_tokens=0)

      return result, usage

  def get_tool_schema(self) -> dict:
    return {
        "type": "function",
        "function": {
            "name":
                self.tool_name,
            "description":
                "This tool predicts the price of a house based on its specifications which are land area, building area, number of bedrooms, and number of bathrooms",
            "parameters": {
                "type": "object",
                "required": ["land_area"],
                "properties": {
                    "land_area": {
                        "type":
                            "number",
                        "description":
                            "The area of land the house is built on in meter squared. Infer this from the user's message. It should be a number.",
                    },
                    "house_size": {
                        "type":
                            "number",
                        "description":
                            "The area of the house in meter squared. Infer this from the user's message. It should be a number.",
                    },
                    "bedrooms": {
                        "type":
                            "integer",
                        "description":
                            "The number of bedrooms. Infer this from the user's message. It should be a number.",
                    },
                    "bathrooms": {
                        "type":
                            "integer",
                        "description":
                            "The number of bathrooms. Infer this from the user's message. It should be a number.",
                    }
                },
            },
        },
    }
