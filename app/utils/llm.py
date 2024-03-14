from dataclasses import dataclass

import streamlit as st

from haystack import Pipeline
from haystack.components.routers import ConditionalRouter
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from utils.config import get_settings
from utils.llm_prompts import (PROMPT_FOR_ROUTER,
                               PROMPT_FOR_PREDICTION_EXTRACTION,
                               PROMPT_FOR_PREDICTION_RESULT, PROMPT_FOR_RAG)
from utils.llm_components import (PredictHousePrice,
                                  ReturnDocumentsFromRetriever)


@dataclass
class QueryDocument:
    id: str
    city: str
    district: str
    price: float
    url: str
    main_image_url: str


@dataclass
class QueryResult:
    success: bool
    content: str
    documents: list[QueryDocument]
    raw: dict


AGENT_ROUTER = [{
    "condition": "{{'PREDICTION' in replies[0]}}",
    "output": "{{query}}",
    "output_name": "features",
    "output_type": str,
}, {
    "condition": "{{'PREDICTION' not in replies[0]}}",
    "output": "{{query}}",
    "output_name": "question",
    "output_type": str,
}]


@st.cache_resource(show_spinner=False)
def get_document_store() -> QdrantDocumentStore:
    # get settings
    settings = get_settings()

    # create document store
    hnsw_config = dict(m=settings.qdrant_hnsw_m,
                       ef_construct=settings.qdrant_hnsw_ef)
    return QdrantDocumentStore(host=settings.qdrant_host,
                               port=settings.qdrant_port,
                               grpc_port=settings.qdrant_grpc_port,
                               prefer_grpc=True,
                               index=settings.qdrant_index,
                               embedding_dim=settings.qdrant_embedding_dim,
                               hnsw_config=hnsw_config,
                               return_embedding=True,
                               wait_result_from_api=True)


@st.cache_resource(show_spinner=False)
def get_rag_pipeline(_document_store: QdrantDocumentStore) -> Pipeline:
    # get settings
    settings = get_settings()

    # router
    router = ConditionalRouter(AGENT_ROUTER)
    router_prompt = PromptBuilder(PROMPT_FOR_ROUTER)
    router_llm = OpenAIGenerator(model=settings.openai_chat_model)

    # extraction of input features
    prediction_prompt = PromptBuilder(PROMPT_FOR_PREDICTION_EXTRACTION)
    prediction_llm = OpenAIGenerator(
        model=settings.openai_chat_model,
        generation_kwargs={"response_format": {
            "type": "json_object"
        }})
    prediction_component = PredictHousePrice()

    # prediction result
    prediction_result_prompt = PromptBuilder(PROMPT_FOR_PREDICTION_RESULT)
    prediction_result_llm = OpenAIGenerator(model=settings.openai_chat_model)

    # RAG
    rag_embedder = OpenAITextEmbedder(model=settings.openai_embedding_model)
    rag_retriever = QdrantEmbeddingRetriever(document_store=_document_store)
    rag_prompt = PromptBuilder(template=PROMPT_FOR_RAG)
    rag_llm = OpenAIGenerator(model=settings.openai_chat_model)
    rag_doc_returner = ReturnDocumentsFromRetriever()

    # create pipeline
    pipeline = Pipeline()

    # router phase
    pipeline.add_component("router_prompt", router_prompt)
    pipeline.add_component("router_llm", router_llm)
    pipeline.add_component("router", router)

    # if the route is PREDICTION
    pipeline.add_component("prediction_prompt", prediction_prompt)
    pipeline.add_component("prediction_llm", prediction_llm)
    pipeline.add_component("prediction_component", prediction_component)
    pipeline.add_component("prediction_prompt_for_result",
                           prediction_result_prompt)
    pipeline.add_component("prediction_result_llm", prediction_result_llm)

    # if the route is DATABASE_SEARCH
    pipeline.add_component("rag_embedder", rag_embedder)
    pipeline.add_component("rag_retriever", rag_retriever)
    pipeline.add_component("rag_prompt", rag_prompt)
    pipeline.add_component("rag_llm", rag_llm)
    pipeline.add_component("rag_doc_returner", rag_doc_returner)

    # connect the components
    pipeline.connect("router_prompt", "router_llm")
    pipeline.connect("router_llm.replies", "router.replies")

    pipeline.connect("router.features", "prediction_prompt")
    pipeline.connect("prediction_prompt", "prediction_llm")
    pipeline.connect("prediction_llm", "prediction_component")
    pipeline.connect("prediction_component.prediction",
                     "prediction_prompt_for_result.prediction")
    pipeline.connect("prediction_component.features",
                     "prediction_prompt_for_result.features")
    pipeline.connect("prediction_prompt_for_result", "prediction_result_llm")

    pipeline.connect("router.question", "rag_embedder.text")
    pipeline.connect("router.question", "rag_prompt.question")
    pipeline.connect("rag_embedder.embedding", "rag_retriever.query_embedding")
    pipeline.connect("rag_retriever", "rag_prompt.documents")
    pipeline.connect("rag_retriever", "rag_doc_returner")
    pipeline.connect("rag_prompt", "rag_llm")

    return pipeline


def query(_pipeline: Pipeline, question: str) -> QueryResult:
    # get settings
    settings = get_settings()

    # run the pipeline
    result = _pipeline.run(
        {
            "router_prompt": {
                "query": question
            },
            "router": {
                "query": question
            },
        },
        debug=settings.debug)

    # print if debug
    if settings.debug:
        print(result)

    # check if the result is RAG
    if "prediction_llm" in result:
        return QueryResult(
            success=True,
            content=result["prediction_result_llm"]["replies"][0],
            documents=[],
            raw=result)

    # create query documents
    inserted_ids = []
    documents = []

    for doc in result["rag_doc_returner"]["documents"]:
        # check if the document is already inserted
        if doc["id"] in inserted_ids:
            continue

        # insert the document
        inserted_ids.append(doc["id"])
        documents.append(
            QueryDocument(id=doc["id"],
                          city=doc["city"],
                          district=doc["district"],
                          price=doc["price"],
                          url=doc["url"],
                          main_image_url=doc["main_image_url"]))

    # parse the result
    response = QueryResult(success=True,
                           content=result["rag_llm"]["replies"][0],
                           documents=documents,
                           raw=result)

    return response
