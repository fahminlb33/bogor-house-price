import streamlit as st

from haystack import Pipeline, Document, component
from haystack.components.embedders import OpenAITextEmbedder
from haystack.components.generators import OpenAIGenerator
from haystack.components.builders.prompt_builder import PromptBuilder
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore
from haystack_integrations.components.retrievers.qdrant import QdrantEmbeddingRetriever

from utils.config import get_settings

RAG_PROMPT_TEMPLATE = (
    "You are an assistant for house recommendation/suggestion tasks. "
    "You will be given a few documents about property listing along with it's price, address, and specifications. "
    "Give a summary about the house specs and address if you have a match. "
    "Do not return the result as lists, but as a paragraph. "
    "You can suggest more than one house based on the context. "
    "If you don't know the answer or there is no relevant answer, just say NO_RESULTS exactly. "
    "Answer in Indonesian language regardless of the prompt language."
    "Use five sentences maximum and keep the answer concise.\n\n"
    "Context:\n"
    "###\n"
    "{% for doc in documents %}"
    "{{ doc.content }}"
    "{% endfor %}"
    "###\n\n"
    "Question: {{question}}\n"
    "Answer:")


@component
class ReturnDocumentsFromRetriever:

    @component.output_types(documents=list[dict])
    def run(self, docs: list[Document]):
        return {"documents": [{"id": doc.id, **doc.meta} for doc in docs]}


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
    # create pipeline
    rag_pipeline = Pipeline()

    # add components
    rag_pipeline.add_component("embedder", OpenAITextEmbedder())
    rag_pipeline.add_component(
        "retriever", QdrantEmbeddingRetriever(document_store=_document_store))
    rag_pipeline.add_component("rag_prompt",
                               PromptBuilder(template=RAG_PROMPT_TEMPLATE))
    rag_pipeline.add_component("llm", OpenAIGenerator(model="gpt-3.5-turbo"))
    rag_pipeline.add_component("return_docs", ReturnDocumentsFromRetriever())

    # connect components
    rag_pipeline.connect("embedder.embedding", "retriever.query_embedding")
    rag_pipeline.connect("retriever", "rag_prompt.documents")
    rag_pipeline.connect("retriever", "return_docs")
    rag_pipeline.connect("rag_prompt", "llm")

    return rag_pipeline


@st.cache_data(show_spinner=False)
def query(_pipeline: Pipeline, question: str) -> dict:
    return _pipeline.run({
        "embedder": {
            "text": question
        },
        "rag_prompt": {
            "question": question
        },
    })
