import json
import datetime
import argparse

import pandas as pd

from haystack import Pipeline, Document
from haystack.components.preprocessors import DocumentSplitter
from haystack.components.embedders import OpenAIDocumentEmbedder
from haystack.components.writers import DocumentWriter
from haystack_integrations.document_stores.qdrant import QdrantDocumentStore

from utils.ml_base import TrainerMixin
from utils.ml_llm import EmbeddingDocumentTemplateEngine


class CreateEmbeddingsPipeline(TrainerMixin):

  def __init__(self, dataset: str, qdrant_index_name: str, qdrant_host: str,
               qdrant_port: int, qdrant_port_grpc: int,
               splitter_chunk_size: int, splitter_chunk_overlap: int,
               openai_model: str, template_name: str) -> None:
    super().__init__()

    # dataset
    self.dataset = dataset

    # qdrant
    self.qdrant_collection_name = qdrant_index_name
    self.qdrant_host = qdrant_host
    self.qdrant_port = qdrant_port
    self.qdrant_port_grpc = qdrant_port_grpc

    # document splitter
    self.splitter_chunk_size = splitter_chunk_size
    self.splitter_chunk_overlap = splitter_chunk_overlap

    # openai
    self.openai_model = openai_model

    # document template
    self.document_template = EmbeddingDocumentTemplateEngine(template_name)

  def load_data(self):
    # load data
    df = pd.read_parquet(self.dataset)

    # load all data
    documents = []
    for _, row in df.iterrows():
      try:
        # render document
        contents = self.document_template(row.to_dict())
        metadata = dict(
            id=row.id,
            price=row.price,
            district=row.district,
            city=row.city,
            url=row.url,
            main_image_url=row.main_image_url)

        documents.append(Document(id=row.id, content=contents, meta=metadata))
      except Exception as e:
        self.logger.error(f"Error rendering document {row.id}. Error: {e}")

    # set documents
    self.documents = documents

  def train(self):
    # create document store
    hnsw = dict(m=16, ef_construct=100)
    document_store = QdrantDocumentStore(
        host=self.qdrant_host,
        port=self.qdrant_port,
        grpc_port=self.qdrant_port_grpc,
        prefer_grpc=True,
        index=self.qdrant_collection_name,
        embedding_dim=1536,  # 1536 for text-embedding-3-small
        hnsw_config=hnsw,
        return_embedding=True,
        wait_result_from_api=True)

    # create embedding pipeline
    pipeline = Pipeline()

    # add components
    pipeline.add_component(
        "split",
        DocumentSplitter(
            split_by="word",
            split_length=self.splitter_chunk_size,
            split_overlap=self.splitter_chunk_overlap))
    pipeline.add_component("embedder",
                           OpenAIDocumentEmbedder(model=self.openai_model))
    pipeline.add_component("store",
                           DocumentWriter(document_store=document_store))

    # connect components
    pipeline.connect("split", "embedder")
    pipeline.connect("embedder", "store")

    # run pipeline
    stats = pipeline.run({"split": {"documents": self.documents}})

    # print statistics
    self.logger.info(f"Total documents: {stats['store']['documents_written']}")
    self.logger.info(f"Model: {stats['embedder']['meta']['model']}")
    self.logger.info(
        f"Total tokens: {stats['embedder']['meta']['usage']['total_tokens']}")
    self.logger.info(
        f"Prompt tokens: {stats['embedder']['meta']['usage']['prompt_tokens']}")

    # create filename
    filename = f"embeddings_{datetime.datetime.now().strftime('%Y%m%d%H%M%S')}.json"
    with open(filename, "w") as f:
      data = {
          "total_documents": stats['store']['documents_written'],
          "model": stats['embedder']["meta"]["model"],
          "total_tokens": stats['embedder']["meta"]["usage"]["total_tokens"],
          "prompt_tokens": stats['embedder']["meta"]["usage"]["prompt_tokens"],
      }

      json.dump(data, f)


if __name__ == "__main__":
  # setup command-line arguments
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--dataset",
      help="Input dataset from L3",
      default="./dataset/curated/marts_llm_houses.parquet")
  parser.add_argument(
      "--index-name", help="Index name in Qdrant", default="houses")
  parser.add_argument("--host", help="Qdrant host", default="localhost")
  parser.add_argument("--port", help="Qdrant port", default=6333)
  parser.add_argument("--port-grpc", help="Qdrant gRPC port", default=0)
  parser.add_argument(
      "--chunk-size", help="Text splitter chunk size", default=700)
  parser.add_argument(
      "--chunk-overlap", help="Text splitter chunk overlap", default=50)
  parser.add_argument(
      "--openai-model",
      help="OpenAI model",
      choices=[
          "text-embedding-3-small", "text-embedding-3-large",
          "text-embedding-ada-002"
      ],
      default="text-embedding-3-small")
  parser.add_argument(
      "--template-name",
      help="Document template name",
      default="document_v3.jinja2")

  args = parser.parse_args()

  # create embeddings
  embedder = CreateEmbeddingsPipeline(
      dataset=args.dataset,
      qdrant_index_name=args.index_name,
      qdrant_host=args.host,
      qdrant_port=args.port,
      qdrant_port_grpc=args.port_grpc,
      splitter_chunk_size=args.chunk_size,
      splitter_chunk_overlap=args.chunk_overlap,
      openai_model=args.openai_model,
      template_name=args.template_name)

  embedder.run()
