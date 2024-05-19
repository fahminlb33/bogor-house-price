import os
import io
import uuid
import zipfile
import argparse

os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

import requests
import pandas as pd
import tensorflow as tf
import tensorflow_hub as hub

from tqdm import tqdm
from qdrant_client import QdrantClient
from qdrant_client.models import VectorParams, Distance, PointStruct

from utils.ml_base import TrainerMixin


class CreateEmbeddingsPipeline(TrainerMixin):

  def __init__(self,
               dataset_index: str,
               dataset_zip: str,
               model_path: str,
               qdrant_index_name: str,
               qdrant_host: str,
               qdrant_port: int,
               qdrant_port_grpc: int,
               batch_size: int = 24) -> None:
    super().__init__()

    # vars
    self.dataset_index = dataset_index
    self.dataset_zip = dataset_zip
    self.qdrant_index_name = qdrant_index_name
    self.batch_size = batch_size

    # qdrant
    self.client = QdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        grpc_port=qdrant_port_grpc,
        prefer_grpc=True,
    )

    # qdrant
    self.client = QdrantClient(
        host=qdrant_host,
        port=qdrant_port,
        grpc_port=qdrant_port_grpc,
        prefer_grpc=True,
    )

    # create collection
    if not self.client.collection_exists(qdrant_index_name):
      self.client.create_collection(
          collection_name=qdrant_index_name,
          vectors_config=VectorParams(size=1280, distance=Distance.COSINE),
      )

    # embedding model
    self.model = tf.keras.Sequential(
        [hub.KerasLayer(model_path, trainable=False)])

    self.model.build([None, 224, 224, 3])

  def load_data(self):
    # load data
    self.df = pd.read_csv(self.dataset_index)

    # load zip file
    self.zip = zipfile.ZipFile(self.dataset_zip)

  def train(self):
    # process in batches
    batches = []
    failed_files = []
    for _, row in tqdm(self.df.iterrows(), total=len(self.df)):
      # get file name
      file_name = os.path.basename(row.photo_url)

      try:
        # get content
        file_content = io.BytesIO(self.zip.read(file_name))

        # preprocess
        img = tf.keras.utils.load_img(file_content, target_size=(224, 224))
        x = tf.keras.utils.img_to_array(img) / 255.0
        # x = tf.expand_dims(x, axis=0)

        # append to batch
        batches.append((x, row.reference_id, row.photo_url))

        # if this batch has reached self.batch_size, process
        if len(batches) == self.batch_size:
          self.process_batch(batches)
          batches = []
      except Exception as e:
        failed_files.append(file_name)
        # self.logger.error(f"Error processing image: {file_name}. Error: {e}")

    # process the rest
    if len(batches) != 0:
      self.process_batch(batches)

    # close zip
    self.zip.close()

    # report
    self.logger.info(f"Failed files: {len(failed_files)} ({len(failed_files) / len(self.df) * 100:.2f}%)")
    with open("failed_files.txt", "w") as f:
      f.write("\n".join(failed_files))

  def process_batch(self, batch):
    # concat features
    images = tf.stack([img for img, _, _ in batch])

    # extract features
    features = self.model.predict(images, verbose=0)

    # create points
    points = [
            PointStruct(
                id=str(uuid.uuid4()),
                vector=img.ravel().tolist(),
                payload={
                    "document_id": meta[1],
                    "image_url": meta[2]
                }) for img, meta in zip(features, batch)
        ]

    # store
    self.client.upsert(
        collection_name=self.qdrant_index_name,
        points=points)


if __name__ == "__main__":
  # setup command-line arguments
  parser = argparse.ArgumentParser()

  parser.add_argument(
      "--dataset-index",
      help="Input dataset from L3",
      default="./dataset/curated/stg_rumah123_images.csv")
  parser.add_argument(
      "--dataset-images",
      help="Input dataset from L3",
      default="./dataset/images.zip")
  parser.add_argument(
      "--model-path",
      help="MobileNetV3 weights",
      default="./.docker/models/mobilenet_v3")
  parser.add_argument(
      "--index-name", help="Index name in Qdrant", default="rumah_bogor_images")
  parser.add_argument("--host", help="Qdrant host", default="localhost")
  parser.add_argument("--port", help="Qdrant port", default=6333)
  parser.add_argument("--port-grpc", help="Qdrant gRPC port", default=6334)

  args = parser.parse_args()

  # create embeddings
  embedder = CreateEmbeddingsPipeline(
      dataset_index=args.dataset_index,
      dataset_zip=args.dataset_images,
      model_path=args.model_path,
      qdrant_index_name=args.index_name,
      qdrant_host=args.host,
      qdrant_port=args.port,
      qdrant_port_grpc=args.port_grpc)

  embedder.run()
