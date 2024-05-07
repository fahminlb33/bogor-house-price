import tensorflow as tf
import tensorflow_hub as hub

from haystack import Document
from qdrant_client import QdrantClient
from qdrant_client.models import Filter, FieldCondition, MatchValue, MatchAny

from appv2.utils.shared import get_settings, get_logger


class ImageSearcher:

  def __init__(self):
    # get config
    self.settings = get_settings()
    self.logger = get_logger("ImageSearcher")

    # create qdrant client
    self.client = QdrantClient(
        host=self.settings.QDRANT_HOST,
        port=self.settings.QDRANT_PORT,
        grpc_port=self.settings.QDRANT_GRPC_PORT,
        prefer_grpc=True,
    )

    # load mobilenet_v3 embedding model
    self.logger.info("Loading model...")
    self.model = tf.keras.Sequential([
        hub.KerasLayer(
            self.settings.TENSORFLOW_MOBILENET_MODEL, trainable=False)
    ])

    # build model
    self.logger.info("Building model...")
    self.model.build([None, 224, 224, 3])

  def search(self, image_path: str) -> list[Document]:
    self.logger.debug("Loading image...")

    # load the image and preprocess
    img = tf.keras.utils.load_img(image_path, target_size=(224, 224))
    x = tf.keras.utils.img_to_array(img) / 255.0
    x = tf.expand_dims(x, axis=0)

    # run inference
    self.logger.debug("Extracting image features...")
    features = self.model.predict(x).ravel()

    # search
    self.logger.debug("Searching for similar images...")
    results = self.client.search(
        collection_name=self.settings.QDRANT_IMAGE_INDEX,
        query_vector=features,
        limit=self.settings.QDRANT_TOP_K,
        with_payload=["document_id"])

    # retrieve documents
    self.logger.debug("Retrieving documents...")
    doc_ids = [item.payload["document_id"] for item in results]
    documents, _ = self.client.scroll(
        collection_name=self.settings.QDRANT_DOCUMENT_INDEX,
        scroll_filter=Filter(
            must=[FieldCondition(key="meta.id", match=MatchAny(any=doc_ids))]),
        with_payload=True)

    # return results
    return [
        Document(doc.id, doc.payload["content"], meta=doc.payload["meta"])
        for doc in documents
    ]


searcher = ImageSearcher()
