import streamlit as st

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image

torch.classes.__path__ = []


# --------------------- CACHED RESOURCES ---------------------


@st.cache_resource()
def get_text_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    text_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        safe_serialization=True,
    ).to(device)

    text_model.eval()

    return device, tokenizer, text_model


@st.cache_resource()
def get_image_model():
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
    vision_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
    ).to(device)

    return device, processor, vision_model


# --------------------- HELPERS ---------------------


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )

    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


# --------------------- EMBEDDINGS ---------------------


def embed_text(text: str):
    device, tokenizer, text_model = get_text_model()

    encoded_input = tokenizer(
        [text], padding=True, truncation=True, return_tensors="pt"
    ).to(device)
    with torch.no_grad():
        model_output = text_model(**encoded_input)

    embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
    embeddings = F.normalize(embeddings, p=2, dim=1)

    return embeddings.cpu().numpy()[0].tolist()


def embed_image(file_path: str):
    device, processor, vision_model = get_text_model()

    image = Image.open(file_path)
    inputs = processor(image, return_tensors="pt").to(device)

    img_emb = vision_model(**inputs).last_hidden_state
    img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)

    return img_embeddings.detach().cpu().numpy()[0].tolist()
