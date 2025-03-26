import re
import json
import pathlib
import hashlib
import argparse

import torch
import torch.nn.functional as F
from transformers import AutoTokenizer, AutoModel, AutoImageProcessor
from PIL import Image, UnidentifiedImageError

import tqdm
import psycopg
from pgvector.psycopg import register_vector
from jinja2 import Environment, FileSystemLoader, TemplateError, select_autoescape


class DocumentPromptBuilder(object):
    def __init__(self, template_path: str, template_name: str) -> None:
        # create Jinja environment
        fs_loader = FileSystemLoader(template_path)
        self.env = Environment(loader=fs_loader, autoescape=select_autoescape())

        # add custom filters
        self.env.filters["norm_description"] = DocumentPromptBuilder.norm_description
        self.env.filters["norm_scalar"] = DocumentPromptBuilder.norm_scalar
        self.env.filters["num_max"] = DocumentPromptBuilder.num_max
        self.env.filters["translate_compass"] = DocumentPromptBuilder.translate_compass
        self.env.filters["translate_gaya"] = DocumentPromptBuilder.translate_gaya
        self.env.filters["translate_pemandangan"] = (
            DocumentPromptBuilder.translate_pemandangan
        )
        self.env.filters["translate_water_source"] = (
            DocumentPromptBuilder.translate_water_source
        )

        # load template
        self.template = self.env.get_template(template_name)

    def __call__(self, tp: dict) -> str:
        return self.template.render(**tp)

    @staticmethod
    def norm_description(s: str) -> str:
        # remove emojis
        s = s.encode("ascii", "ignore").decode("ascii")

        # remove non-ascii characters
        s = re.sub(r"[^\x00-\x7F]+", "", s)

        # convert newlines to full stops
        s = s.replace("\n", ". ")

        # remove multiple spaces
        s = re.sub(r"\s+", " ", s)

        # remove space before punctuation
        s = re.sub(r'\s([?.!:"](?:\s|$))', r"\1", s)

        # remove double punctuation
        s = re.sub(r'([?.!"])([?.!"])+', r"\1", s)

        return s

    @staticmethod
    def norm_scalar(
        s: float | int, suffix: str = "", default_value: str = "tidak disebutkan"
    ) -> str:
        if s == 0:
            return default_value

        return f"{s}{suffix}"

    @staticmethod
    def num_max(x, y):
        if x > y:
            return x
        return y

    @staticmethod
    def translate_compass(direction: str):
        if direction == "Utara":
            return "North"
        elif direction == "Selatan":
            return "South"
        elif direction == "Barat":
            return "West"
        elif direction == "Timur":
            return "East"
        elif direction == "Barat Daya":
            return "Northeast"
        elif direction == "Barat Laut":
            return "Northeest"
        elif direction == "Timur Laut":
            return "Southeest"
        elif direction == "Tenggara":
            return "Southeast"
        else:
            return direction

    @staticmethod
    def translate_gaya(s: str):
        return s.replace("Minimalis", "Minimalist")

    @staticmethod
    def translate_pemandangan(s: str):
        if s == "Pemukiman Warga":
            return "neighborhood"
        elif s == "Pedesaan":
            return "village"
        elif s == "Perkotaan":
            return "cities"
        elif s == "Pegunungan":
            return "mountains"
        elif s == "Taman Kota":
            return "park"
        else:
            return s

    @staticmethod
    def translate_water_source(s: str):
        if s == "Sumur Bor":
            return "wells"
        elif s == "PAM atau PDAM":
            return "municipal water"
        elif s == "Sumur Resapan":
            return "wells"
        elif s == "Sumur Pompa":
            return "wells"
        else:
            return s


def mean_pooling(model_output, attention_mask):
    token_embeddings = model_output[0]
    input_mask_expanded = (
        attention_mask.unsqueeze(-1).expand(token_embeddings.size()).float()
    )
    return torch.sum(token_embeddings * input_mask_expanded, 1) / torch.clamp(
        input_mask_expanded.sum(1), min=1e-9
    )


def create_tables(conn: psycopg.Connection):
    with conn.cursor() as cur:
        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS houses (
                id              varchar(255)    not null,
                parent_id       varchar(255)    null,
                content         text            not null,
                embedding       vector(768)     not null,
                primary key (id),
                constraint fk_house_linked foreign key (parent_id) references houses (id)
            );
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS ON public.houses USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
            """
        )

        cur.execute(
            """
            CREATE TABLE IF NOT EXISTS house_images (
                id              varchar(255)    not null,
                parent_id       varchar(255)    null,
                file_path       text            not null,
                embedding       vector(768)     not null,
                primary key (id),
                constraint fk_house foreign key (parent_id) references houses (id)
            );
            """
        )

        cur.execute(
            """
            CREATE INDEX IF NOT EXISTS ON public.house_images USING hnsw (embedding vector_cosine_ops) WITH (m = 16, ef_construction = 64);
            """
        )


def index_text(conn: psycopg.Connection, args):
    # create prompt builder
    prompt_builder = DocumentPromptBuilder(args.template_path, args.template_name)

    # https://huggingface.co/nomic-ai/nomic-embed-text-v1.5
    # load model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
    model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-text-v1.5",
        trust_remote_code=True,
        safe_serialization=True,
    ).to(device)

    model.eval()

    # open db cursor and dataset
    with conn.cursor() as cur, open(args.dataset_path) as dataset_file:
        for line in (pbar := tqdm.tqdm(dataset_file)):
            parsed = json.loads(line)

            try:
                # generate text
                house_id = parsed["id"]
                content = prompt_builder(parsed)

                pbar.set_description(house_id)

                # build sentences to embed
                sentences = [f"search_document: {content}"]
                if (
                    parsed["description"] is not None
                    and len(parsed["description"]) > 10
                ):
                    sentences += [f"search_document: {parsed['description']}"]

                # tokenize sentences
                encoded_input = tokenizer(
                    sentences, padding=True, truncation=True, return_tensors="pt"
                ).to(device)

                # encode text
                with torch.no_grad():
                    model_output = model(**encoded_input)

                embeddings = mean_pooling(model_output, encoded_input["attention_mask"])
                embeddings = F.normalize(embeddings, p=2, dim=1)
                embeddings_np = embeddings.cpu().numpy()

                # save to DB
                cur.execute(
                    "INSERT INTO houses (id, content, embedding) VALUES (%s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (house_id, content, embeddings_np[0, :].tolist()),
                )

                if embeddings_np.shape[0] > 1:
                    cur.execute(
                        "INSERT INTO houses (id, parent_id, content, embedding) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                        (
                            f"{house_id}-desc",
                            house_id,
                            parsed["description"],
                            embeddings_np[1, :].tolist(),
                        ),
                    )
            except TemplateError:
                print(line)

    # release memory
    del tokenizer
    del model


def index_image(conn: psycopg.Connection, args):
    # input image root dir
    source_path = pathlib.Path(args.images_path)

    # https://huggingface.co/nomic-ai/nomic-embed-vision-v1.5
    # create model
    device = "cuda:0" if torch.cuda.is_available() else "cpu"
    processor = AutoImageProcessor.from_pretrained("nomic-ai/nomic-embed-vision-v1.5")
    vision_model = AutoModel.from_pretrained(
        "nomic-ai/nomic-embed-vision-v1.5", trust_remote_code=True
    ).to(device)

    # open db cursor
    with conn.cursor() as cur:
        for file_path in (pbar := tqdm.tqdm(source_path.glob("**/*.jpg"))):
            # get meta
            parent_id = file_path.parent.name
            relative_path = file_path.parent.name + "/" + file_path.name
            image_id = hashlib.sha256(relative_path.encode("utf-8")).hexdigest()

            pbar.set_description(parent_id)

            try:
                # load image and tokenize
                image = Image.open(file_path)
                inputs = processor(image, return_tensors="pt").to(device)

                # encode image
                img_emb = vision_model(**inputs).last_hidden_state
                img_embeddings = F.normalize(img_emb[:, 0], p=2, dim=1)
                embeddings_list = img_embeddings.detach().cpu().numpy()[0].tolist()

                # save to DB
                cur.execute(
                    "INSERT INTO house_images (id, parent_id, file_path, embedding) VALUES (%s, %s, %s, %s) ON CONFLICT (id) DO NOTHING",
                    (image_id, parent_id, relative_path, embeddings_list),
                    prepare=True,
                )
            except UnidentifiedImageError:
                print(f"ERROR: {relative_path}")

    # release memory
    del processor
    del vision_model


def main(args):
    with psycopg.connect(args.database_uri, autocommit=True) as conn:
        register_vector(conn)

        print("Create tables...")
        create_tables(conn)

        print("Indexing text data...")
        index_text(conn, args)

        print("Indexing image data...")
        index_image(conn, args)


if __name__ == "__main__":
    # setup command-line arguments
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--dataset-path",
        type=str,
        help="Input dataset from L3",
        default="../data/marts_houses_downstream.parquet",
    )
    parser.add_argument(
        "--images-path",
        type=str,
        help="Input dataset from L3",
        default="../data/rumah123/images",
    )
    parser.add_argument(
        "--database-uri",
        type=str,
        help="Postgres DB URI",
        default="dbname=rumah-db user=rumah-user password=rumah-password host=localhost port=5432",
    )
    parser.add_argument(
        "--template-path",
        type=str,
        help="Template path root",
        default="./templates",
    )
    parser.add_argument(
        "--template-name",
        type=str,
        help="Template name",
        default="document_v3.jinja2",
    )

    args = parser.parse_args()
    main(args)
