from pathlib import Path
from textwrap import dedent
from dataclasses import asdict

from utils.embeddings import embed_image, embed_text
from utils.price_predictor import get_subdistricts, predict_price
from utils.db import (
    query_locations,
    query_hybrid,
    query_image,
    query_houses,
    query_house_images,
)


SYSTEM_INSTRUCTION = dedent(
    """
    Kamu adalah sales rumah yang baik.
    Selalu gunakan Bahasa Indonesia dalam percakapan.
    """
)


def top_listing_by_location() -> list[dict[str, str | int | float]]:
    """List locations with the most available house sales listing along with its average price.

    Returns:
        A list of dictionary containing the area subdistrict, district, city and province, along with the number of house for sale and its average price
    """

    locations = query_locations()
    return [asdict(x) for x in locations]


def search_by_keyword(query: str, image_mandatory: bool) -> list[dict[str, str]]:
    """Search house sale listing using a search query.

    Args:
        query: Search query describing the house information including price, location, number of bedrooms, etc.
        image_mandatory: Whether to return properties with an image. Defaults to False.

    Returns:
        A list of dictionary containing a unique house ID and detailed house description.
    """

    text_embedding = embed_text(query)
    retrieved_documents = query_hybrid(query, text_embedding, image_mandatory)
    documents = query_houses(
        [x.id for x in retrieved_documents]
        + [x.parent_id for x in retrieved_documents if x.parent_id is not None]
    )

    return [asdict(x) for x in documents]


def search_by_image_id(image_id: str) -> list[dict[str, str]]:
    """Search house sale listing using an image.

    Args:
        image_id: Unique ID supplied by the user after it is uploaded to the system.

    Returns:
        A list of dictionary containing a unique house ID and detailed house description.
    """

    image_path = Path("../data/rumah123/images") / image_id
    if not image_path.exists():
        raise ValueError("Image not found")

    with open(image_path, "rb") as f:
        image_embedding = embed_image(f.read())

    retrieved_documents = query_image(image_embedding)
    documents = query_houses([x.parent_id for x in retrieved_documents])

    return [asdict(x) for x in documents]


def get_house_images(house_id: str) -> list[str]:
    """Gets the image paths associated with the specified house ID.

    Args:
        house_id: Unique house ID that must starts with "hos".

    Returns:
        A list of file paths to the images.
    """

    if not house_id.startswith("hos"):
        raise ValueError("House ID must start with hos")

    norm_id = house_id.replace("-desc", "")
    retrieved_documents = query_house_images(norm_id)

    return retrieved_documents


def get_available_subdistricts() -> list[str]:
    """Lists the available subdistrict locations of the house listings. Used for searching for sale properties and predicting property prices.

    Returns:
        A list of subdistrict names.
    """

    return get_subdistricts()


def predict_house_price(
    subdistrict: str,
    land_area: float,
    building_area: float,
    num_bedrooms: int,
    num_bathrooms: int,
    num_floors: int,
    year_built: int,
    electricity_rate: float,
) -> float:
    """Predicts a house price based on its features.

    Args:
        subdistrict: Subdistrict name of the property. This is a required field.
        land_area: Estimated land area of the property in meters squared. This is a required field.
        building_area: Estimated building area on top of the land in meters squared. This is a required field.
        num_bedrooms: Number of bedrooms. The default value is 1.
        num_bathrooms: Number of bathrooms. The default value is 1.
        num_floors: Number of floors. The default value is 1.
        year_built: What year the property is built. The default value is 0.
        electricity_rate: The electrical wattage subscription from electricity provider. The default value is 1300.

    Returns:
        The predicted property price in IDR.
    """

    return predict_price(
        {
            "subdistrict": [subdistrict],
            "luas_tanah": [land_area],
            "luas_bangunan": [building_area],
            "jumlah_lantai": [num_floors],
            "tahun_dibangun": [year_built],
            "daya_listrik": [electricity_rate],
            "land_building_ratio": [land_area / building_area],
            "total_bedrooms": [num_bedrooms],
            "total_bathrooms": [num_bathrooms],
            "building_area_floor_ratio": [building_area / num_floors],
        }
    )
