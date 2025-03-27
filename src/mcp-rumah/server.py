import io
import base64
import random
from pathlib import Path
from dataclasses import asdict
from urllib.parse import unquote

from mcp.server.fastmcp import FastMCP
from starlette.applications import Starlette
from starlette.routing import Mount

from config import config
from embeddings import embed_image, embed_text
from price_predictor import get_subdistricts, predict_price
from db import (
    query_locations,
    query_hybrid,
    query_image,
    query_houses,
    query_house_images,
)

mcp = FastMCP("Project Rumah Bogor")

# --------------------- RESOURCES ---------------------

@mcp.resource("locations://statistics", mime_type="application/json")
def top_listing_by_location() -> list[dict[str, str | int | float]]:
    """List locations with the most available house sales listing along with its average price.

    Returns:
        A list of dictionary containing the area subdistrict, district, city and province, along with the number of house for sale and its average price
    """

    locations = query_locations()
    return [asdict(x) for x in locations]

@mcp.resource("locations://subdistricts", mime_type="application/json")
def get_available_subdistricts() -> list[str]:
    """Lists the available subdistrict locations of the house listings. Used for searching for sale properties and predicting property prices.

    Returns:
        A list of subdistrict names.
    """

    return get_subdistricts()

@mcp.resource("search://text/{keyword}/{with_image}", mime_type="application/json")
def search_by_keyword(keyword: str, with_image: bool) -> list[dict[str, str]]:
    """Search house sale listing using a search query.

    Args:
        keyword: Search query describing the house information including price, location, number of bedrooms, etc.
        image_mandatory: Whether to return properties with an image. Defaults to False.

    Returns:
        A list of dictionary containing a unique house ID and detailed house description.
    """

    text_embedding = embed_text(keyword)
    retrieved_documents = query_hybrid(keyword, text_embedding, with_image)
    documents = query_houses(
        [x.id for x in retrieved_documents]
        + [x.parent_id for x in retrieved_documents if x.parent_id is not None]
    )

    return [asdict(x) for x in documents]

@mcp.resource("search://image/{image_base64}", mime_type="application/json")
def search_by_image(image_base64: str) -> list[dict[str, str]]:
    """Search house sale listing using an image.

    Args:
        image_base64: Base64 encoded image to search.

    Returns:
        A list of dictionary containing a unique house ID and detailed house description.
    """

    image_data = unquote(image_base64)
    if image_data.find(",") != -1:
        image_data = image_data[image_data.index(",")+1:]

    image_data = base64.b64decode(image_data)
    image_data = io.BytesIO(image_data)
    
    image_embedding = embed_image(image_data)
    retrieved_documents = query_image(image_embedding)
    documents = query_houses([x.parent_id for x in retrieved_documents])

    return [asdict(x) for x in documents]

@mcp.resource("image://{house_id}", mime_type="image/jpg")
def get_house_image(house_id: str) -> str:
    """Gets the base64 encoded image associated with the specified house ID.

    Args:
        house_id: Unique house ID that must starts with "hos".

    Returns:
        A base64 encoded data of the image.
    """

    if not house_id.startswith("hos"):
        raise ValueError("House ID must start with hos")

    norm_id = house_id.replace("-desc", "")
    retrieved_documents = query_house_images(norm_id)

    if len(retrieved_documents) == 0:
        raise ValueError("No image associated with the specified house ID")

    image_path = Path(config("IMAGE_ROOT")) / random.choice(retrieved_documents)
    with open(image_path, "rb") as f:
        encoded_string = base64.b64encode(f.read())
        return encoded_string

# --------------------- TOOLS ---------------------

@mcp.tool()
def predict_house_price(
    subdistrict: str,
    land_area: float,
    building_area: float,
    num_bedrooms: int=1,
    num_bathrooms: int=1,
    num_floors: int=1,
    year_built: int=0,
    electricity_rate: float=0,
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
            "land_building_ratio": [land_area / max(building_area, 1)],
            "total_bedrooms": [num_bedrooms],
            "total_bathrooms": [num_bathrooms],
            "building_area_floor_ratio": [building_area / max(num_floors, 1)],
        }
    )


# --------------------- SSE SERVER ---------------------

app = Starlette(
    routes=[
        Mount('/', app=mcp.sse_app()),
    ]
)
