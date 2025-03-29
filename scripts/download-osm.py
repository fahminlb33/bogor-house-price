import argparse

import osmnx as ox
import pandas as pd
import geopandas as gpd


def main(args):
    print("Geocoding relations...")
    tags = {"amenity": True, "landuse": ["retail", "commercial"]}
    place = ox.geocoder.geocode_to_gdf(args.rels.split(","), by_osmid=True)

    print("Downloading relations...")
    gdf_osm = gpd.GeoDataFrame(
        pd.concat(
            [
                ox.features.features_from_polygon(
                    place["geometry"][0], tags
                ).reset_index(),  # Kota
                ox.features.features_from_polygon(
                    place["geometry"][1], tags
                ).reset_index(),  # Kabupaten
            ],
            ignore_index=True,
        )
    )

    print("Cleaning data...")
    gdf_osm_clean = gdf_osm[
        ["element", "id", "amenity", "landuse", "name", "name:en", "geometry"]
    ].copy()

    gdf_osm_clean["name"] = gdf_osm_clean["name"].fillna(gdf_osm_clean["name:en"])
    gdf_osm_clean["amenity"] = gdf_osm_clean["amenity"].fillna(gdf_osm_clean["landuse"])

    gdf_osm_clean = gdf_osm_clean.drop(columns=["landuse", "name:en"])
    print(gdf_osm_clean.head())

    print("Saving data...")
    gdf_osm_clean.to_file(args.output_file, driver="GeoJSON")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--rels", 
        type=str, 
        help="Nominatim relation IDs", 
        default="R14745927,R14762112"
    )
    parser.add_argument(
        "--output-file",
        type=str,
        help="Output directory",
        default="../data/osm/amenities.json",
    )

    args = parser.parse_args()
    main(args)
