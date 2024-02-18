import numpy as np
import pandas as pd
import geopandas as gpd

SPATIAL_OVERPASS_CRS = "EPSG:4326"
SPATIAL_GEODETIC_CRS = "EPSG:23839"
SPATIAL_SHP_DROP_COLUMNS = [
    "KDPPUM", "REMARK", "KDPBPS", "FCODE", "LUASWH", "UUPP", "SRS_ID",
    "METADATA", "KDEBPS", "KDEPUM", "KDCBPS", "KDCPUM", "KDBBPS", "KDBPUM",
    "WADMKD", "WIADKD", "WIADKC", "WIADKK", "WIADPR", "TIPADM"
]
SPATIAL_PLACE_NORM_RULES = {
    # from houses to SHP
    # "Pajajaran": "Babakan",
    # "Taman Kencana": "Babakan",
    "Babakan Madang": "Babakanmadang",
    # "Bukit Sentul": "Babakanmadang",
    "Babakan Pasar": "Babakanpasar",
    "Balumbang Jaya": "Balumbangjaya",
    "Bantar Jati": "Bantarjati",
    # "Indraprasta": "Bantarjati",
    # "Ardio": "Bogor Tengah",
    "Bojong Gede": "Bojonggede",
    "Bojong Kulur": "Bojongkulur",
    # "Cilendek": "Cilendek Barat",
    "Curug Mekar": "Curugmekar",
    "Gunung Batu": "Gunungbatu",
    "Gunung Putri": "Gunungputri",
    # "Kota Wisata": "Gunungputri",
    # "Legenda Wisata": "Gunungputri",
    # "Kranggan": "Gunungputri",
    "Gunung Sindur": "Gunungsindur",
    # "Harjamukti": "INI GA ADA DI SHP",
    "Karang Tengah": "Karangtengah",
    "Kebon Kelapa": "Kebonkalapa",
    "Kedungbadak": "Kedungbadak",
    "Kedung Halang": "Kedunghalang",
    "Leuwinanggung": "Lewinanggung",
    # "Jl Dr Semeru": "Menteng",
    "Muara Sari": "Muarasari",
    # "Bogor Nirwana Residence": "Mulyaharja",
    "Parung Panjang": "Parungpanjang",
    "Pasir Jaya": "Pasirjaya",
    "Pasir Kuda": "Pasirkuda",
    "Pasir Muncang": "Pasirmuncang",
    "Ranca Bungur": "Rancabungur",
    "Rangga Mekar": "Ranggamekar",
    "Sentul City": "Sentul",
    "Sindang Barang": "Sindangbarang",
    "Sindang Sari": "Sindangsari",
    "Situ Gede": "Situgede",
    "Tajur Halang": "Tajurhalang",
    # "Ahmadyani": "Tanahsareal",
    # "Jl A Yani": "Tanahsareal",
    "Tanah Sareal": "Tanahsareal",
    "Tegal Gundi": "Tegalgundil",
    "Tegal Gundil": "Tegalgundil",
    "Duta Pakuan": "Tegallega",

    # dedupe in SHP
    "Bantar Gebang": "Bantargebang",
}


def model(dbt, session):
  	# load shapefile dataset
	shapefiles = [
		gpd.read_file(path) for path in dbt.config.get("shapefile_paths")
	]

	df_bogor = pd.concat(shapefiles, ignore_index=True)

	# load overpass dataset
	df_osm = dbt.ref("stg_osm_amenities").df().drop(columns=["nodes", "tags"])
	df_overpass = gpd.GeoDataFrame(df_osm, geometry=gpd.points_from_xy(df_osm.lon, df_osm.lat), crs=SPATIAL_OVERPASS_CRS)
	df_overpass = df_overpass.drop(columns=["lat", "lon"])

	# load houses dataset
	df_houses = dbt.ref("stg_rumah123_houses").select("district", "price").df()
	df_houses["place"] = df_houses["district"].replace(SPATIAL_PLACE_NORM_RULES)
	df_houses["price"] = np.log(df_houses["price"])

	avg_house_prices = df_houses.groupby("place")["price"].mean()

	# join spatial datasets (points from overpass and polygons from shapefile)
	df_places = gpd.sjoin_nearest(df_bogor.to_crs(SPATIAL_GEODETIC_CRS), df_overpass.to_crs(SPATIAL_GEODETIC_CRS), distance_col="distance") \
		.drop(columns=["index_right"])

	# convert to pandas DataFrame
	df_places = pd.DataFrame(df_places.drop(columns=["geometry", "SHAPE_Leng", "SHAPE_Area"]))

	# calculate number of amenities per place
	df_amenities = df_places\
			.pivot_table(index="NAMOBJ", columns="category", values="id", aggfunc="count", fill_value=0) \
			.reset_index() \
			.rename(columns={"NAMOBJ": "place"})

	# normalize place names
	df_amenities["place"] = df_amenities["place"].replace(SPATIAL_PLACE_NORM_RULES)

	# join with average house prices
	df_spatial = df_amenities \
		.join(avg_house_prices, on="place") \
		.reset_index(drop=True) \
		.dropna()

	return df_spatial
