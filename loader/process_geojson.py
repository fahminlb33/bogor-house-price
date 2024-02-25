import argparse
import logging

import pandas as pd
import geopandas as gpd

SPATIAL_SHP_DROP_COLUMNS = [
    "KDPPUM", "REMARK", "KDPBPS", "FCODE", "LUASWH", "UUPP", "SRS_ID",
    "METADATA", "KDEBPS", "KDEPUM", "KDCBPS", "KDCPUM", "KDBBPS", "KDBPUM",
    "WADMKD", "WIADKD", "WIADKC", "WIADKK", "WIADPR", "TIPADM"
]

SHAPEFILE_PATHS = [
    "./dataset/shp/kota/ADMINISTRASIDESA_AR_25K.shp",
    "./dataset/shp/kab/ADMINISTRASIDESA_AR_25K.shp"
]

def MultipleArgsParser(s):
    return [str(item) for item in s.split(',')]

class ShapefileToGeojsonPipeline:
    def __init__(self, shapefile_paths: list[str], output_file: str) -> None:
        super().__init__()

        self.shapefile_paths = shapefile_paths
        self.output_file = output_file

    def run(self):
        # load shapefiles
        shapefiles = [gpd.read_file(path) for path in self.shapefile_paths]

        # merge shapefiles
        df_bogor = pd.concat(shapefiles, ignore_index=True)
        df_bogor = gpd.GeoDataFrame(df_bogor.drop(columns=SPATIAL_SHP_DROP_COLUMNS))

        # save to file
        df_bogor.to_file(self.output_file, driver='GeoJSON')


if __name__ == '__main__':
    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--shapefiles",
                        nargs=MultipleArgsParser,
                        help="Shapefiles to convert to GeoJSON",
                        default=SHAPEFILE_PATHS)
    parser.add_argument("--output-file",
                        help="Output file for GeoJSON",
                        default="./dataset/bogor.json")

    args = parser.parse_args()

    # run pipeline
    pipeline = ShapefileToGeojsonPipeline(args.shapefiles, args.output_file)
    pipeline.run()

