import logging
import argparse

import duckdb

from etl_base import ETLMixin
from etl_normalizer import mask_name, mask_phone, clean_facility, clean_agency_company


class HouseL1Pipeline(ETLMixin):

    def __init__(self, input_dir: str, output_dir: str):
        super().__init__()

        self.input_dir = input_dir
        self.output_dir = output_dir

        # connect to database
        logging.info("Connecting to database")
        self.conn = duckdb.connect()

        self.conn.create_function("MASK_NAME", mask_name)
        self.conn.create_function("MASK_PHONE", mask_phone)
        self.conn.create_function("CLEAN_FACILITY", clean_facility)
        self.conn.create_function("CLEAN_AGENCY_COMPANY", clean_agency_company)

    def transform_spec_base(self):
        # create new table
        self.conn.sql(
            "CREATE OR REPLACE TABLE house_specs_raw AS SELECT specs.*, id AS reference_id FROM raw"
        )

        # get all columns
        cols = self.conn.sql("DESCRIBE house_specs_raw").df()
        cols = cols["column_name"].values

        # lowercase and replace spaces with underscores
        cols = [
            f"\"{col}\" AS {col.lower().replace(' ', '_')}" for col in cols
            if col != "reference_id"
        ] + ["reference_id"]

        # create house_specs table
        self.conn.sql(
            f"CREATE OR REPLACE TABLE house_specs AS SELECT {', '.join(cols)} FROM house_specs_raw"
        )

    def transform_spec_numeric(self):
        # create new table
        self.conn.sql(
            "CREATE OR REPLACE TABLE house_specs AS SELECT *, "
            "TRY_CAST(rtrim(luas_tanah, ' m²') AS FLOAT) AS luas_tanah_num, "
            "TRY_CAST(rtrim(luas_bangunan, ' m²') AS FLOAT) AS luas_bangunan_num, "
            "TRY_CAST(replace(rtrim(lower(daya_listrik), 'watt'), 'lainnya', '0') AS FLOAT) AS daya_listrik_num, "
            "TRY_CAST(rtrim(lebar_jalan, ' Mobil') AS FLOAT) AS lebar_jalan_num "
            "FROM house_specs")

        # drop original columns
        drop_cols = [
            "luas_tanah", "luas_bangunan", "daya_listrik", "lebar_jalan"
        ]

        for col in drop_cols:
            self.conn.sql(f"ALTER TABLE house_specs DROP COLUMN {col}")

        # rename columns
        rename_cols = {
            "luas_tanah_num": "luas_tanah",
            "luas_bangunan_num": "luas_bangunan",
            "daya_listrik_num": "daya_listrik",
            "lebar_jalan_num": "lebar_jalan"
        }

        for col_old, col_new in rename_cols.items():
            self.conn.sql(
                f"ALTER TABLE house_specs RENAME {col_old} TO {col_new}")

    def transform_spec_property_state(self):
        # create new table
        self.conn.sql("CREATE OR REPLACE TABLE house_specs AS SELECT *, "
                      "lower(kondisi_properti) AS kondisi_properti_norm, "
                      "lower(kondisi_perabotan) AS kondisi_perabotan_norm "
                      "FROM house_specs")

        for col in ["kondisi_properti", "kondisi_perabotan"]:
            # normalize values
            self.conn.sql(
                f"UPDATE house_specs SET {col}_norm = CASE {col}_norm "
                "WHEN 'bagus sekali' THEN 'furnished' "
                "WHEN 'sudah renovasi' THEN 'furnished' "
                "WHEN 'butuh renovasi' THEN 'unfurnished' "
                "WHEN 'bagus' THEN 'furnished' "
                f"WHEN 'baru' THEN 'furnished' ELSE {col}_norm END")

            # drop original column
            self.conn.sql(f"ALTER TABLE house_specs DROP COLUMN {col}")

        # rename columns
        rename_cols = {
            "kondisi_properti_norm": "kondisi_properti",
            "kondisi_perabotan_norm": "kondisi_perabotan",
        }

        for col_old, col_new in rename_cols.items():
            self.conn.sql(
                f"ALTER TABLE house_specs RENAME {col_old} TO {col_new}")

    def transform_spec_boolean(self):
        # create new table
        self.conn.sql(
            "CREATE OR REPLACE TABLE house_specs AS SELECT *, "
            "ruang_makan = 'Ya' AS ruang_makan_available, "
            "ruang_tamu = 'Ya' AS ruang_tamu_available, "
            "terjangkau_internet = 'Ya' AS terjangkau_internet_available, "
            "hook = 'Ya' AS hook_available "
            "FROM house_specs")

        # drop original columns
        for col in ["ruang_makan", "ruang_tamu", "terjangkau_internet", "hook"]:
            self.conn.sql(f"ALTER TABLE house_specs DROP COLUMN {col}")

        # rename columns
        rename_cols = {
            "ruang_makan_available": "ruang_makan",
            "ruang_tamu_available": "ruang_tamu",
            "terjangkau_internet_available": "terjangkau_internet",
            "hook_available": "hook",
        }

        for col_old, col_new in rename_cols.items():
            self.conn.sql(
                f"ALTER TABLE house_specs RENAME {col_old} TO {col_new}")

    def transform_spec_material(self):
        material_map = {
            "material_bangunan": "house_material",
            "material_lantai": "house_floor_material"
        }

        for col in ["material_bangunan", "material_lantai"]:
            # create new table
            self.conn.sql(
                f"CREATE OR REPLACE TABLE {material_map[col]} AS SELECT "
                f"unnest(string_split(lower({col}), ',')) AS material, reference_id "
                "FROM house_specs")

            # replace spaces
            self.conn.sql(
                f"CREATE OR REPLACE TABLE {material_map[col]} AS SELECT "
                "replace(trim(material), ' ', '_') AS material, reference_id "
                f"FROM {material_map[col]}")

            # drop original column
            self.conn.sql(f"ALTER TABLE house_specs DROP COLUMN {col}")

    def transform_specs(self):
        # PHASE 1: normalize specs column
        self.transform_spec_base()

        # PHASE 2: parse numeric columns
        self.transform_spec_numeric()

        # PHASE 3: normalize property states
        self.transform_spec_property_state()

        # PHASE 4: convert boolean columns
        self.transform_spec_boolean()

        # PHASE 5: process material_bangunan and material_lantai
        self.transform_spec_material()

        # PHASE 6: drop unused columns
        for col in ["id_iklan", "tipe_properti"]:
            self.conn.sql(f"ALTER TABLE house_specs DROP COLUMN {col}")

    def transform_tags(self):
        self.conn.sql(
            "CREATE OR REPLACE TABLE house_tags AS SELECT unnest(raw.tags) AS tag, id AS reference_id FROM raw"
        )

        self.conn.sql(
            "CREATE OR REPLACE TABLE house_tags AS SELECT "
            "unnest(string_split(replace(lower(tag), ' ', '_'), '/')) AS tag, reference_id "
            "FROM house_tags")

    def transform_facilities(self):
        self.conn.sql(
            "CREATE OR REPLACE TABLE house_facilities AS "
            "SELECT reference_id, CLEAN_FACILITY(facility_raw) AS facility "
            "FROM (SELECT unnest(raw.facilities) AS facility_raw, id AS reference_id FROM raw)"
        )

    def transform_images(self):
        self.conn.sql(
            "CREATE TABLE house_images AS SELECT unnest(raw.images) AS url, id AS reference_id FROM raw"
        )

    def transform_agent(self):
        self.conn.sql(
            "CREATE OR REPLACE TABLE house_agent_raw AS SELECT agent.*, id AS reference_id FROM raw"
        )

        self.conn.sql(
            "CREATE OR REPLACE TABLE house_agent_company AS "
            "SELECT CLEAN_AGENCY_COMPANY(company.name) AS name, company.url, reference_id "
            "FROM house_agent_raw")

        self.conn.sql("CREATE OR REPLACE TABLE house_agent AS SELECT "
                      "mask_name(name) AS name, "
                      "mask_phone(phone) AS phone, "
                      "sha256(concat(name, phone)) AS agent_hash, "
                      "reference_id "
                      "FROM house_agent_raw")

    def transform_houses(self):
        self.conn.sql(
            "CREATE OR REPLACE TABLE houses AS "
            "SELECT id, price, installment, address, description, url, last_modified AS last_modified_at, scraped_at "
            "FROM raw")

    # MAIN ETL PHASES
    def extract(self):
        # load data from JSON files
        self.conn.sql(
            f"CREATE TABLE raw AS SELECT * FROM read_json_auto('{self.input_dir}', format='newline_delimited')"
        )

        # count rows
        result = self.conn.sql("SELECT COUNT(*) FROM raw").fetchone()[0]
        logging.info(f"Number of rows extracted: {result}")

    def transform(self):
        logging.info("Transforming specs...")
        self.transform_specs()

        logging.info("Transforming tags...")
        self.transform_tags()

        logging.info("Transforming facilities...")
        self.transform_facilities()

        logging.info("Transforming images...")
        self.transform_images()

        logging.info("Transforming agent...")
        self.transform_agent()

        logging.info("Transforming houses...")
        self.transform_houses()

    def load(self):
        # tables to export
        tables = [
            "houses", "house_specs", "house_agent", "house_agent_company",
            "house_images", "house_material", "house_floor_material",
            "house_specs", "house_tags", "house_facilities"
        ]

        # export tables to Parquet
        for table in tables:
            self.conn.sql(
                f"COPY {table} TO '{self.output_dir}/L1.{table}.parquet' (FORMAT 'parquet')"
            )


if __name__ == "__main__":
    # setup command-line arguments
    parser = argparse.ArgumentParser()

    parser.add_argument("--input-dir",
                        help="Input directory containing JSON files",
                        default="./dataset/all/houses-20k.json")
    parser.add_argument("--output-dir",
                        help="Output directory for Parquet files",
                        default="./dataset/etl")

    args = parser.parse_args()

    # run pipeline
    pipeline = HouseL1Pipeline(args.input_dir, args.output_dir)
    pipeline.run()
