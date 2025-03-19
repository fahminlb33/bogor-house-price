import pathlib
import argparse

import tqdm
import duckdb


def main(args):
    # create output dir
    output_path = pathlib.Path(args.output_path)
    output_path.mkdir(parents=True, exist_ok=True)

    # open duckdb
    with duckdb.connect(args.duckdb_path, read_only=False) as conn:
        # connect postgres
        conn.execute(
            f"ATTACH '{args.pg_uri}' AS pg_db (TYPE postgres, SCHEMA 'public')"
        )

        # get all tables
        print("Load: L3 tables to postgres...")
        tables = conn.execute("SHOW TABLES").df()["name"].tolist()
        for table in (pbar := tqdm.tqdm(tables)):
            # only import marts (L3) tables
            if not table.startswith("marts"):
                continue

            pbar.set_description(table)

            # drop target table in postgres if exists and recreate
            conn.execute(f"DROP TABLE IF EXISTS pg_db.{table}")
            conn.execute(f"CREATE TABLE pg_db.{table} AS SELECT * FROM {table}")

        # load marts_houses_downstream to parquet
        print("Load: L3 tables to parquet...")
        conn.execute(
            f"COPY marts_houses_downstream TO '{output_path / 'marts_houses_downstream.parquet'}'"
        )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--duckdb-path",
        type=str,
        help="DuckDB database file path",
        default="../data/rumah.duckdb",
    )
    parser.add_argument(
        "--output-path",
        type=str,
        help="Output directory for data files",
        default="../data/curated",
    )
    parser.add_argument(
        "--pg-uri",
        type=str,
        help="Postgres DB URI",
        default="dbname=rumah-db user=rumah-user password=rumah-password host=127.0.0.1 port=5432",
    )

    args = parser.parse_args()
    main(args)
