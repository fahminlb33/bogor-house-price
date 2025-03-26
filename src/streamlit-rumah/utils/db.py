from dataclasses import dataclass

import streamlit as st

import psycopg
from pgvector.psycopg import register_vector

# --------------------- DATA CLASSES ---------------------


@dataclass
class LocationStatistics:
    subdistrict: str
    district: str
    city: str
    province: str
    listing_count: int
    average_price_idr: float


@dataclass
class RetrievedDocument:
    id: str
    parent_id: str | None
    source: str
    score: float


@dataclass
class Document:
    id: str
    content: str


# --------------------- CACHED RESOURCES ---------------------


@st.cache_resource()
def get_connection():
    db_conn = psycopg.connect(st.secrets["DB_URI"])
    register_vector(db_conn)

    return db_conn


# --------------------- QUERIES ---------------------


def query_locations() -> list[LocationStatistics]:
    db_conn = get_connection()

    with db_conn.cursor() as cur:
        try:
            top_area_sql = """
                SELECT 
                    a.subdistrict,
                    a.district,
                    a.city,
                    a.province,
                    count(*) as	listing_count,
                    avg(h.price) as average_price
                FROM
                    marts_dim_area a
                INNER JOIN
                    marts_fact_houses h ON h.area_sk = a.area_sk
                GROUP BY
                    a.subdistrict, a.district, a.city , a.province
                ORDER BY 
                    listing_count desc
                LIMIT 15
                """

            cur.execute(top_area_sql)

            return [LocationStatistics(*x) for x in cur.fetchall()]
        except Exception as e:
            db_conn.rollback()
            print(e)

            raise ValueError("Error when querying the database")


def query_hybrid(
    text_query: str, text_embedding, image_mandatory: bool
) -> list[RetrievedDocument]:
    db_conn = get_connection()

    with db_conn.cursor() as cur:
        try:
            base_cte_sql = """
                base_query as (
                    SELECT h.*
                    FROM houses h
                    INNER JOIN house_images i ON i.parent_id = h.id
                ),
            """

            nearest_docs_sql = f"""
                WITH
                {base_cte_sql if image_mandatory else ""}
                bm25_query AS (
                    SELECT 
                        id, 
                        parent_id,
                        'bm25' AS source,
                        paradedb.score(id) AS score
                    FROM
                        {"base_query" if image_mandatory else "houses"}
                    WHERE
                        content @@@ %(keyword)s 
                    LIMIT 3
                ),
                embedding_query as (
                    SELECT
                        id, 
                        parent_id,
                        'embedding' AS source,
                        1 - (embedding <=> %(embedding)s::vector) AS score
                    FROM
                        {"base_query" if image_mandatory else "houses"}
                    ORDER BY
                        score DESC
                    LIMIT 3
                )
                (SELECT * FROM bm25_query LIMIT 3)
                UNION
                (SELECT * FROM embedding_query LIMIT 3)
                """

            cur.execute(
                nearest_docs_sql, {"keyword": text_query, "embedding": text_embedding}
            )

            return [RetrievedDocument(*x) for x in cur.fetchall()]
        except Exception as e:
            db_conn.rollback()
            print(e)

            raise ValueError("Error when querying the database")


def query_image(image_embedding) -> list[RetrievedDocument]:
    db_conn = get_connection()

    with db_conn.cursor() as cur:
        try:
            images_sql = """
                SELECT
                    id, 
                    parent_id,
                    'embedding' AS source,
                    1 - (embedding <=> %(embedding)s::vector) AS score
                FROM
                    house_images
                ORDER BY
                    score DESC
                LIMIT 3
                """

            cur.execute(images_sql, {"embedding": image_embedding})

            results = [RetrievedDocument(*x) for x in cur.fetchall()]

            added_ids = []
            return [x for x in results if x.parent_id not in added_ids]

        except Exception as e:
            db_conn.rollback()
            print(e)

            raise ValueError("Error when querying the database")


def query_houses(ids: list[str]) -> list[Document]:
    db_conn = get_connection()

    with db_conn.cursor() as cur:
        try:
            related_docs_sql = """
                WITH RECURSIVE
                related_houses AS (
                    SELECT
                        id,
                        content
                    FROM
                        houses
                    WHERE
                        id = ANY(%(ids)s)
                    UNION
                        SELECT
                            e.id,
                            e.content
                        FROM
                            houses e
                        INNER JOIN related_houses s ON s.id = e.parent_id
                ) 
                SELECT
                    *
                FROM
                    related_houses
                WHERE
                    length(content) > 50
                """

            norm_ids = [x.replace("-desc", "") for x in ids]
            cur.execute(related_docs_sql, {"ids": list(set(norm_ids))[:5]})

            return [Document(*x) for x in cur.fetchall()]

        except Exception as e:
            db_conn.rollback()
            print(e)

            raise ValueError("Error when querying the database")


def query_house_images(house_id: str) -> list[str]:
    db_conn = get_connection()

    with db_conn.cursor() as cur:
        try:
            related_docs_sql = """
                SELECT
                    file_path
                FROM
                    house_images
                WHERE
                    parent_id = %(house_id)s
                """

            cur.execute(related_docs_sql, {"house_id": house_id})

            return [x[0] for x in cur.fetchall()]

        except Exception as e:
            db_conn.rollback()
            print(e)

            raise ValueError("Error when querying the database")
