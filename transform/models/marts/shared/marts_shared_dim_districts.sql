{{ config(materialized='external', format='csv') }}

WITH
stg_houses AS (
    SELECT
        district,
        city
    FROM
        {{ ref("stg_rumah123_houses") }}
    GROUP BY
        district,
        city
),
final AS (
    SELECT
        -- key
        {{ dbt_utils.generate_surrogate_key(['district', 'city']) }} AS district_sk,

        -- attributes
        district,
        city
    FROM
        stg_houses
)

SELECT * FROM final
