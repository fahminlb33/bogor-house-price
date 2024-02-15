WITH
stg_houses AS (
    SELECT
        *
    FROM
        {{ ref("stg_rumah123_houses") }}
),
final AS (
    SELECT
        -- key
        {{ dbt_utils.generate_surrogate_key(['stg_houses.district', 'stg_houses.city']) }} as district_key,

        -- attributes
        avg(price) AS price_avg,
        median(price) AS price_median
    FROM
        stg_houses
    GROUP BY
        district,
        city
)

SELECT * FROM final
