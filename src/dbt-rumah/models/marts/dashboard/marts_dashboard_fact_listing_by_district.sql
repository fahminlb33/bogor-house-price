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
        {{ dbt_utils.generate_surrogate_key(['stg_houses.district', 'stg_houses.city']) }} AS district_sk,

        -- attributes
        count(*) AS listing_count
    FROM
        stg_houses
    GROUP BY
        district,
        city
)

SELECT * FROM final
