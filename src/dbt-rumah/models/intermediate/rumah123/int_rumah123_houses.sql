SELECT
    id,
    price,
    installment,
    coalesce(districts.target, district) as subdistrict,
    city,
    coalesce(description, '') AS description,
    url,
    last_modified,
    scraped_at
FROM
    {{ ref('stg_rumah123_houses') }} houses
LEFT JOIN
    {{ ref('stg_osm_districts') }} districts ON houses.district = districts.source
WHERE
    price IS NOT NULL
