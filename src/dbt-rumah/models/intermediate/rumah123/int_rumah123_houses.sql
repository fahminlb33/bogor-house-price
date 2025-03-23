SELECT
    houses.id,
    houses.price,
    houses.installment,
    coalesce(houses.description, '') AS description,
    coalesce(districts.target, houses.subdistrict) as subdistrict,
    houses.city,
    houses.url,
    houses.last_modified,
    houses.scraped_at
FROM
    {{ ref('stg_rumah123_houses') }} houses
LEFT JOIN
    -- might contains diplicate subdistricts!
    {{ ref('stg_osm_districts') }} districts ON houses.subdistrict = districts.source
WHERE
    price IS NOT NULL
