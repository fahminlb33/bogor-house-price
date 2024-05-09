SELECT
    id,
    price,
    installment,
    district,
    city,
    coalesce(description, '') AS description,
    url,
    last_modified,
    scraped_at
FROM
    {{ ref('stg_rumah123_houses') }}
WHERE
    price IS NOT NULL
