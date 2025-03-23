WITH
houses AS (
	SELECT
        *,
        regexp_split_to_array(address, ', ') AS address_parts
    FROM
        {{ source('raw_rumah123', 'houses') }}
)

SELECT
    id,
    price,
    installment,
    description,
    address_parts[1] AS subdistrict,
    address_parts[2] AS city,
    url,
    last_modified::TIMESTAMP AS last_modified,
    scraped_at::TIMESTAMP AS scraped_at
FROM
    houses
