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
    address_parts[1] AS district,
    address_parts[2] AS city,
    description,
    url,
    last_modified,
    scraped_at
FROM
    houses
