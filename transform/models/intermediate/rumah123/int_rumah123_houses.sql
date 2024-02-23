{%- set query_median_price -%}
	SELECT
		median(price)
	FROM
		{{ ref('stg_rumah123_houses') }}
{%- endset -%}

{%- set median_price = dbt_utils.get_single_value(query_median_price) or 0 -%}

SELECT
    id,
    coalesce(price, {{ median_price }}) AS price,
    installment,
    district,
    city,
    coalesce(description, '') AS description,
    url,
    last_modified,
    scraped_at
FROM
    {{ ref('stg_rumah123_houses') }}
