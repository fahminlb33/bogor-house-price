{%- set query_q1 -%}
	SELECT quantile_disc(price, 0.25) FROM {{ ref('int_ml_feature_imputation') }}
{%- endset -%}
{%- set q1 = (dbt_utils.get_single_value(query_q1) or 0) | float -%}

{%- set query_q3 -%}
	SELECT quantile_disc(price, 0.75) FROM {{ ref('int_ml_feature_imputation') }}
{%- endset -%}
{%- set q3 = (dbt_utils.get_single_value(query_q3) or 0) | float -%}

{%- set iqr = q3 - q1 -%}

{%- set IQR_THRESHOLD = 1.5 -%}
{%- set MAX_BEDROOMS = 100 -%}
{%- set MAX_LAND_AREA = 20000 -%}

WITH
outliers_iqr AS (
	SELECT
		*
	FROM
		{{ ref('int_ml_feature_imputation') }}
	WHERE
		price BETWEEN {{ q1 - IQR_THRESHOLD * iqr }} AND {{ q3 + IQR_THRESHOLD * iqr }}
),
outliers_extra AS (
	SELECT
		*
	FROM
		outliers_iqr
	WHERE
		kamar_tidur < {{ MAX_BEDROOMS }} AND
		luas_tanah < {{ MAX_LAND_AREA }}
)

SELECT
	*
FROM
	outliers_extra
