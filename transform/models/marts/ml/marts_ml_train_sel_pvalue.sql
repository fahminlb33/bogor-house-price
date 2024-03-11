{{ config(materialized='external', format='parquet') }}

{%- set P_VALUE_REJECT = 0.05 -%}
{%- set EXCLUDE_COLUMNS = ['id', 'installment', 'district', 'city', 'description', 'url', 'last_modified', 'scraped_at'] -%}

{%- set query_significant_features -%}
	SELECT variable FROM {{ ref('int_ml_feature_correlations') }} WHERE p_value < {{ P_VALUE_REJECT }}
{%- endset -%}

{%- set results = run_query(query_significant_features) -%}
{%- if execute -%}
	{%- set significant_features = results.columns[0].values() -%}
{%- else -%}
	{%- set significant_features = [] -%}
{%- endif -%}

WITH base AS (
	SELECT
		-- target
		price,

		-- features
		{% for feature in significant_features %}
			{%- if "spatial" not in feature and feature not in EXCLUDE_COLUMNS -%}
				{{ feature }}{{ ', ' if not loop.last }}
			{%- endif -%}
		{% endfor %}
	FROM
		{{ ref('int_ml_feature_outlier_removal') }}
)
SELECT
	*
FROM
	base
