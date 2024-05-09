{{ config(materialized='external', format='parquet') }}

SELECT
	* EXCLUDE(id, installment, district, city, description, url, last_modified, scraped_at)
FROM
	{{ ref('int_ml_feature_outlier_removal') }}
