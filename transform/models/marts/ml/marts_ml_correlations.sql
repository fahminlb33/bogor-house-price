{{ config(materialized='external', format='csv') }}

SELECT * FROM {{ ref('int_ml_feature_correlations') }}
