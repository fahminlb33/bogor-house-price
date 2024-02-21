{{ config(materialized='external', format='csv', docs={'show': False}) }}

SELECT
	*
FROM
	{{ ref('marts_dashboard_correlations') }}
