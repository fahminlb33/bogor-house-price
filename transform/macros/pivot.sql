{% macro crosstab(table_name, column_name, colum_prefix, keep_cols) %}
	{%- set unique_values = dbt_utils.get_column_values(table=ref(table_name), column=column_name, default=[]) -%}

	SELECT
		{% for column in keep_cols %}
			{{ column }},
		{% endfor %}

		{% for column in unique_values %}
			{% if column %}
				{{ "CASE {} WHEN '{}' THEN 1 ELSE 0 END AS {}{}".format(column_name, column, colum_prefix, column) }}{{ "," if not loop.last }}
			{% endif %}
		{% endfor %}
	FROM
		{{ ref(table_name) }}
{% endmacro %}
