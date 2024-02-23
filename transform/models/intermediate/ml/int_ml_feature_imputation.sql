{%- set columns = adapter.get_columns_in_relation(ref('int_ml_feature_construction')) -%}

SELECT
	{% for column in columns %}
		{% if column.name.startswith('facility_') or column.name.startswith('tag_') or column.name.startswith('floor_mat_') or column.name.startswith('house_mat_') %}
			-- fill with zeros
			coalesce(orig.{{ column.name }}, 0) AS {{ column.name }}{{ "," if not loop.last }}
		{% else %}
			-- keep {{ column.name }} as is
			orig.{{ column.name }}{{ "," if not loop.last }}
		{% endif %}
	{% endfor %}
FROM
	{{ ref('int_ml_feature_construction') }} AS orig
