{%- set cols = adapter.get_columns_in_relation(ref('stg_rumah123_specs')) -%}

SELECT
	{% for column in cols %}
		{{ "\"{}\" AS {}".format(column.name, column.name.lower().replace(' ', '_')) }}{{ "," if not loop.last }}
	{% endfor %}
FROM
	{{ ref('stg_rumah123_specs') }}
