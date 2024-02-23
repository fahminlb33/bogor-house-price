{% macro bool_to_int(column_name) %}
	CASE
		WHEN {{ column_name }} = true THEN 1
		WHEN {{ column_name }} = false THEN 0
		ELSE 0 END
{% endmacro %}

{% macro normalize_kondisi(column_name) %}
	CASE
		WHEN {{ column_name }} ILIKE '%baru%' THEN 'furnished'
		WHEN {{ column_name }} ILIKE '%bagus%' THEN 'furnished'
		WHEN {{ column_name }} ILIKE '%sudah%' THEN 'furnished'
		WHEN {{ column_name }} ILIKE '%butuh%' THEN 'unfurnished'
		ELSE lower({{ column_name }}) END
{% endmacro %}

{% macro normalize_sertifikat(column_name) %}
	CASE
		WHEN {{ column_name }} ILIKE '%ppjb%' THEN 'Lainnya'
		WHEN {{ column_name }} ILIKE '%lainnya%' THEN 'Lainnya'
		ELSE {{ column_name }} END
{% endmacro %}
