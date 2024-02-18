{% macro bool_to_int(column_name) %}
	CASE {{ column_name }}
		WHEN true THEN 1
		WHEN false THEN 0
		ELSE 0 END
{% endmacro %}

{% macro normalize_kondisi(column_name) %}
	CASE {{ column_name }}
		WHEN 'bagus sekali' THEN 'furnished'
		WHEN 'sudah renovasi' THEN 'furnished'
		WHEN 'butuh renovasi' THEN 'unfurnished'
		WHEN 'bagus' THEN 'furnished'
		WHEN 'baru' THEN 'furnished' ELSE {{ column_name }} END
{% endmacro %}

{% macro normalize_sertifikat(column_name) %}
	CASE
		WHEN {{ column_name }} LIKE 'PPJB' THEN 'Lainnya'
		WHEN {{ column_name }} LIKE 'Lainnya' THEN 'Lainnya'
		ELSE {{ column_name }} END
{% endmacro %}
