{% macro normalize_kondisi(column_name) %}
	CASE {{ column_name }}
		WHEN 'bagus sekali' THEN 'furnished'
		WHEN 'sudah renovasi' THEN 'furnished'
		WHEN 'butuh renovasi' THEN 'unfurnished'
		WHEN 'bagus' THEN 'furnished'
		WHEN 'baru' THEN 'furnished' ELSE {{ column_name }} END
{% endmacro %}

{% macro bool_to_int(column_name) %}
	CASE {{ column_name }}
		WHEN true THEN 1
		WHEN false THEN 0
		ELSE 0 END
{% endmacro %}
