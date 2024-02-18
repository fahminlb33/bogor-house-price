{%- set columns = adapter.get_columns_in_relation(ref('int_ml_feature_construction')) -%}
{%- set impute_rules = {
    "price": {
        "method": "median"
    },
    "description": {
        "method": "constant",
        "value": "-"
    },
    "kamar_tidur": {
        "method": "constant",
        "value": 1
    },
    "kamar_mandi": {
        "method": "constant",
        "value": 1
    },
    "sertifikat": {
        "method": "mode"
    },
    "kamar_pembantu": {
        "method": "constant",
        "value": 0
    },
    "kamar_mandi_pembantu": {
        "method": "constant",
        "value": 0
    },
    "jumlah_lantai": {
        "method": "constant",
        "value": 1
    },
    "tahun_dibangun": {
        "method": "constant",
        "value": 0
    },
    "dapur": {
        "method": "constant",
        "value": 1
    },
    "garasi": {
        "method": "constant",
        "value": 0
    },
    "pemandangan": {
        "method": "mode"
    },
    "carport": {
        "method": "constant",
        "value": 0
    },
    "hadap": {
        "method": "mode"
    },
    "sumber_air": {
        "method": "mode"
    },
    "tahun_di_renovasi": {
        "method": "constant",
        "value": 0
    },
    "konsep_dan_gaya_rumah": {
        "method": "mode"
    },
    "luas_tanah": {
        "method": "mean"
    },
    "luas_bangunan": {
        "method": "mean"
    },
    "daya_listrik": {
        "method": "mode"
    },
    "lebar_jalan": {
        "method": "constant",
        "value": 0
    },
    "kondisi_properti": {
        "method": "constant",
        "value": "unfurnished"
    },
    "kondisi_perabotan": {
        "method": "constant",
        "value": "unfurnished"
    },
    "ruang_makan": {
        "method": "constant",
        "value": 0
    },
    "ruang_tamu": {
        "method": "constant",
        "value": 0
    },
    "terjangkau_internet": {
        "method": "constant",
        "value": 0
    },
    "hook": {
        "method": "constant",
        "value": 0
    },
    "tag_bisa_nego": {
        "method": "constant",
        "value": 0
    },
    "tag_cash_bertahap": {
        "method": "constant",
        "value": 0
    },
    "tag_cash_keras": {
        "method": "constant",
        "value": 0
    },
    "tag_dijual_cepat": {
        "method": "constant",
        "value": 0
    },
    "tag_komplek": {
        "method": "constant",
        "value": 0
    },
    "tag_kpr": {
        "method": "constant",
        "value": 0
    },
    "tag_masuk_gang": {
        "method": "constant",
        "value": 0
    },
    "tag_one_gate_system": {
        "method": "constant",
        "value": 0
    },
    "tag_pedesaan": {
        "method": "constant",
        "value": 0
    },
    "tag_perumahan": {
        "method": "constant",
        "value": 0
    },
    "tag_pinggir_jalan": {
        "method": "constant",
        "value": 0
    },
    "house_mat_bata_hebel": {
        "method": "constant",
        "value": 0
    },
    "house_mat_bata_merah": {
        "method": "constant",
        "value": 1
    },
    "house_mat_batako": {
        "method": "constant",
        "value": 0
    },
    "house_mat_beton": {
        "method": "constant",
        "value": 0
    },
    "floor_mat_granit": {
        "method": "constant",
        "value": 0
    },
    "floor_mat_keramik": {
        "method": "constant",
        "value": 1
    },
    "floor_mat_marmer": {
        "method": "constant",
        "value": 0
    },
    "floor_mat_ubin": {
        "method": "constant",
        "value": 0
    },
    "floor_mat_vinyl": {
        "method": "constant",
        "value": 0
    },
    "facility_ac": {
        "method": "constant",
        "value": 0
    },
    "facility_air_pam": {
        "method": "constant",
        "value": 0
    },
    "facility_air_tanah": {
        "method": "constant",
        "value": 0
    },
    "facility_aula": {
        "method": "constant",
        "value": 0
    },
    "facility_balcony": {
        "method": "constant",
        "value": 0
    },
    "facility_canopy": {
        "method": "constant",
        "value": 0
    },
    "facility_carport": {
        "method": "constant",
        "value": 0
    },
    "facility_dapur": {
        "method": "constant",
        "value": 0
    },
    "facility_dishwasher": {
        "method": "constant",
        "value": 0
    },
    "facility_floorboards": {
        "method": "constant",
        "value": 0
    },
    "facility_garasi": {
        "method": "constant",
        "value": 0
    },
    "facility_gas": {
        "method": "constant",
        "value": 0
    },
    "facility_gym": {
        "method": "constant",
        "value": 0
    },
    "facility_halaman": {
        "method": "constant",
        "value": 0
    },
    "facility_heating": {
        "method": "constant",
        "value": 0
    },
    "facility_internet": {
        "method": "constant",
        "value": 0
    },
    "facility_jalur_telepon": {
        "method": "constant",
        "value": 0
    },
    "facility_keamanan": {
        "method": "constant",
        "value": 0
    },
    "facility_kolam_ikan": {
        "method": "constant",
        "value": 0
    },
    "facility_kolam_renang": {
        "method": "constant",
        "value": 0
    },
    "facility_lapangan": {
        "method": "constant",
        "value": 0
    },
    "facility_laundry": {
        "method": "constant",
        "value": 0
    },
    "facility_lemari_pakaian": {
        "method": "constant",
        "value": 0
    },
    "facility_lemari_sepatu": {
        "method": "constant",
        "value": 0
    },
    "facility_masjid": {
        "method": "constant",
        "value": 0
    },
    "facility_mezzanine": {
        "method": "constant",
        "value": 0
    },
    "facility_musholla": {
        "method": "constant",
        "value": 0
    },
    "facility_one_gate_system": {
        "method": "constant",
        "value": 0
    },
    "facility_parkir": {
        "method": "constant",
        "value": 0
    },
    "facility_playground": {
        "method": "constant",
        "value": 0
    },
    "facility_shed": {
        "method": "constant",
        "value": 0
    },
    "facility_taman": {
        "method": "constant",
        "value": 0
    },
    "facility_wastafel": {
        "method": "constant",
        "value": 0
    },
    "facility_water_heater": {
        "method": "constant",
        "value": 0
    },
    "facility_water_tank": {
        "method": "constant",
        "value": 0
    }
} -%}


SELECT
	{% for column in columns %}
		{%- set rule = impute_rules[column.name] -%}
		{%- if rule -%}
			{% if rule['method'] == 'constant' %}
				-- fill {{ column.name }} with constant value '{{ rule['value'] }}'
				{% set col_value = "'{}'".format(rule["value"]) if rule["value"] is string else rule["value"] %}
				coalesce(orig.{{ column.name }}, {{ col_value }}) AS {{ column.name }}{{ "," if not loop.last }}

			{% elif rule['method'] == 'mode' %}
				-- fill {{ column.name }} with mode value
				{%- set col_mode_query -%}
					SELECT mode({{ column.name }}) FROM {{ ref('int_ml_feature_construction') }}
				{%- endset -%}
				{%- set col_mode = dbt_utils.get_single_value(col_mode_query) -%}
				{% set col_value = "'{}'".format(col_mode) if col_mode is string else col_mode %}

				coalesce(orig.{{ column.name }}, {{ col_value }}) AS {{ column.name }}{{ "," if not loop.last }}

			{% elif rule['method'] == 'mean' %}
				-- fill {{ column.name }} with mean value
				{%- set col_avg_query -%}
					SELECT avg({{ column.name }}) FROM {{ ref('int_ml_feature_construction') }}
				{%- endset -%}
				{% set col_avg = dbt_utils.get_single_value(col_avg_query) %}

				coalesce(orig.{{ column.name }}, {{ col_avg }}) AS {{ column.name }}{{ "," if not loop.last }}

			{% elif rule['method'] == 'median' %}
				-- fill {{ column.name }} with median value
				{%- set col_median_query -%}
					SELECT median({{ column.name }}) FROM {{ ref('int_ml_feature_construction') }}
				{%- endset -%}
				{% set col_median = dbt_utils.get_single_value(col_median_query) %}

				coalesce(orig.{{ column.name }}, {{ col_median }}) AS {{ column.name }}{{ "," if not loop.last }}
			{% endif %}
		{%- else -%}
			-- keep {{ column.name }} as is

			orig.{{ column.name }}{{ "," if not loop.last }}
		{%- endif -%}
	{% endfor %}
FROM
	{{ ref('int_ml_feature_construction') }} AS orig
