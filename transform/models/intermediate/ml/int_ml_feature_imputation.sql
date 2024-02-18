{%- set IMPUTE_RULES = [
    {
        "col": "price",
        "method": "median"
    },
    {
        "col": "description",
        "method": "constant",
        "value": "-"
    },
    {
        "col": "kamar_tidur",
        "method": "constant",
        "value": 1
    },
    {
        "col": "kamar_mandi",
        "method": "constant",
        "value": 1
    },
    {
        "col": "sertifikat",
        "method": "mode"
    },
    {
        "col": "kamar_pembantu",
        "method": "constant",
        "value": 0
    },
    {
        "col": "kamar_mandi_pembantu",
        "method": "constant",
        "value": 0
    },
    {
        "col": "jumlah_lantai",
        "method": "constant",
        "value": 1
    },
    {
        "col": "tahun_dibangun",
        "method": "constant",
        "value": 0
    },
    {
        "col": "dapur",
        "method": "constant",
        "value": 1
    },
    {
        "col": "garasi",
        "method": "constant",
        "value": 0
    },
    {
        "col": "pemandangan",
        "method": "mode"
    },
    {
        "col": "carport",
        "method": "constant",
        "value": 0
    },
    {
        "col": "hadap",
        "method": "mode"
    },
    {
        "col": "sumber_air",
        "method": "mode"
    },
    {
        "col": "tahun_di_renovasi",
        "method": "constant",
        "value": 0
    },
    {
        "col": "konsep_dan_gaya_rumah",
        "method": "mode"
    },
    {
        "col": "luas_tanah",
        "method": "mean"
    },
    {
        "col": "luas_bangunan",
        "method": "mean"
    },
    {
        "col": "daya_listrik",
        "method": "mode"
    },
    {
        "col": "lebar_jalan",
        "method": "constant",
        "value": 0
    },
    {
        "col": "kondisi_properti",
        "method": "constant",
        "value": "unfurnished"
    },
    {
        "col": "kondisi_perabotan",
        "method": "constant",
        "value": "unfurnished"
    },
    {
        "col": "ruang_makan",
        "method": "constant",
        "value": 0
    },
    {
        "col": "ruang_tamu",
        "method": "constant",
        "value": 0
    },
    {
        "col": "terjangkau_internet",
        "method": "constant",
        "value": 0
    },
    {
        "col": "hook",
        "method": "constant",
        "value": 0
    },
    {
        "col": "tags_bisa_nego",
        "method": "constant",
        "value": 0
    },
    {
        "col": "tags_cash_bertahap",
        "method": "constant",
        "value": 0
    },
    {
        "col": "tags_cash_keras",
        "method": "constant",
        "value": 0
    },
    {
        "col": "tags_dijual_cepat",
        "method": "constant",
        "value": 0
    },
    {
        "col": "tags_komplek",
        "method": "constant",
        "value": 0
    },
    {
        "col": "tags_kpr",
        "method": "constant",
        "value": 0
    },
    {
        "col": "tags_masuk_gang",
        "method": "constant",
        "value": 0
    },
    {
        "col": "tags_one_gate_system",
        "method": "constant",
        "value": 0
    },
    {
        "col": "tags_pedesaan",
        "method": "constant",
        "value": 0
    },
    {
        "col": "tags_perumahan",
        "method": "constant",
        "value": 0
    },
    {
        "col": "tags_pinggir_jalan",
        "method": "constant",
        "value": 0
    },
    {
        "col": "house_mat_bata_hebel",
        "method": "constant",
        "value": 0
    },
    {
        "col": "house_mat_bata_merah",
        "method": "constant",
        "value": 1
    },
    {
        "col": "house_mat_batako",
        "method": "constant",
        "value": 0
    },
    {
        "col": "house_mat_beton",
        "method": "constant",
        "value": 0
    },
    {
        "col": "floor_mat_granit",
        "method": "constant",
        "value": 0
    },
    {
        "col": "floor_mat_keramik",
        "method": "constant",
        "value": 1
    },
    {
        "col": "floor_mat_marmer",
        "method": "constant",
        "value": 0
    },
    {
        "col": "floor_mat_ubin",
        "method": "constant",
        "value": 0
    },
    {
        "col": "floor_mat_vinyl",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_ac",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_air_pam",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_air_tanah",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_aula",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_balcony",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_canopy",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_carport",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_dapur",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_dishwasher",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_floorboards",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_garasi",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_gas",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_gym",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_halaman",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_heating",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_internet",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_jalur_telepon",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_keamanan",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_kolam_ikan",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_kolam_renang",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_lapangan",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_laundry",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_lemari_pakaian",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_lemari_sepatu",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_masjid",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_mezzanine",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_musholla",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_one_gate_system",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_parkir",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_playground",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_shed",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_taman",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_wastafel",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_water_heater",
        "method": "constant",
        "value": 0
    },
    {
        "col": "facility_water_tank",
        "method": "constant",
        "value": 0
    },
] -%}


SELECT
	{% for column in IMPUTE_RULES %}
		{% if column['method'] == 'constant' %}
			-- fill {{ column['col'] }} with constant value '{{ column['value'] }}'
			{% set col_value = "'{}'".format(column["value"]) if column["value"] is string else column["value"] %}
			coalesce({{ column['col'] }}, {{ col_value }}) AS {{ column['col'] }}{{ "," if not loop.last }}

		{% elif column['method'] == 'mode' %}
			-- fill {{ column['col'] }} with mode value
			{% set col_mode_query %}
				SELECT mode({{ column['col'] }}) FROM {{ ref('int_ml_feature_construction') }}
			{% endset %}
			{% set col_mode = dbt_utils.get_single_value(col_mode_query) %}
			{% set col_value = "'{}'".format(col_mode) if col_mode is string else col_mode %}

			coalesce({{ column['col'] }}, {{ col_value }}) AS {{ column['col'] }}{{ "," if not loop.last }}

		{% elif column['method'] == 'mean' %}
			-- fill {{ column['col'] }} with mean value
			{% set col_avg_query %}
				SELECT avg({{ column['col'] }}) FROM {{ ref('int_ml_feature_construction') }}
			{% endset %}
			{% set col_avg = dbt_utils.get_single_value(col_avg_query) %}

			coalesce({{ column['col'] }}, {{ col_avg }}) AS {{ column['col'] }}{{ "," if not loop.last }}

		{% elif column['method'] == 'mean' %}
			-- fill {{ column['col'] }} with mean value
			{% set col_median_query %}
				SELECT median({{ column['col'] }}) FROM {{ ref('int_ml_feature_construction') }}
			{% endset %}
			{% set col_median = dbt_utils.get_single_value(col_median_query) %}

			coalesce({{ column['col'] }}, {{ col_median }}) AS {{ column['col'] }}{{ "," if not loop.last }}
		{% endif %}
	{% endfor %}
FROM
	{{ ref('int_ml_feature_construction') }}
