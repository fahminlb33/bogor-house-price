{%- set cols = adapter.get_columns_in_relation(ref('stg_rumah123_specs')) -%}

WITH
house_specs_wide AS (
	SELECT
		{% for column in cols %}
			{{ "\"{}\" AS {}".format(column.name, column.name.lower().replace(' ', '_')) }}{{ "," if not loop.last }}
		{% endfor %}
	FROM
		{{ ref('stg_rumah123_specs') }}
),
house_specs AS (
	SELECT
		reference_id,

		--- parse numeric columns
		try_cast(rtrim(pv.luas_tanah, ' m²') AS FLOAT) AS luas_tanah,
		try_cast(rtrim(pv.luas_bangunan, ' m²') AS FLOAT) AS luas_bangunan,
		try_cast(rtrim(pv.lebar_jalan, ' Mobil') AS FLOAT) AS lebar_jalan,
		try_cast(replace(rtrim(lower(pv.daya_listrik), 'watt'), 'lainnya', '0') AS FLOAT) AS daya_listrik,
		try_cast(pv.kamar_tidur AS INT) AS kamar_tidur,
		try_cast(pv.kamar_mandi AS INT) AS kamar_mandi,
		try_cast(pv.kamar_pembantu AS INT) AS kamar_pembantu,
		try_cast(pv.kamar_mandi_pembantu AS INT) AS kamar_mandi_pembantu,
		try_cast(pv.dapur AS INT) AS dapur,
		try_cast(pv.jumlah_lantai AS INT) AS jumlah_lantai,
		try_cast(pv.garasi AS INT) AS garasi,
		try_cast(pv.carport AS INT) AS carport,
		try_cast(pv.tahun_dibangun AS INT) AS tahun_dibangun,
		try_cast(pv.tahun_di_renovasi AS INT) AS tahun_di_renovasi,

		--- parse property state columns
		pv.hadap,
		pv.sumber_air,
		pv.pemandangan,
		pv.tipe_properti,
		pv.konsep_dan_gaya_rumah,
		pv.material_lantai,
		pv.material_bangunan,

		--- parse property state columns
		{{ normalize_sertifikat('pv.sertifikat') }} AS sertifikat,
		{{ normalize_kondisi('pv.kondisi_properti') }} AS kondisi_properti,
		{{ normalize_kondisi('pv.kondisi_perabotan') }} AS kondisi_perabotan,

		--- parse boolean values
		pv.hook = 'Ya' AS hook,
		pv.ruang_makan = 'Ya' AS ruang_makan,
		pv.ruang_tamu = 'Ya' AS ruang_tamu,
		pv.terjangkau_internet = 'Ya' AS terjangkau_internet
	FROM
		house_specs_wide pv
),
houses_impute AS (
	SELECT
		reference_id,

		--- numerical columns
		coalesce(kamar_tidur, 1) AS kamar_tidur,
		coalesce(kamar_mandi, 1) AS kamar_mandi,
		coalesce(kamar_pembantu, 0) AS kamar_pembantu,
		coalesce(kamar_mandi_pembantu, 0) AS kamar_mandi_pembantu,
		coalesce(dapur, 1) AS dapur,
		luas_tanah,
		luas_bangunan,
		
		coalesce(jumlah_lantai, 1) AS jumlah_lantai,
		coalesce(garasi, 0) AS garasi,
		coalesce(carport, 0) AS carport,
		coalesce(lebar_jalan, 0) AS lebar_jalan,
		coalesce(tahun_dibangun, 0) AS tahun_dibangun,
		coalesce(tahun_di_renovasi, 0) AS tahun_di_renovasi,
		daya_listrik,

		--- categorial columns
		hadap,
		sertifikat,
		sumber_air,
		pemandangan,
		tipe_properti,
		konsep_dan_gaya_rumah,
		kondisi_properti,
		kondisi_perabotan,
		material_lantai,
		material_bangunan,

		--- boolean columns
		coalesce(hook, 0) AS hook,
		coalesce(ruang_tamu, 1) AS ruang_tamu,
		coalesce(ruang_makan, 0) AS ruang_makan,
		coalesce(terjangkau_internet, 0) AS terjangkau_internet
	FROM
		house_specs
)

SELECT * FROM houses_impute
