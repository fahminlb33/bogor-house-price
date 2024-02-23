{% set common_aggregates %}
	SELECT
		--- numerical columns
		avg(luas_tanah) AS luas_tanah,
		avg(luas_bangunan) AS luas_bangunan,
		mode(daya_listrik) AS daya_listrik,

		--- categorial columns
		mode(hadap) AS hadap,
		mode(sertifikat) AS sertifikat,
		mode(sumber_air) AS sumber_air,
		mode(pemandangan) AS pemandangan,
		mode(tipe_properti) AS tipe_properti,
		mode(konsep_dan_gaya_rumah) AS konsep_dan_gaya_rumah
	FROM
		{{ ref('int_rumah123_norm_specs') }}
{% endset %}

{% set agg_values = dbt_utils.get_query_results_as_dict(common_aggregates) %}

WITH
houses_impute AS (
	SELECT
		reference_id,

		--- numerical columns
		coalesce(kamar_tidur, 1) AS kamar_tidur,
		coalesce(kamar_mandi, 1) AS kamar_mandi,
		coalesce(kamar_pembantu, 0) AS kamar_pembantu,
		coalesce(kamar_mandi_pembantu, 0) AS kamar_mandi_pembantu,
		coalesce(dapur, 1) AS dapur,
		coalesce(luas_tanah, {{ agg_values['luas_tanah'][0] }}) AS luas_tanah,
		coalesce(luas_bangunan, {{ agg_values['luas_bangunan'][0] }}) AS luas_bangunan,
		coalesce(jumlah_lantai, 1) AS jumlah_lantai,
		coalesce(garasi, 0) AS garasi,
		coalesce(carport, 0) AS carport,
		coalesce(lebar_jalan, 0) AS lebar_jalan,
		coalesce(tahun_dibangun, 0) AS tahun_dibangun,
		coalesce(tahun_di_renovasi, 0) AS tahun_di_renovasi,
		coalesce(daya_listrik, {{ agg_values['daya_listrik'][0] }}) AS daya_listrik,

		--- categorial columns
		coalesce(hadap, '{{ agg_values["hadap"][0] }}') AS hadap,
		coalesce(sertifikat, '{{ agg_values["sertifikat"][0] }}') AS sertifikat,
		coalesce(sumber_air, '{{ agg_values["sumber_air"][0] }}') AS sumber_air,
		coalesce(pemandangan, '{{ agg_values["pemandangan"][0] }}') AS pemandangan,
		coalesce(tipe_properti, '{{ agg_values["tipe_properti"][0] }}') AS tipe_properti,
		coalesce(konsep_dan_gaya_rumah, '{{ agg_values["konsep_dan_gaya_rumah"][0] }}') AS konsep_dan_gaya_rumah,
		coalesce(kondisi_properti, 'unfurnished') AS kondisi_properti,
		coalesce(kondisi_perabotan, 'unfurnished') AS kondisi_perabotan,

		--- boolean columns
		coalesce(hook, 0) AS hook,
		coalesce(ruang_tamu, 1) AS ruang_tamu,
		coalesce(ruang_makan, 0) AS ruang_makan,
		coalesce(terjangkau_internet, 0) AS terjangkau_internet
	FROM
		{{ ref('int_rumah123_norm_specs') }}
)

SELECT * FROM houses_impute
