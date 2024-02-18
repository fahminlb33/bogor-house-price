WITH
house_specs_proc AS (
	SELECT
		*,

		--- parse numeric columns
		try_cast(kamar_tidur AS INT) AS kamar_tidur_num,
		try_cast(kamar_mandi AS INT) AS kamar_mandi_num,
		try_cast(kamar_pembantu AS INT) AS kamar_pembantu_num,
		try_cast(kamar_mandi_pembantu AS INT) AS kamar_mandi_pembantu_num,
		try_cast(dapur AS INT) AS dapur_num,
		try_cast(rtrim(luas_tanah, ' m²') AS FLOAT) AS luas_tanah_num,
		try_cast(rtrim(luas_bangunan, ' m²') AS FLOAT) AS luas_bangunan_num,
		try_cast(jumlah_lantai AS INT) AS jumlah_lantai_num,
		try_cast(carport AS INT) AS carport_num,
		try_cast(replace(rtrim(lower(daya_listrik), 'watt'), 'lainnya', '0') AS FLOAT) AS daya_listrik_num,
		try_cast(rtrim(lebar_jalan, ' Mobil') AS FLOAT) AS lebar_jalan_num,
		try_cast(tahun_dibangun AS INT) AS tahun_dibangun_num,
		try_cast(tahun_di_renovasi AS INT) AS tahun_di_renovasi_num,
		try_cast(garasi AS INT) AS garasi_num,

		--- parse property state columns
		lower(kondisi_properti) AS kondisi_properti_lower,
		lower(kondisi_perabotan) AS kondisi_perabotan_lower,
		lower(hadap) AS hadap_lower,

		--- parse property state columns
		{{ normalize_kondisi('lower(kondisi_properti)') }} AS kondisi_properti_norm,
		{{ normalize_kondisi('lower(kondisi_perabotan)') }} AS kondisi_perabotan_norm,

		--- parse boolean values
		ruang_makan = 'Ya' AS ruang_makan_available,
		ruang_tamu = 'Ya' AS ruang_tamu_available,
		terjangkau_internet = 'Ya' AS terjangkau_internet_available,
		hook = 'Ya' AS hook_available
	FROM
		{{ ref('int_rumah123_norm_specs') }}
)

SELECT
	reference_id,

	--- numerical columns
	kamar_tidur_num AS kamar_tidur,
	kamar_mandi_num AS kamar_mandi,
	kamar_pembantu_num AS kamar_pembantu,
	kamar_mandi_pembantu_num AS kamar_mandi_pembantu,
	dapur_num AS dapur,
	luas_tanah_num AS luas_tanah,
	luas_bangunan_num AS luas_bangunan,
	jumlah_lantai_num AS jumlah_lantai,
	carport_num AS carport,
	daya_listrik_num AS daya_listrik,
	lebar_jalan_num AS lebar_jalan,
	tahun_dibangun_num AS tahun_dibangun,
	tahun_di_renovasi_num AS tahun_di_renovasi,

	--- categorial columns
	tipe_properti AS tipe_properti,
	sertifikat AS sertifikat,
	kondisi_properti_norm AS kondisi_properti,
	kondisi_perabotan_norm AS kondisi_perabotan,
	hadap_lower AS hadap,
	konsep_dan_gaya_rumah,
	pemandangan,
	sumber_air,
	garasi_num AS garasi,

	--- boolean columns
	ruang_makan_available AS ruang_makan,
	ruang_tamu_available AS ruang_tamu,
	terjangkau_internet_available AS terjangkau_internet,
	hook_available AS hook,
FROM
	house_specs_proc
