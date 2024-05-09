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

		--- boolean columns
		coalesce(hook, 0) AS hook,
		coalesce(ruang_tamu, 1) AS ruang_tamu,
		coalesce(ruang_makan, 0) AS ruang_makan,
		coalesce(terjangkau_internet, 0) AS terjangkau_internet
	FROM
		{{ ref('int_rumah123_norm_specs') }}
)

SELECT * FROM houses_impute
