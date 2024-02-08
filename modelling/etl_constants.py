#
# OpenStreetMap APIs (Overpass, Nominatim)
#

OVERPASS_DEFAULT_BASE_URL = "https://overpass-api.de"
OVERPASS_DEFAULT_TIMEOUT = 360
OVERPASS_DEFAULT_DOWNLOAD_SLEEP = 5
OVERPASS_DEFAULT_RELATIONS = ["14745927", "14762112"]
OVERPASS_DEFAULT_AMENITIES = [
    "cafe", "restaurant", "bar", "fast_food", "college", "university",
    "kindergarten", "library", "school", "bicycle_parking", "bus_station",
    "car_rental", "car_wash", "charging_station", "fuel", "parking", "atm",
    "bank", "clinic", "dentist", "doctors", "hospital", "pharmacy",
    "veterinary", "cinema", "community_center", "events_venue",
    "conference_center", "social_centre", "theatre", "courthouse", "police",
    "fire_station", "post_office", "prison", "bench", "toilets",
    "vending_machine", "food_court"
]

OVERPASS_AMENITIES_CATEGORY = {
    "sustenance": [
        "bar", "biergarten", "cafe", "fast_food", "food_court", "ice_cream",
        "pub", "restaurant"
    ],
    "education": [
        "college", "dancing_school", "driving_school", "first_aid_school",
        "kindergarten", "language_school", "library", "surf_school",
        "toy_library", "research_institute", "training", "music_school",
        "school", "traffic_park", "university"
    ],
    "transportation": [
        "bicycle_parking", "bicycle_repair_station", "bicycle_rental",
        "bicycle_wash", "boat_rental", "boat_sharing", "bus_station",
        "car_rental", "car_sharing", "car_wash", "compressed_air",
        "vehicle_inspection", "charging_station", "driver_training",
        "ferry_terminal", "fuel", "grit_bin", "motorcycle_parking", "parking",
        "parking_entrance", "parking_space", "taxi", "weighbridge"
    ],
    "financial": ["atm", "payment_terminal", "bank", "bureau_de_change"],
    "healthcare": [
        "baby_hatch", "clinic", "dentist", "doctors", "hospital",
        "nursing_home", "pharmacy", "social_facility", "veterinary"
    ],
    "entertainment": [
        "arts_centre", "brothel", "casino", "cinema", "community_centre",
        "conference_centre", "events_venue", "exhibition_centre", "fountain",
        "gambling", "love_hotel", "music_venue", "nightclub", "planetarium",
        "public_bookcase", "social_centre", "stripclub", "studio",
        "swingerclub", "theatre"
    ],
    "public_service": [
        "courthouse", "fire_station", "police", "post_box", "post_depot",
        "post_office", "prison", "ranger_station", "townhall"
    ],
    "facilities": [
        "bbq", "bench", "dog_toilet", "dressing_room", "drinking_water",
        "give_box", "mailroom", "parcel_locker", "shelter", "shower",
        "telephone", "toilets", "water_point", "watering_place"
    ],
    "waste_management": [
        "sanitary_dump_station", "recycling", "waste_basket", "waste_disposal",
        "waste_transfer_station"
    ],
    "others": [
        "animal_boarding", "animal_breeding", "animal_shelter",
        "animal_training", "baking_oven", "clock", "crematorium", "dive_centre",
        "funeral_hall", "grave_yard", "hunting_stand", "internet_cafe",
        "kitchen", "kneipp_water_cure", "lounger", "marketplace", "monastery",
        "photo_booth", "place_of_mourning", "place_of_worship", "public_bath",
        "public_building", "refugee_site", "vending_machine", "user defined"
    ],
}

#
# Outlier Removal Rules for Preprocessing
#

OUTLIERS_DEFAULT_IQR_THRESHOLD = 1.5
OUTLIERS_MAX_BEDROOMS = 100
OUTLIERS_MAX_LAND_AREA = 20000

#
# Selection Rules for Preprocessing
#

SELECTION_DEFAULT_MODE = "r"
SELECTION_DEFAULT_MIN_R = 0.1
SELECTION_DEFAULT_CRIT_PVALUE = 0.05
SELECTION_DROP_COLUMNS = [
    "id",
    "installment",
    "description",
    "url",
    "last_modified_at",
    "scraped_at",
    "district",
    "city",
    # # spec
    # "kamar_pembantu",
    # "kamar_mandi_pembantu",
    # "tahun_dibangun",
    # "garasi",
    # "hadap",
    # "tahun_di_renovasi",
]

#
# Normalization Rules for Preprocessing
#

FACILITY_NAME_OUTLIERS = [
    "-", "SAMPING & BELAKANG", "LISTRIK", "ATAP BAJA", "ROW JALAN",
    'BUILT IN ROBES', "R.TAMU", "STUDY", "BATH", "R. MAKAN", "SERTIFIKAT",
    "HADAP", "DEKAT", "CLOSE TO", "PROMO", "CAFE", "PET F", "BONUS", "SOFA",
    "DLL", "KURSI", "VIEW", "PINTU TOL", "STRATEGIS", "NEGO", "PAVILLION", "TV",
    "MASIH ADA", "TOKEN", "POLISHED", "KAWASAN", "LENGKAP", "ASRI", "HYPERMART",
    "POSISI", "SMOKING", "RUANG", "CLUB", "SEPARATE", "CEILING", "GROUND FLOOR",
    "SPRING BED", "MEJA"
]

FACILITY_NAME_RULESET = {
    "JET PUMP": "AIR TANAH",
    "PAM": "AIR PAM",
    "PARKIR": "PARKIR",
    "PARKING": "PARKIR",
    "GARASI": "GARASI",
    "CARPORT": "CARPORT",
    "CARPOT": "CARPORT",
    "AC": "AC",
    "AIR COND": "AC",
    "AIRCON": "AC",
    "HEATING": "HEATING",
    # places
    "AULA": "AULA",
    "LOUNGE": "AULA",
    # security
    "SECURITY": "KEAMANAN",
    "KEAMANAN": "KEAMANAN",
    "SMART DOOR": "KEAMANAN",
    "ALARM SYSTEM": "KEAMANAN",
    # outdoor
    "SPA": "HALAMAN",
    "BBQ": "HALAMAN",
    "DECK": "HALAMAN",
    "COURTYARDS": "HALAMAN",
    "HALAMAN": "HALAMAN",
    "TERAS": "HALAMAN",
    "KEBUN": "HALAMAN",
    "TAMAN": "TAMAN",
    "GARDEN": "TAMAN",
    "BERMAIN": "TAMAN",
    "ENTERTAINING": "TAMAN",
    "KIDS PARK": "TAMAN",
    "PLAYGR": "PLAYGROUND",
    "PLAY GROUND": "PLAYGROUND",
    "LAPANGAN BERMAIN": "PLAYGROUND",
    "KOI": "KOLAM IKAN",
    "KOLAM PANCING": "KOLAM IKAN",
    # sports
    "GYM": "GYM",
    "SWIMMING POOL": "KOLAM RENANG",
    "POOL INGROUND": "KOLAM RENANG",
    "JOGGING TRACK": "TRACK LARI",
    "LAPANGAN": "LAPANGAN",
    "FUTSAL": "LAPANGAN",
    # kitchen
    "KITCHEN": "DAPUR",
    "KOMPOR": "DAPUR",
    "KULKAS": "DAPUR",
    "JEMURAN": "LAUNDRY",
    "LAUNDRY": "LAUNDRY",
    "CUCI": "LAUNDRY",
    "PEMANAS AIR": "WATER HEATER",
    "HOT WATER": "WATER HEATER",
    "WATER HEATER": "WATER HEATER",
    "GAS": "GAS",
    # internet
    "INTERNET": "INTERNET",
    "WIFI": "INTERNET",
    "WI-FI": "INTERNET",
    "BROADBAND": "INTERNET",
    # pray
    "MASJID": "MASJID",
    "MUSHOLA": "MUSHOLLA",
}

COMPANY_NAME_COMMONS = [
    "BRIGHTON", "RAY WHITE", "LJ HOOKER", "XAVIER MARKS", "MR REALTY",
    "ERA FIESTA", "ASIA ONE", "RE/MAX", "ERA PROJECT", "ERA VICTORIA", "MPRO",
    "9 PRO", "58 PRO", "ATLANTIS", "BEE", "BEYOND PROPERTI", "BINTANG PROPERTI",
    "BUANA REALTY", "VISION PROPERTI", "SUNRISE", "KUNCI REALTY", "HARCOURTS",
    "GADING PRO", "EXIST", "ERA STAR", "ERA SKY", "DISCOVERY", "DREAM HOME",
    "EAGLE TREE", "ERA", "INDOHOUSE", "JPI", "MITRA PROPERTI", "MYPRO",
    "ONE REALTY", "PJ PRO", "PROFESIONAL BROKER", "PROMEX", "PROPNEX",
    "SUCCESS PROPERTI", "TUNGGADEWI LAND"
]

COMPANY_NAME_STOPWORDS = [
    "DEPOK", "TANGGERANG", "BANDUNG", "JAKARTA", "GARDEN CITY",
    "GADING SERPONG", "KELAPA GADING", "CILEUNGSI", "CENGKARENG", "BINTARO",
    "MALANG", "TANGERANG", "CIBUBUR", "PALEMBANG", "TANJUNG DUREN", "SENTUL",
    "JATIASIH", "BRANCH", "AGENCY", "INTERIOR DESIGN", "&"
]

COMPANY_NAME_INDEPENDENT = [
    "NOT IDENTIFIED", "INDEPDENDENT", "INDEPEDENT", "INDEPENDENT"
]

#
# Impute Rules for Preprocessing
#

IMPUTE_RULES = [
    # house
    {
        "col": "price",
        "method": "median"
    },
    {
        "col": "description",
        "method": "constant",
        "value": "-"
    },
    # specs
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
    # tags
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
    # house material
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
    # floor material
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
    # facility
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
]

#
# Bogor Shapefile
#

SPATIAL_OVERPASS_CRS = "EPSG:4326"
SPATIAL_GEODETIC_CRS = "EPSG:23839"
SPATIAL_SHP_DROP_COLUMNS = [
    "KDPPUM", "REMARK", "KDPBPS", "FCODE", "LUASWH", "UUPP", "SRS_ID",
    "METADATA", "KDEBPS", "KDEPUM", "KDCBPS", "KDCPUM", "KDBBPS", "KDBPUM",
    "WADMKD", "WIADKD", "WIADKC", "WIADKK", "WIADPR", "TIPADM"
]
SPATIAL_PLACE_NORM_RULES = {
    # from houses to SHP
    # "Pajajaran": "Babakan",
    # "Taman Kencana": "Babakan",
    "Babakan Madang": "Babakanmadang",
    # "Bukit Sentul": "Babakanmadang",
    "Babakan Pasar": "Babakanpasar",
    "Balumbang Jaya": "Balumbangjaya",
    "Bantar Jati": "Bantarjati",
    # "Indraprasta": "Bantarjati",
    # "Ardio": "Bogor Tengah",
    "Bojong Gede": "Bojonggede",
    "Bojong Kulur": "Bojongkulur",
    # "Cilendek": "Cilendek Barat",
    "Curug Mekar": "Curugmekar",
    "Gunung Batu": "Gunungbatu",
    "Gunung Putri": "Gunungputri",
    # "Kota Wisata": "Gunungputri",
    # "Legenda Wisata": "Gunungputri",
    # "Kranggan": "Gunungputri",
    "Gunung Sindur": "Gunungsindur",
    # "Harjamukti": "INI GA ADA DI SHP",
    "Karang Tengah": "Karangtengah",
    "Kebon Kelapa": "Kebonkalapa",
    "Kedungbadak": "Kedungbadak",
    "Kedung Halang": "Kedunghalang",
    "Leuwinanggung": "Lewinanggung",
    # "Jl Dr Semeru": "Menteng",
    "Muara Sari": "Muarasari",
    # "Bogor Nirwana Residence": "Mulyaharja",
    "Parung Panjang": "Parungpanjang",
    "Pasir Jaya": "Pasirjaya",
    "Pasir Kuda": "Pasirkuda",
    "Pasir Muncang": "Pasirmuncang",
    "Ranca Bungur": "Rancabungur",
    "Rangga Mekar": "Ranggamekar",
    "Sentul City": "Sentul",
    "Sindang Barang": "Sindangbarang",
    "Sindang Sari": "Sindangsari",
    "Situ Gede": "Situgede",
    "Tajur Halang": "Tajurhalang",
    # "Ahmadyani": "Tanahsareal",
    # "Jl A Yani": "Tanahsareal",
    "Tanah Sareal": "Tanahsareal",
    "Tegal Gundi": "Tegalgundil",
    "Tegal Gundil": "Tegalgundil",
    "Duta Pakuan": "Tegallega",

    # dedupe in SHP
    "Bantar Gebang": "Bantargebang",
}
