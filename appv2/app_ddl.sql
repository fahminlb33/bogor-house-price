-- rumah_bogor.marts_dashboard_fact_price definition

CREATE TABLE `marts_dashboard_fact_price` (
  `district_sk` varchar(50) DEFAULT NULL,
  `price` double DEFAULT NULL,
  `installment` double DEFAULT NULL,
  KEY `marts_dashboard_fact_price_district_sk_IDX` (`district_sk`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


-- rumah_bogor.marts_dashboard_fact_price_ratio definition

CREATE TABLE `marts_dashboard_fact_price_ratio` (
  `district_sk` varchar(50) DEFAULT NULL,
  `price_per_land_area` double DEFAULT NULL,
  `price_per_building_area` double DEFAULT NULL,
  KEY `marts_dashboard_fact_price_ratio_district_sk_IDX` (`district_sk`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


-- rumah_bogor.marts_ml_correlations definition

CREATE TABLE `marts_ml_correlations` (
  `variable` varchar(50) NOT NULL,
  `method` varchar(50) DEFAULT NULL,
  `correlation` double DEFAULT NULL,
  `p_value` double DEFAULT NULL,
  PRIMARY KEY (`variable`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


-- rumah_bogor.marts_shared_dim_districts definition

CREATE TABLE `marts_shared_dim_districts` (
  `district_sk` varchar(50) DEFAULT NULL,
  `district` varchar(50) DEFAULT NULL,
  `city` varchar(50) DEFAULT NULL,
  KEY `marts_shared_dim_districts_district_sk_IDX` (`district_sk`) USING BTREE
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


-- rumah_bogor.marts_spatial_amenities definition

CREATE TABLE `marts_spatial_amenities` (
  `id` varchar(50) NOT NULL,
  `rel` int(11) DEFAULT NULL,
  `amenity` varchar(50) DEFAULT NULL,
  `object_type` varchar(50) DEFAULT NULL,
  `category` varchar(50) DEFAULT NULL,
  `lat` double DEFAULT NULL,
  `lon` double DEFAULT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


-- rumah_bogor.predictions definition

CREATE TABLE `predictions` (
  `id` varchar(36) NOT NULL,
  `request` text NOT NULL,
  `predicted` float NOT NULL,
  `created_at` datetime NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


-- rumah_bogor.sessions definition

CREATE TABLE `sessions` (
  `id` varchar(36) NOT NULL,
  `session_token` varchar(36) NOT NULL,
  `created_at` datetime NOT NULL,
  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


-- rumah_bogor.chats definition

CREATE TABLE `chats` (
  `id` varchar(36) NOT NULL,
  `session_id` varchar(36) NOT NULL,
  `prompt` text NOT NULL,
  `response` text DEFAULT NULL,
  `model` varchar(100) NOT NULL,
  `role` varchar(100) NOT NULL,
  `prompt_tokens` int(11) NOT NULL,
  `completion_tokens` int(11) NOT NULL,
  `created_at` datetime NOT NULL,
  PRIMARY KEY (`id`),
  KEY `chats_sessions_FK` (`session_id`),
  CONSTRAINT `chats_sessions_FK` FOREIGN KEY (`session_id`) REFERENCES `sessions` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;


-- rumah_bogor.retrieved_documents definition

CREATE TABLE `retrieved_documents` (
  `id` varchar(36) NOT NULL,
  `city` varchar(255) NOT NULL,
  `district` varchar(255) NOT NULL,
  `price` float NOT NULL,
  `chat_id` varchar(36) NOT NULL,
  `qdrant_document_id` varchar(36) NOT NULL,
  `created_at` datetime NOT NULL,
  PRIMARY KEY (`id`),
  KEY `retrieved_documents_chats_FK` (`chat_id`),
  CONSTRAINT `retrieved_documents_chats_FK` FOREIGN KEY (`chat_id`) REFERENCES `chats` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_general_ci;
