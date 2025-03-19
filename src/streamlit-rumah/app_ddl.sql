--
-- DDL for the Project Rumah Bogor
--

CREATE TABLE `sessions` (
  `id` varchar(36) NOT NULL ,
  `session_token` varchar(36) NOT NULL,
  `created_at` datetime NOT NULL,

  PRIMARY KEY (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;

CREATE TABLE `predictions` (
  `id` varchar(36) NOT NULL ,
  `request` text NOT NULL,
  `predicted` float(4) NOT NULL,
  `session_id` varchar(36) NOT NULL,
  `created_at` datetime NOT NULL,

  PRIMARY KEY (`id`),
  FOREIGN KEY (`session_id`) REFERENCES `sessions` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;

CREATE TABLE `openai_prompts` (
  `id` varchar(36) NOT NULL ,
  `prompt` varchar(1024) NOT NULL,
  `session_id` varchar(36) NOT NULL,
  `created_at` datetime NOT NULL,

  PRIMARY KEY (`id`),
  FOREIGN KEY (`session_id`) REFERENCES `sessions` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;

CREATE TABLE `openai_responses` (
  `id` varchar(36) NOT NULL ,
  `contents` text NOT NULL,
  `prompt_id` varchar(36) NOT NULL,
  `created_at` datetime NOT NULL,

  PRIMARY KEY (`id`),
  FOREIGN KEY (`prompt_id`) REFERENCES `openai_prompts` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;

CREATE TABLE `openai_usages` (
  `id` varchar(36) NOT NULL ,
  `model` varchar(50) NOT NULL,
  `usage_type` varchar(50) NOT NULL,
  `prompt_tokens` int(11) NOT NULL,
  `completion_tokens` int(11) NOT NULL,
  `total_tokens` int(11) NOT NULL,
  `prompt_id` varchar(36) NOT NULL,
  `response_id` varchar(36) NULL,
  `created_at` datetime NOT NULL,

  PRIMARY KEY (`id`),
  FOREIGN KEY (`prompt_id`) REFERENCES `openai_prompts` (`id`),
  FOREIGN KEY (`response_id`) REFERENCES `openai_responses` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;

CREATE TABLE `retrieved_documents` (
  `id` varchar(36) NOT NULL ,
  `city` varchar(255) NOT NULL,
  `district` varchar(255) NOT NULL,
  `price` float(4) NOT NULL,
  `document_id` varchar(36) NOT NULL,
  `prompt_id` varchar(36) NOT NULL,
  `created_at` datetime NOT NULL,

  PRIMARY KEY (`id`),
  FOREIGN KEY (`prompt_id`) REFERENCES `openai_prompts` (`id`)
) ENGINE=InnoDB DEFAULT CHARSET=utf8mb4 COLLATE=utf8mb4_uca1400_ai_ci;
