-- LLM Sizer model database: filled daily from Hugging Face, exported nightly for the tool page.
-- Purely additive and isolated in its own schema so it can be extracted with `pg_dump -n llm_sizer`.
-- Apply with `python3 scripts/llm-sizer/migrate.py` (or `psql $DATABASE_URL -f migrations/0003_llm-sizer-schema.sql`).
CREATE SCHEMA IF NOT EXISTS "llm_sizer";

CREATE TABLE IF NOT EXISTS "llm_sizer"."schema_migrations" (
  "filename" text PRIMARY KEY,
  "applied_at" timestamp with time zone DEFAULT now() NOT NULL
);

-- One row per canonical base model (the maker's repo; the instruct variant when both exist).
CREATE TABLE IF NOT EXISTS "llm_sizer"."models" (
  "id" text PRIMARY KEY,                                  -- url slug, e.g. 'qwen3.8-27b'
  "name" text NOT NULL,
  "provider" text NOT NULL,                               -- maker org on Hugging Face, e.g. 'Qwen'
  "hf_repo" text NOT NULL,
  "variant" text DEFAULT 'instruct' NOT NULL,             -- instruct | base | thinking
  "released" date,
  "params_total" bigint,
  "params_active" bigint,
  "params_active_source" text,                            -- dense | estimated | override
  "context_max" integer,
  "native_bits" real,                                     -- 16 bf16 | 8 fp8 | 4 mxfp4/fp4
  "license" text,
  "license_url" text,
  "license_commercial" boolean,
  "license_note" text,
  "tags" text[] DEFAULT '{}'::text[] NOT NULL,
  "featured" boolean DEFAULT false NOT NULL,
  "featured_tier" text,                                   -- laptop | studio | frontier
  "featured_rank" integer,
  "gated" text,                                           -- 'false' | 'auto' | 'manual' | NULL unknown
  "downloads" integer,
  "status" text DEFAULT 'active' NOT NULL,                -- active | superseded | hidden | needs_token
  "superseded_by" text,
  "notes" text[] DEFAULT '{}'::text[] NOT NULL,
  "first_seen" timestamp with time zone DEFAULT now() NOT NULL,
  "last_checked" timestamp with time zone DEFAULT now() NOT NULL,
  "last_changed" timestamp with time zone DEFAULT now() NOT NULL,
  CONSTRAINT "llm_sizer_models_hf_repo_unique" UNIQUE ("hf_repo"),
  CONSTRAINT "llm_sizer_models_status_check" CHECK ("status" IN ('active', 'superseded', 'hidden', 'needs_token')),
  CONSTRAINT "llm_sizer_models_superseded_by_fk" FOREIGN KEY ("superseded_by")
    REFERENCES "llm_sizer"."models" ("id") ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS "idx_llm_sizer_models_featured" ON "llm_sizer"."models" ("featured") WHERE "featured";
CREATE INDEX IF NOT EXISTS "idx_llm_sizer_models_status" ON "llm_sizer"."models" ("status");

-- Every Hugging Face repo the job has looked at, with the verdict, so daily runs skip known repos.
CREATE TABLE IF NOT EXISTS "llm_sizer"."repos" (
  "id" text PRIMARY KEY,                                  -- 'org/name'
  "model_id" text,
  "kind" text NOT NULL,                                   -- base | precision_variant | quant | draft | rejected | pending
  "org" text NOT NULL,
  "pipeline_tag" text,
  "card_base_model" text[] DEFAULT '{}'::text[] NOT NULL,
  "gated" text,
  "downloads" integer,
  "sha" text,                                             -- Hub commit sha; unchanged => nothing to re-read
  "last_modified" timestamp with time zone,
  "created_at" timestamp with time zone,
  "reject_reason" text,
  "first_seen" timestamp with time zone DEFAULT now() NOT NULL,
  "last_checked" timestamp with time zone DEFAULT now() NOT NULL,
  CONSTRAINT "llm_sizer_repos_kind_check" CHECK ("kind" IN ('base', 'precision_variant', 'quant', 'draft', 'rejected', 'pending')),
  CONSTRAINT "llm_sizer_repos_model_id_fk" FOREIGN KEY ("model_id")
    REFERENCES "llm_sizer"."models" ("id") ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS "idx_llm_sizer_repos_model" ON "llm_sizer"."repos" ("model_id");
CREATE INDEX IF NOT EXISTS "idx_llm_sizer_repos_last_modified" ON "llm_sizer"."repos" ("last_modified");

-- What the context-cache and speed math needs, read from config.json (1:1 with models).
CREATE TABLE IF NOT EXISTS "llm_sizer"."model_architecture" (
  "model_id" text PRIMARY KEY,
  "model_type" text,
  "attention" text NOT NULL,                              -- gqa | mla | latent | assumed
  "layers" integer NOT NULL,
  "kv_layers" integer NOT NULL,                           -- layers whose cache grows with context
  "sliding_layers" integer DEFAULT 0 NOT NULL,
  "linear_layers" integer DEFAULT 0 NOT NULL,
  "kv_heads" integer,
  "head_dim" integer,
  "k_eq_v" boolean DEFAULT false NOT NULL,
  "kv_lora_rank" integer,
  "rope_dim" integer,
  "window_size" integer,
  "full_bytes_per_token_8bit" integer NOT NULL,
  "sliding_bytes_per_token_8bit" integer DEFAULT 0 NOT NULL,
  "linear_state_bytes" bigint DEFAULT 0 NOT NULL,
  "hidden_size" integer,
  "experts_total" integer,
  "experts_active" integer,
  "experts_shared" integer,
  "moe_intermediate" integer,
  "moe_layers" integer,
  "accel_mtp" boolean DEFAULT false NOT NULL,
  "accel_dflash_repo" text,
  "accel_dspark_repo" text,
  "conventional_assumed" boolean DEFAULT false NOT NULL,
  "source" text NOT NULL,                                 -- config | override | assumed
  "config" jsonb,                                         -- text_config unwrapped, quantization_config stripped
  "config_sha" text,
  "notes" text[] DEFAULT '{}'::text[] NOT NULL,
  "updated_at" timestamp with time zone DEFAULT now() NOT NULL,
  CONSTRAINT "llm_sizer_model_architecture_attention_check" CHECK ("attention" IN ('gqa', 'mla', 'latent', 'assumed')),
  CONSTRAINT "llm_sizer_model_architecture_source_check" CHECK ("source" IN ('config', 'override', 'assumed')),
  CONSTRAINT "llm_sizer_model_architecture_model_id_fk" FOREIGN KEY ("model_id")
    REFERENCES "llm_sizer"."models" ("id") ON DELETE CASCADE
);

-- One row per (quant repo, label): exact file sizes from the Hub listing.
CREATE TABLE IF NOT EXISTS "llm_sizer"."model_quants" (
  "id" bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  "model_id" text NOT NULL,
  "repo_id" text NOT NULL,
  "format" text NOT NULL,                                 -- gguf | mlx | safetensors
  "label" text NOT NULL,                                  -- 'Q4_K_M', 'UD-IQ3_XXS', 'MLX-4bit', 'FP8'
  "bits_effective" real,
  "size_bytes" bigint NOT NULL,
  "size_gb" real NOT NULL,                                -- decimal gigabytes
  "quantizer" text NOT NULL,                              -- org, or 'official' for the maker's own repo
  "file_count" integer NOT NULL,
  "qat" boolean DEFAULT false NOT NULL,
  "first_seen" timestamp with time zone DEFAULT now() NOT NULL,
  "last_checked" timestamp with time zone DEFAULT now() NOT NULL,
  "last_changed" timestamp with time zone DEFAULT now() NOT NULL,
  CONSTRAINT "llm_sizer_model_quants_repo_label_unique" UNIQUE ("repo_id", "label"),
  CONSTRAINT "llm_sizer_model_quants_format_check" CHECK ("format" IN ('gguf', 'mlx', 'safetensors')),
  CONSTRAINT "llm_sizer_model_quants_model_id_fk" FOREIGN KEY ("model_id")
    REFERENCES "llm_sizer"."models" ("id") ON DELETE CASCADE,
  CONSTRAINT "llm_sizer_model_quants_repo_id_fk" FOREIGN KEY ("repo_id")
    REFERENCES "llm_sizer"."repos" ("id") ON DELETE CASCADE
);
CREATE INDEX IF NOT EXISTS "idx_llm_sizer_model_quants_model" ON "llm_sizer"."model_quants" ("model_id");

CREATE TABLE IF NOT EXISTS "llm_sizer"."discovery_runs" (
  "id" bigint GENERATED ALWAYS AS IDENTITY PRIMARY KEY,
  "mode" text NOT NULL,                                   -- daily | backfill | only
  "started_at" timestamp with time zone DEFAULT now() NOT NULL,
  "finished_at" timestamp with time zone,
  "ok" boolean,
  "code_sha" text,
  "hf_calls" integer DEFAULT 0 NOT NULL,
  "models_seen" integer DEFAULT 0 NOT NULL,
  "models_added" integer DEFAULT 0 NOT NULL,
  "quants_added" integer DEFAULT 0 NOT NULL,
  "quants_changed" integer DEFAULT 0 NOT NULL,
  "errors" jsonb DEFAULT '[]'::jsonb NOT NULL,
  "summary" jsonb
);

-- The JSON the site serves: 'index' plus one 'model:<id>' row, each staged then promoted to live.
CREATE TABLE IF NOT EXISTS "llm_sizer"."exports" (
  "name" text NOT NULL,
  "status" text NOT NULL,                                 -- staged | live
  "body" jsonb NOT NULL,
  "sha" text NOT NULL,                                    -- sha256 prefix of the canonical body; served as the ETag
  "generated_at" timestamp with time zone DEFAULT now() NOT NULL,
  "run_id" bigint,
  CONSTRAINT "llm_sizer_exports_pkey" PRIMARY KEY ("name", "status"),
  CONSTRAINT "llm_sizer_exports_status_check" CHECK ("status" IN ('staged', 'live')),
  CONSTRAINT "llm_sizer_exports_run_id_fk" FOREIGN KEY ("run_id")
    REFERENCES "llm_sizer"."discovery_runs" ("id") ON DELETE SET NULL
);
CREATE INDEX IF NOT EXISTS "idx_llm_sizer_exports_status" ON "llm_sizer"."exports" ("status");
