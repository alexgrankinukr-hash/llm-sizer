-- LLM Sizer: quantizations are withdrawn, never deleted, so "what changed" stays answerable.
-- A row gets withdrawn_at when its file disappears from the quantizer's repo; it is resurrected if the file returns.
-- Apply with `python3 scripts/llm-sizer/migrate.py`.
ALTER TABLE "llm_sizer"."model_quants" ADD COLUMN IF NOT EXISTS "withdrawn_at" timestamp with time zone;
CREATE INDEX IF NOT EXISTS "idx_llm_sizer_model_quants_active" ON "llm_sizer"."model_quants" ("model_id") WHERE "withdrawn_at" IS NULL;
