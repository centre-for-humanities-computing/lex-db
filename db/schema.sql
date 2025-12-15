-- PostgreSQL schema for Lex DB
-- 
-- This file sets up the metadata infrastructure needed BEFORE running the migration.
-- It does NOT create the articles table - that's created by pgloader during migration.
--
-- What this creates:
--   - Vector index metadata table (tracks configuration for vector indexes)
--   - Trigger functions for auto-updating timestamps
--
-- What this does NOT create:
--   - articles table (created by pgloader from SQLite schema)
--   - FTS column/indexes (added by run_migration.sh after pgloader)
--   - Vector index tables (created on-demand by create_vector_index.py)
--
-- Prerequisites:
--   - Database and user already created (use devops setup_db.sh)
--   - pgvector extension enabled (done by setup_db.sh)
--
-- Usage:
--   PGPASSWORD=$DB_PASSWORD psql -U $DB_USER -d $DB_NAME -f db/schema.sql
--
-- Or if pgvector wasn't enabled by setup_db.sh:
--   sudo -u postgres psql -d $DB_NAME -c "CREATE EXTENSION IF NOT EXISTS vector;"
--   PGPASSWORD=$DB_PASSWORD psql -U $DB_USER -d $DB_NAME -f db/schema.sql

-- Vector index metadata table
-- This table tracks configuration for all vector indexes
CREATE TABLE IF NOT EXISTS vector_index_metadata (
    index_name TEXT PRIMARY KEY,
    source_table TEXT NOT NULL,
    source_column TEXT NOT NULL,
    embedding_model TEXT NOT NULL,
    chunk_size INTEGER NOT NULL,
    chunk_overlap INTEGER NOT NULL,
    chunking_strategy TEXT NOT NULL,
    updated_at_column TEXT NOT NULL DEFAULT 'changed_at',
    created_at TIMESTAMP WITH TIME ZONE DEFAULT NOW(),
    updated_at TIMESTAMP WITH TIME ZONE DEFAULT NOW()
);

-- Trigger function for auto-updating updated_at timestamp
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = NOW();
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

-- Trigger for metadata table to auto-update the updated_at column
DROP TRIGGER IF EXISTS update_vector_index_metadata_updated_at ON vector_index_metadata;
CREATE TRIGGER update_vector_index_metadata_updated_at
    BEFORE UPDATE ON vector_index_metadata
    FOR EACH ROW
    EXECUTE FUNCTION update_updated_at_column();

-- Note: The articles table will be created by pgloader during migration
-- Note: Vector index tables will be created on-demand using create_vector_index.py
