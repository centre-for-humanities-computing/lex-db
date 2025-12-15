#!/bin/bash
#
# Migration script for Lex DB: SQLite to PostgreSQL
#
# This script:
# 1. Runs pgloader to migrate the articles table from SQLite to PostgreSQL
# 2. Adds the FTS (Full-Text Search) generated column and GIN index
# 3. Verifies the migration was successful
#
# Prerequisites:
# - pgloader installed (sudo apt install pgloader)
# - PostgreSQL database created and accessible
# - Environment variables set (DB_HOST, DB_PORT, DB_NAME, DB_USER, DB_PASSWORD)
#
# Usage:
#   ./db/run_migration.sh
#
# Or with custom SQLite database path:
#   SQLITE_DB=/path/to/custom.db ./db/run_migration.sh

set -e  # Exit immediately if any command fails
set -u  # Exit if undefined variable is used

# Color codes for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
BLUE='\033[0;34m'
NC='\033[0m' # No Color

# Configuration from environment variables (with defaults)
SQLITE_DB="${SQLITE_DB:-/home/au338890/repos/lex-db/db/lex_1.5.0.db}"
PG_HOST="${DB_HOST:-localhost}"
PG_PORT="${DB_PORT:-5432}"
PG_DB="${DB_NAME:-lex_db}"
PG_USER="${DB_USER:-lex_user}"
PG_PASSWORD="${DB_PASSWORD:-changeme}"

# Construct PostgreSQL connection string
PG_CONN="postgresql://${PG_USER}:${PG_PASSWORD}@${PG_HOST}:${PG_PORT}/${PG_DB}"

echo -e "${BLUE}========================================${NC}"
echo -e "${BLUE}Lex DB Migration: SQLite → PostgreSQL${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo -e "${YELLOW}Configuration:${NC}"
echo "  Source (SQLite): $SQLITE_DB"
echo "  Target (PostgreSQL): postgresql://${PG_USER}@${PG_HOST}:${PG_PORT}/${PG_DB}"
echo ""

# Verify SQLite database exists
if [ ! -f "$SQLITE_DB" ]; then
    echo -e "${RED}Error: SQLite database not found at $SQLITE_DB${NC}"
    exit 1
fi

# Count articles in SQLite (for verification)
echo -e "${YELLOW}Counting articles in SQLite database...${NC}"
SQLITE_COUNT=$(sqlite3 "$SQLITE_DB" "SELECT COUNT(*) FROM articles;")
echo -e "  SQLite articles: ${GREEN}${SQLITE_COUNT}${NC}"
echo ""

# Step 1: Run pgloader
echo -e "${YELLOW}Step 1: Running pgloader to migrate articles table...${NC}"
echo "  This may take a few minutes depending on database size."
echo "  Using configuration from: db/migrate_articles.load"
echo ""

# Update the .load file with actual connection strings using sed
TEMP_CONFIG=$(mktemp /tmp/pgloader_config.XXXXXX.load)
cp db/migrate_articles.load "$TEMP_CONFIG"

# Replace placeholder paths with actual values
sed -i "s|FROM sqlite://.*|FROM sqlite://${SQLITE_DB}|" "$TEMP_CONFIG"
sed -i "s|INTO postgresql://.*|INTO ${PG_CONN}|" "$TEMP_CONFIG"

# Run pgloader with increased memory limit
# --dynamic-space-size sets the heap size in MB (default is 1024MB)
# For 161k articles, we'll use 2GB
if pgloader --dynamic-space-size 2048 "$TEMP_CONFIG"; then
    echo -e "${GREEN}✓ pgloader completed successfully${NC}"
else
    echo -e "${RED}✗ pgloader failed${NC}"
    rm "$TEMP_CONFIG"
    exit 1
fi

# Clean up temporary config
rm "$TEMP_CONFIG"
echo ""

# Step 2: Add FTS column and index
echo -e "${YELLOW}Step 2: Adding Full-Text Search (FTS) support...${NC}"

export PGPASSWORD="$PG_PASSWORD"
psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" <<EOF
-- Add tsvector column for Full-Text Search
-- This is a generated column that automatically updates when xhtml_md changes
ALTER TABLE articles ADD COLUMN IF NOT EXISTS xhtml_md_tsv tsvector 
    GENERATED ALWAYS AS (
        to_tsvector('danish', coalesce(xhtml_md, ''))
    ) STORED;

-- Create GIN index for fast full-text search queries
-- GIN (Generalized Inverted Index) is optimized for tsvector columns
CREATE INDEX IF NOT EXISTS idx_articles_fts ON articles USING GIN(xhtml_md_tsv);

-- Also create indexes on commonly queried columns
CREATE INDEX IF NOT EXISTS idx_articles_encyclopedia ON articles(encyclopedia_id);
CREATE INDEX IF NOT EXISTS idx_articles_changed_at ON articles(changed_at);
EOF

if [ $? -eq 0 ]; then
    echo -e "${GREEN}✓ FTS column and indexes created successfully${NC}"
else
    echo -e "${RED}✗ Failed to create FTS column and indexes${NC}"
    exit 1
fi
echo ""

# Step 3: Verify migration
echo -e "${YELLOW}Step 3: Verifying migration...${NC}"

# Count articles in PostgreSQL
PG_COUNT=$(psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -t -c "SELECT COUNT(*) FROM articles;")
PG_COUNT=$(echo "$PG_COUNT" | xargs)  # Trim whitespace

echo "  PostgreSQL articles: ${GREEN}${PG_COUNT}${NC}"
echo ""

# Compare counts
if [ "$SQLITE_COUNT" -eq "$PG_COUNT" ]; then
    echo -e "${GREEN}✓ Row counts match! Migration successful.${NC}"
else
    echo -e "${RED}✗ Row count mismatch!${NC}"
    echo "  Expected: $SQLITE_COUNT"
    echo "  Got: $PG_COUNT"
    exit 1
fi

# Verify FTS is working
echo ""
echo -e "${YELLOW}Testing FTS functionality...${NC}"
FTS_COUNT=$(psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -t -c "SELECT COUNT(*) FROM articles WHERE xhtml_md_tsv IS NOT NULL;")
FTS_COUNT=$(echo "$FTS_COUNT" | xargs)

echo "  Articles with FTS data: ${GREEN}${FTS_COUNT}${NC}"

if [ "$FTS_COUNT" -gt 0 ]; then
    echo -e "${GREEN}✓ FTS is working${NC}"
else
    echo -e "${YELLOW}⚠ Warning: No articles have FTS data${NC}"
fi

# Show sample data
echo ""
echo -e "${YELLOW}Sample articles (first 3):${NC}"
psql -h "$PG_HOST" -p "$PG_PORT" -U "$PG_USER" -d "$PG_DB" -c "SELECT id, headword, encyclopedia_id, changed_at FROM articles ORDER BY id LIMIT 3;"

echo ""
echo -e "${BLUE}========================================${NC}"
echo -e "${GREEN}Migration completed successfully!${NC}"
echo -e "${BLUE}========================================${NC}"
echo ""
echo "Next steps:"
echo "  1. Test FTS search: See updated search_lex_fts() function"
echo "  2. Create vector indexes on-demand using create_vector_index.py"
echo "  3. Generate embeddings locally and upload using the new scripts"
