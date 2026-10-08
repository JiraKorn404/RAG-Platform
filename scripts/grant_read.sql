-- Lets the SQL chatbot's reader role read one schema that is already in the database. Run it once for
-- each schema of `tables_database.existing_schemas` (config/connections.yaml), with psql, connected to
-- that database as the owner of the tables or an admin:
--
--   psql -h HOST -U ADMIN -d DATABASE -v schema=public -v reader=rag_reader -f scripts/grant_read.sql
--
-- The reader gets nothing else: it can only select, and only in the schemas it was granted this way.
-- A table made later is readable only if the role that runs this script makes it; run the script
-- again after another role adds tables. Then describe the tables: python -m rag_lab tables register

\set ON_ERROR_STOP on

GRANT USAGE ON SCHEMA :"schema" TO :"reader";
GRANT SELECT ON ALL TABLES IN SCHEMA :"schema" TO :"reader";
ALTER DEFAULT PRIVILEGES IN SCHEMA :"schema" GRANT SELECT ON TABLES TO :"reader";
