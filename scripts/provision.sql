-- Prepares a PostgreSQL server for RAG-Platform: the roles, the databases and our schema. It makes only
-- what is missing, so it can be run again. Run it with psql as a role that may create roles and
-- databases, connected to any database of the server (`postgres`, for example):
--
--   psql -h HOST -U ADMIN -d postgres \
--     -v app_role=rag           -v app_password=...     \
--     -v app_db=rag_metrics     -v app_schema=rag       \
--     -v tables_db=rag_data                             \
--     -v reader=rag_reader      -v reader_password=...  \
--     -v loader=rag_loader      -v loader_password=...  \
--     -v dagster_db=dagster                             \
--     -f scripts/provision.sql
--
--   app_role, app_password      the role of `app_database.url` (config/connections.yaml), which is also
--                               Dagster's (POSTGRES_USER and POSTGRES_PASSWORD in .env). It owns our schema.
--   app_db, app_schema          where our tables go: `app_database.url` and `app_database.schema`.
--   tables_db                   the database the SQL chatbot answers from (`tables_database`). It may be
--                               the same database as app_db.
--   reader, reader_password     the role of `tables_database.reader_url`: it can only select.
--   loader, loader_password     the role of `tables_database.loader_url`, for the CSV import. Leave both
--                               out where the tables are already in the database.
--   dagster_db                  Dagster's own database (DAGSTER_POSTGRES_DB in .env). Leave it out when it
--                               is made some other way.
--
-- A database that this script creates is ours, so it is closed to every other role. A database that is
-- already there is not changed, apart from the grants to our roles and our schema in it; the reader is
-- then given what it may read with scripts/grant_read.sql, one schema at a time.
--
-- The passwords go into a URL in config/connections.yaml, so use letters and digits only.

\set ON_ERROR_STOP on

-- The roles. The reader and the loader are ours alone, so their passwords are set to the ones given
-- (this is how a changed password in .env reaches the server). The app role's password is only set
-- when the role is made here.
SELECT format('CREATE ROLE %I LOGIN PASSWORD %L', :'app_role', :'app_password')
WHERE NOT EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'app_role') \gexec

SELECT format('%s ROLE %I LOGIN PASSWORD %L',
              CASE WHEN EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'reader') THEN 'ALTER' ELSE 'CREATE' END,
              :'reader', :'reader_password') \gexec

\if :{?loader}
SELECT format('%s ROLE %I LOGIN PASSWORD %L',
              CASE WHEN EXISTS (SELECT 1 FROM pg_roles WHERE rolname = :'loader') THEN 'ALTER' ELSE 'CREATE' END,
              :'loader', :'loader_password') \gexec
\endif

-- The databases. Which of them are new is decided before any is made, because app_db and tables_db
-- may be one database.
SELECT NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'app_db') AS app_db_is_new,
       NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'tables_db') AS tables_db_is_new,
       :'app_db' = :'tables_db' AS one_database \gset

\if :app_db_is_new
CREATE DATABASE :"app_db" OWNER :"app_role";
REVOKE ALL ON DATABASE :"app_db" FROM PUBLIC;
\endif

\if :tables_db_is_new
\if :one_database
\else
CREATE DATABASE :"tables_db" OWNER :"app_role";
REVOKE ALL ON DATABASE :"tables_db" FROM PUBLIC;
\endif
\endif

\if :{?dagster_db}
SELECT format('CREATE DATABASE %I OWNER %I', :'dagster_db', :'app_role')
WHERE NOT EXISTS (SELECT 1 FROM pg_database WHERE datname = :'dagster_db') \gexec
\endif

-- Our schema, owned by the app role, which makes our tables in it (python -m rag_lab setup).
\connect :"app_db"
GRANT CONNECT ON DATABASE :"app_db" TO :"app_role";
CREATE SCHEMA IF NOT EXISTS :"app_schema" AUTHORIZATION :"app_role";

-- The database the SQL chatbot reads. The reader can only select (it is granted nothing else), and
-- these stop a bad query from holding a connection or the server for long.
\connect :"tables_db"
GRANT CONNECT ON DATABASE :"tables_db" TO :"reader";
ALTER ROLE :"reader" CONNECTION LIMIT 10;
ALTER ROLE :"reader" IN DATABASE :"tables_db" SET default_transaction_read_only = on;
ALTER ROLE :"reader" IN DATABASE :"tables_db" SET statement_timeout = '15s';
ALTER ROLE :"reader" IN DATABASE :"tables_db" SET idle_in_transaction_session_timeout = '30s';

\if :{?loader}
-- The loader makes a schema for each folder of CSV files and grants it to the reader itself.
GRANT CONNECT, CREATE ON DATABASE :"tables_db" TO :"loader";
ALTER ROLE :"loader" IN DATABASE :"tables_db" SET statement_timeout = '10min';
\endif

\if :tables_db_is_new
-- Nothing of the chatbot's lives in `public` of a database of our own: data goes in the schemas the
-- loader makes.
REVOKE ALL ON SCHEMA public FROM PUBLIC;
\endif
