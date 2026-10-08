-- A schema in the registry is either made by the CSV import, or was already in the database and is only
-- described here (ingest/register.py). Removing a registered schema from the registry never touches its
-- tables.
ALTER TABLE db_schemas ADD COLUMN registered boolean NOT NULL DEFAULT false;
