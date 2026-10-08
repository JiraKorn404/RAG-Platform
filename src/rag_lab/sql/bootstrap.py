"""Make sure the database for imported tables exists, with its two roles and their limits. Safe to run
again: it creates only what is missing and changes nothing when everything is there.

It runs as the admin role (the credentials in METRICS_DATABASE_URL) and is the only thing that does. It
is called by `python -m rag_lab.setup`, which the `setup` service runs when the stack starts."""

import os

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from rag_lab.sql.database import DATABASE, LOADER, READER, loader_url, reader_url

# What the reader may be. It can only select (it is granted nothing else), and these stop a bad query
# from holding a connection or the server for long.
READER_SETTINGS = {
    "default_transaction_read_only": "on",
    "statement_timeout": "15s",
    "idle_in_transaction_session_timeout": "30s",
}
LOADER_SETTINGS = {"statement_timeout": "10min"}
READER_CONNECTIONS = 10


def _admin_conninfo(dbname: str) -> str:
    return make_conninfo(os.environ["METRICS_DATABASE_URL"], dbname=dbname)


def _password(url: str) -> str:
    return conninfo_to_dict(url)["password"]


def _can_log_in(url: str) -> bool:
    try:
        psycopg.connect(url, connect_timeout=5).close()
    except psycopg.OperationalError:
        return False
    return True


def ensure_database() -> list[str]:
    """Create what is missing and return what was done (empty when nothing was)."""
    done: list[str] = []
    roles = {LOADER: loader_url(), READER: reader_url()}
    role = {name: sql.Identifier(name) for name in roles}
    database = sql.Identifier(DATABASE)

    with psycopg.connect(_admin_conninfo("postgres"), autocommit=True) as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (DATABASE,)).fetchone():
            admin.execute(sql.SQL("CREATE DATABASE {}").format(database))
            done.append(f"created the database {DATABASE}")
        for name, url in roles.items():
            exists = admin.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (name,)).fetchone()
            if not exists:
                admin.execute(
                    sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(role[name], sql.Literal(_password(url)))
                )
                done.append(f"created the role {name}")
            elif not _can_log_in(url):  # the password in .env was changed
                admin.execute(
                    sql.SQL("ALTER ROLE {} PASSWORD {}").format(role[name], sql.Literal(_password(url)))
                )
                done.append(f"set a new password for the role {name}")
        admin.execute(sql.SQL("ALTER ROLE {} CONNECTION LIMIT {}").format(role[READER], sql.Literal(READER_CONNECTIONS)))
        for name, settings in ((READER, READER_SETTINGS), (LOADER, LOADER_SETTINGS)):
            for setting, value in settings.items():
                admin.execute(
                    sql.SQL("ALTER ROLE {} IN DATABASE {} SET {} = {}").format(
                        role[name], database, sql.Identifier(setting), sql.Literal(value)
                    )
                )
        # Only the two roles may connect, and the reader may do nothing else.
        admin.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(database))
        admin.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(database, role[READER]))
        admin.execute(sql.SQL("GRANT CONNECT, CREATE ON DATABASE {} TO {}").format(database, role[LOADER]))

    with psycopg.connect(_admin_conninfo(DATABASE), autocommit=True) as admin:
        # Nothing lives in `public`: data goes in schemas the loader makes and the reader is granted.
        admin.execute("REVOKE ALL ON SCHEMA public FROM PUBLIC")
    return done
