"""Make sure the database for imported tables exists, with its two roles and their limits. Safe to run
again: it creates only what is missing and changes nothing when everything is there.

The database and the role names are the ones in config/connections.yaml (`tables_database`). It runs as
the admin role (the one in `app_database.url`) and is the only thing that does. It is called by
`python -m rag_lab.setup`, which the `setup` service runs when the stack starts."""

import psycopg
from psycopg import sql
from psycopg.conninfo import conninfo_to_dict, make_conninfo

from rag_lab.settings import load
from rag_lab.sql.database import loader_url, reader_url, role_of

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
    return make_conninfo(load().connections.app_database.url, dbname=dbname)


def _can_log_in(url: str) -> bool:
    try:
        psycopg.connect(url, connect_timeout=5).close()
    except psycopg.OperationalError:
        return False
    return True


def ensure_database() -> list[str]:
    """Create what is missing and return what was done (empty when nothing was)."""
    done: list[str] = []
    loader_name, reader_name = role_of(loader_url()), role_of(reader_url())
    loader, reader = sql.Identifier(loader_name), sql.Identifier(reader_name)
    name = conninfo_to_dict(loader_url())["dbname"]
    database = sql.Identifier(name)

    with psycopg.connect(_admin_conninfo("postgres"), autocommit=True) as admin:
        if not admin.execute("SELECT 1 FROM pg_database WHERE datname = %s", (name,)).fetchone():
            admin.execute(sql.SQL("CREATE DATABASE {}").format(database))
            done.append(f"created the database {name}")
        for role_name, role, url in ((loader_name, loader, loader_url()), (reader_name, reader, reader_url())):
            password = sql.Literal(conninfo_to_dict(url)["password"])
            if not admin.execute("SELECT 1 FROM pg_roles WHERE rolname = %s", (role_name,)).fetchone():
                admin.execute(sql.SQL("CREATE ROLE {} LOGIN PASSWORD {}").format(role, password))
                done.append(f"created the role {role_name}")
            elif not _can_log_in(url):  # the password in .env was changed
                admin.execute(sql.SQL("ALTER ROLE {} PASSWORD {}").format(role, password))
                done.append(f"set a new password for the role {role_name}")
        admin.execute(sql.SQL("ALTER ROLE {} CONNECTION LIMIT {}").format(reader, sql.Literal(READER_CONNECTIONS)))
        for role, settings in ((reader, READER_SETTINGS), (loader, LOADER_SETTINGS)):
            for setting, value in settings.items():
                admin.execute(
                    sql.SQL("ALTER ROLE {} IN DATABASE {} SET {} = {}").format(
                        role, database, sql.Identifier(setting), sql.Literal(value)
                    )
                )
        # Only the two roles may connect, and the reader may do nothing else.
        admin.execute(sql.SQL("REVOKE ALL ON DATABASE {} FROM PUBLIC").format(database))
        admin.execute(sql.SQL("GRANT CONNECT ON DATABASE {} TO {}").format(database, reader))
        admin.execute(sql.SQL("GRANT CONNECT, CREATE ON DATABASE {} TO {}").format(database, loader))

    with psycopg.connect(_admin_conninfo(name), autocommit=True) as admin:
        # Nothing lives in `public`: data goes in schemas the loader makes and the reader is granted.
        admin.execute("REVOKE ALL ON SCHEMA public FROM PUBLIC")
    return done
