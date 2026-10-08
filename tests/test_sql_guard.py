import pytest

from rag_lab.serve.guard import Refused, guard

SCHEMA, TABLES, MAX_ROWS = "shop", ["orders", "customers"], 201


def ok(sql: str) -> str:
    return guard(sql, SCHEMA, TABLES, MAX_ROWS)


def refused(sql: str) -> str:
    with pytest.raises(Refused) as e:
        ok(sql)
    return str(e.value)


def test_a_select_is_written_out_schema_qualified_with_a_limit():
    assert ok("select count(*) from orders") == "SELECT COUNT(*) FROM shop.orders LIMIT 201"
    assert ok("select * from shop.orders") == "SELECT * FROM shop.orders LIMIT 201"


def test_joins_ctes_and_unions_are_fine():
    out = ok("with big as (select * from orders where total > 5) select c.name from big join customers c on c.id = big.customer_id")
    assert "shop.orders" in out and "shop.customers" in out and "FROM big" in out  # the CTE is not qualified
    assert ok("select id from orders union select id from customers").endswith("LIMIT 201")


def test_more_than_one_statement_is_refused():
    assert "exactly one" in refused("select 1; select 2")
    assert "exactly one" in refused("select * from orders; drop table orders")


@pytest.mark.parametrize(
    "sql",
    [
        "insert into orders values (1)",
        "update orders set total = 0",
        "delete from orders",
        "drop table orders",
        "truncate orders",
        "create table x (a int)",
        "alter table orders add column z int",
        "set statement_timeout = 0",
        "copy orders to program 'id'",
        "explain select * from orders",
        "values (1)",
        "grant all on orders to public",
    ],
)
def test_anything_but_a_select_is_refused(sql):
    refused(sql)


def test_select_into_locking_and_a_data_modifying_cte_are_refused():
    refused("select * into copy_of_orders from orders")
    refused("select * from orders for update")
    refused("with gone as (delete from orders returning *) select * from gone")
    refused("with new as (insert into orders values (1) returning *) select * from new")


def test_other_schemas_catalogs_and_unknown_tables_are_refused():
    assert "other" in refused("select * from other.orders")
    refused("select * from pg_catalog.pg_class")
    refused("select * from information_schema.tables")
    refused("select * from somedb.shop.orders")
    assert "secrets" in refused("select * from secrets")
    refused("select * from orders where id in (select id from other.orders)")  # inside a subquery too
    refused("select * from pg_class")


def test_functions_that_read_files_sleep_or_run_other_queries_are_refused():
    refused("select pg_read_file('/etc/passwd')")
    refused("select pg_sleep(60)")
    refused("select set_config('statement_timeout', '0', false)")
    refused("select nextval('s')")
    refused("select query_to_xml('select * from other.orders', true, true, '')")
    refused("select * from pg_ls_dir('/')")


def test_ordinary_functions_are_allowed():
    ok("select lower(name), date_trunc('month', created), coalesce(total, 0), count(*) filter (where total > 1) from orders group by 1, 2, 3")
    ok("select percentile_cont(0.5) within group (order by total) from orders")
    ok("select age(created), extract(year from created), to_char(created, 'YYYY') from orders")
    ok("select cast(total as text), total::int from orders")


def test_the_limit_is_added_kept_or_lowered():
    assert ok("select * from orders").endswith("LIMIT 201")
    assert ok("select * from orders limit 5").endswith("LIMIT 5")
    assert ok("select * from orders limit 100000").endswith("LIMIT 201")
    assert ok("select * from orders limit all").endswith("LIMIT 201")
    assert ok("select * from orders limit 5 offset 10").endswith("LIMIT 5 OFFSET 10")
    assert ok("select * from orders fetch first 5 rows only").endswith("LIMIT 5")  # read as a limit
    assert ok("select * from orders fetch first 999999 rows only").endswith("LIMIT 201")
    refused("select * from orders order by total fetch first 5 rows with ties")


def test_what_runs_is_what_was_checked():
    assert ok("select 1 -- ; drop table orders") == "SELECT 1 LIMIT 201"  # no comment reaches the database
    refused("select 1 /* x */ ; drop table orders")  # a second statement after a comment is still one
    out = ok("select ';drop table orders' as x from orders")
    assert out.count(";") == 1 and out.endswith("LIMIT 201")  # the semicolon stays inside a string
    out = ok("select 1 -- */ ; drop table orders")
    assert "drop" not in out.lower() and "*/" not in out


def test_names_follow_postgresql_folding():
    ok('select * from "orders"')
    ok("select * from ORDERS")  # unquoted names are folded to lower case
    refused('select * from "Orders"')  # a quoted name is exact


def test_an_unparsable_or_empty_query_is_refused():
    assert "could not be parsed" in refused("select from where")
    assert "no SQL" in refused("")
    assert "no SQL" in refused(";")


def test_a_schema_name_that_needs_quotes_is_quoted():
    assert guard("select * from t", "my-data", ["t"], 5) == 'SELECT * FROM "my-data".t LIMIT 5'
