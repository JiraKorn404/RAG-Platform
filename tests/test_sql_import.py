from rag_lab.ingest.importer import clean_identifier, clean_names, infer_type, parse_sample, sniff_delimiter


def test_clean_identifier_makes_a_plain_name():
    assert clean_identifier("Order Date") == "order_date"
    assert clean_identifier("  Unit Price ($)  ") == "unit_price"
    assert clean_identifier("2024 sales") == "column_2024_sales"
    assert clean_identifier("!!!") == "column"
    assert clean_identifier("a" * 100) == "a" * 63


def test_clean_identifier_keeps_letters_of_any_language():
    assert clean_identifier("ชื่อ ลูกค้า") == "ชื่อ_ลูกค้า"  # Thai vowel and tone marks survive
    assert clean_identifier("Straße") == "straße"
    assert len(clean_identifier("ก" * 100).encode()) <= 63  # cut by bytes, not in the middle of a character


def test_clean_names_are_unique_and_avoid_system_columns():
    assert clean_names(["Name", "name", "NAME "]) == ["name", "name_2", "name_3"]
    assert clean_names(["xmin", "ctid"]) == ["xmin_col", "ctid_col"]
    assert len(set(clean_names(["a" * 70, "a" * 70]))) == 2


def test_infer_type_numbers():
    assert infer_type(["1", "-20", "300"]) == "bigint"
    assert infer_type(["1", "2.5", "1e3"]) == "double precision"
    assert infer_type(["007", "010"]) == "text"  # a code written with digits
    assert infer_type(["007", "1.5"]) == "text"
    assert infer_type(["0", "0.5"]) == "double precision"
    assert infer_type(["9223372036854775808"]) == "text"  # does not fit a bigint
    assert infer_type(["1,234"]) == "text"
    assert infer_type(["nan", "1"]) == "text"


def test_infer_type_ignores_empty_values_and_defaults_to_text():
    assert infer_type(["1", "", "3"]) == "bigint"
    assert infer_type(["", " "]) == "text"
    assert infer_type([]) == "text"


def test_infer_type_dates_booleans_and_timestamps():
    assert infer_type(["2024-01-31", "2023-12-01"]) == "date"
    assert infer_type(["31/01/2024"]) == "text"  # only ISO dates are guessed
    assert infer_type(["2024-02-30"]) == "text"  # not a real day
    assert infer_type(["true", "False"]) == "boolean"
    assert infer_type(["2024-01-31 10:00:00", "2024-01-31T10:00"]) == "timestamp"
    assert infer_type(["2024-01-31", "2024-01-31 10:00:00"]) == "timestamp"
    assert infer_type(["2024-01-31T10:00:00Z", "2024-01-31 10:00:00+07:00"]) == "timestamptz"
    assert infer_type(["2024-01-31T10:00:00Z", "2024-01-31 10:00:00"]) == "text"  # an offset on only one


def test_sniff_delimiter():
    assert sniff_delimiter("a,b,c\n1,2,3\n") == ","
    assert sniff_delimiter("a;b;c\n1;2;3\n") == ";"
    assert sniff_delimiter("a\tb\n1\t2\n") == "\t"
    assert sniff_delimiter("a|b\n1|2\n") == "|"
    assert sniff_delimiter("a;b\n1,5;2,5\n3,5;4\n") == ";"  # decimal commas do not make it a comma file
    assert sniff_delimiter("just one column\nanother\n") == ","


def test_parse_sample_reads_headers_names_and_types():
    sample = parse_sample("Order ID;Price;Status\n007;1,5;ok\n008;2,5;ok\n".encode())
    assert sample.delimiter == ";"
    assert sample.names == ["order_id", "price", "status"]
    assert sample.types == ["text", "text", "text"]  # leading zeros, decimal commas
    sample = parse_sample(b"id,total\n1,2.5\n2,3\n")
    assert sample.types == ["bigint", "double precision"]
