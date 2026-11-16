import csv
import runpy
import sqlite3
from collections import Counter
from datetime import date
from pathlib import Path

import pytest

import clean_orders
import generate_orders


FIELDNAMES = clean_orders.FIELDNAMES


def _base_order() -> dict[str, str]:
    return {
        "order_date": "2025-03-01",
        "customer": "Cliente 001",
        "country": "España",
        "product": "Laptop",
        "amount": "12.34",
        "status": "paid",
    }


def _write_csv(
    path: Path,
    rows: list[dict[str, str]],
    fieldnames: list[str] | None = None,
) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(
            csv_file,
            fieldnames=fieldnames or FIELDNAMES,
            extrasaction="ignore",
        )
        writer.writeheader()
        writer.writerows(rows)
    return path


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="", encoding="utf-8") as csv_file:
        return list(csv.DictReader(csv_file))


@pytest.mark.parametrize(
    ("cents", "expected"),
    [(0, "0.00"), (1, "0.01"), (100, "1.00"), (12345, "123.45")],
)
def test_format_amount_always_writes_two_decimal_places(cents, expected):
    assert generate_orders._format_amount(cents) == expected


def test_generate_orders_is_reproducible_for_same_seed(tmp_path):
    first_path = tmp_path / "first" / "orders.csv"
    second_path = tmp_path / "second" / "orders.csv"

    first_count = generate_orders.generate_orders(first_path, seed=17, valid_orders=25)
    second_count = generate_orders.generate_orders(second_path, seed=17, valid_orders=25)

    assert first_count == second_count == 55
    assert first_path.read_bytes() == second_path.read_bytes()


def test_generate_orders_contains_expected_rows_and_intentional_errors(tmp_path):
    csv_path = tmp_path / "orders.csv"

    row_count = generate_orders.generate_orders(csv_path, seed=7, valid_orders=40)

    rows = _read_csv(csv_path)
    duplicate_count = sum(count - 1 for count in Counter(
        tuple(row[field] for field in FIELDNAMES) for row in rows
    ).values())
    invalid_date_count = sum(
        not _is_canonical_date(row["order_date"]) for row in rows
    )
    missing_amount_count = sum(not row["amount"] for row in rows)
    clean_unique_rows = {
        tuple(row[field] for field in FIELDNAMES)
        for row in rows
        if _is_canonical_date(row["order_date"]) and row["amount"]
    }

    assert csv_path.exists()
    assert row_count == len(rows) == 70
    assert duplicate_count == 12
    assert invalid_date_count == 10
    assert missing_amount_count == 8
    assert len(clean_unique_rows) == 40


def _is_canonical_date(value: str) -> bool:
    try:
        return date.fromisoformat(value).isoformat() == value
    except ValueError:
        return False


def test_generate_orders_creates_missing_parent_directories(tmp_path):
    output = tmp_path / "nested" / "folder" / "orders.csv"

    count = generate_orders.generate_orders(output, valid_orders=1)

    assert output.is_file()
    assert count == 4


@pytest.mark.parametrize("valid_orders", [0, -1])
def test_generate_orders_rejects_non_positive_order_count(tmp_path, valid_orders):
    with pytest.raises(ValueError, match="mayor que cero"):
        generate_orders.generate_orders(tmp_path / "orders.csv", valid_orders=valid_orders)


def test_generate_orders_retries_when_random_data_collides(tmp_path, monkeypatch):
    class CollidingRandom:
        def __init__(self):
            self.day_calls = 0

        def randrange(self, stop):
            self.day_calls += 1
            return 0 if self.day_calls <= 2 else 1

        def randint(self, start, stop):
            return start

        def choice(self, values):
            return values[0]

        def sample(self, population, count):
            return list(population)[:count]

        def shuffle(self, values):
            return None

    monkeypatch.setattr(generate_orders.random, "Random", lambda seed: CollidingRandom())
    output = tmp_path / "orders.csv"

    generate_orders.generate_orders(output, valid_orders=2)

    rows = _read_csv(output)
    valid_rows = [row for row in rows if _is_canonical_date(row["order_date"]) and row["amount"]]
    assert len(valid_rows) == 4
    assert len({tuple(row[field] for field in FIELDNAMES) for row in valid_rows}) == 2


def test_generate_orders_cli_accepts_output_and_count(tmp_path, monkeypatch, capsys):
    output = tmp_path / "cli-orders.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["generate_orders.py", "--output", str(output), "--valid-orders", "5"],
    )

    generate_orders.main()

    assert len(_read_csv(output)) == 20
    assert "20 filas" in capsys.readouterr().out


def test_generate_orders_script_entry_point(tmp_path, monkeypatch):
    output = tmp_path / "entrypoint-orders.csv"
    monkeypatch.setattr(
        "sys.argv",
        ["generate_orders.py", "--output", str(output), "--valid-orders", "2"],
    )

    runpy.run_path(generate_orders.__file__, run_name="__main__")

    assert len(_read_csv(output)) == 8


def test_read_clean_orders_accepts_iso_date_and_converts_amount_to_cents(tmp_path):
    csv_path = _write_csv(tmp_path / "orders.csv", [_base_order()])

    rows, stats = clean_orders.read_clean_orders(csv_path)

    assert rows == [("2025-03-01", "Cliente 001", "España", "Laptop", 1234, "paid")]
    assert stats == {
        "read": 1,
        "duplicates": 0,
        "invalid_dates": 0,
        "missing_amounts": 0,
        "invalid_amounts": 0,
        "loaded": 1,
    }


@pytest.mark.parametrize(
    "order_date",
    [
        "",
        "01/03/2025",
        "2025/03/01",
        "2025-02-30",
        "2025-3-1",
        "20250301",
        "2025-03-01T12:00:00",
    ],
)
def test_read_clean_orders_skips_noncanonical_or_invalid_dates(tmp_path, order_date):
    order = _base_order()
    order["order_date"] = order_date
    csv_path = _write_csv(tmp_path / "orders.csv", [order])

    rows, stats = clean_orders.read_clean_orders(csv_path)

    assert rows == []
    assert stats["invalid_dates"] == 1
    assert stats["loaded"] == 0


@pytest.mark.parametrize("amount", ["", "   "])
def test_read_clean_orders_counts_missing_amounts(tmp_path, amount):
    order = _base_order()
    order["amount"] = amount
    rows, stats = clean_orders.read_clean_orders(
        _write_csv(tmp_path / "orders.csv", [order])
    )

    assert rows == []
    assert stats["missing_amounts"] == 1


@pytest.mark.parametrize("amount", ["abc", "NaN", "Infinity", "-Infinity", "1.234", "1,23"])
def test_read_clean_orders_skips_invalid_amounts(tmp_path, amount):
    order = _base_order()
    order["amount"] = amount
    rows, stats = clean_orders.read_clean_orders(
        _write_csv(tmp_path / "orders.csv", [order])
    )

    assert rows == []
    assert stats["invalid_amounts"] == 1
    assert stats["loaded"] == 0


@pytest.mark.parametrize(
    ("amount", "expected_cents"),
    [("0.00", 0), ("12", 1200), ("12.3", 1230), ("-1.25", -125)],
)
def test_read_clean_orders_accepts_valid_decimal_amounts(
    tmp_path, amount, expected_cents
):
    order = _base_order()
    order["amount"] = amount

    rows, stats = clean_orders.read_clean_orders(
        _write_csv(tmp_path / "orders.csv", [order])
    )

    assert rows[0][4] == expected_cents
    assert stats["invalid_amounts"] == 0


def test_read_clean_orders_strips_whitespace_and_accepts_utf8_bom(tmp_path):
    path = tmp_path / "orders.csv"
    row = _base_order()
    row["customer"] = "  Cliente 001  "
    _write_csv(path, [row])
    path.write_text(path.read_text(encoding="utf-8"), encoding="utf-8-sig")

    rows, stats = clean_orders.read_clean_orders(path)

    assert rows[0][1] == "Cliente 001"
    assert stats["loaded"] == 1


def test_read_clean_orders_deduplicates_rows_after_trimming_fields(tmp_path):
    first = _base_order()
    second = _base_order()
    second["customer"] = " Cliente 001 "
    second["amount"] = "12.34 "
    rows, stats = clean_orders.read_clean_orders(
        _write_csv(tmp_path / "orders.csv", [first, second])
    )

    assert len(rows) == 1
    assert stats["read"] == 2
    assert stats["duplicates"] == 1


@pytest.mark.parametrize("field", ["customer", "country", "product", "status"])
def test_read_clean_orders_rejects_blank_required_fields(tmp_path, field):
    order = _base_order()
    order[field] = " "

    with pytest.raises(ValueError, match="campos obligatorios vacíos"):
        clean_orders.read_clean_orders(
            _write_csv(tmp_path / "orders.csv", [order])
        )


@pytest.mark.parametrize(
    "headers",
    [
        FIELDNAMES[:-1],
        ["date", *FIELDNAMES[1:]],
        [*FIELDNAMES, "unexpected"],
    ],
)
def test_read_clean_orders_rejects_missing_reordered_or_extra_headers(
    tmp_path, headers
):
    with pytest.raises(ValueError, match="columnas"):
        clean_orders.read_clean_orders(
            _write_csv(tmp_path / "orders.csv", [_base_order()], headers)
        )


@pytest.mark.parametrize(
    "row_values",
    [
        ["2025-03-01", "Cliente 001", "España", "Laptop", "12.34"],
        ["2025-03-01", "Cliente 001", "España", "Laptop", "12.34", "paid", "extra"],
    ],
)
def test_read_clean_orders_rejects_rows_with_wrong_column_count(tmp_path, row_values):
    path = tmp_path / "malformed.csv"
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.writer(csv_file)
        writer.writerow(FIELDNAMES)
        writer.writerow(row_values)

    with pytest.raises(ValueError, match="exactamente seis columnas"):
        clean_orders.read_clean_orders(path)


def test_load_orders_creates_database_and_parent_directories(tmp_path):
    csv_path = _write_csv(tmp_path / "orders.csv", [_base_order()])
    db_path = tmp_path / "nested" / "data" / "orders.sqlite"

    stats = clean_orders.load_orders(csv_path, db_path)

    assert stats["loaded"] == 1
    assert db_path.is_file()
    with sqlite3.connect(db_path) as connection:
        order = connection.execute(
            "SELECT order_date, customer, amount_cents FROM orders"
        ).fetchone()
    assert order == ("2025-03-01", "Cliente 001", 1234)


def test_load_orders_replaces_existing_rows_on_repeated_load(tmp_path):
    csv_path = _write_csv(tmp_path / "orders.csv", [_base_order()])
    db_path = tmp_path / "orders.sqlite"
    clean_orders.load_orders(csv_path, db_path)

    replacement = _base_order()
    replacement["customer"] = "Cliente 002"
    _write_csv(csv_path, [replacement])
    stats = clean_orders.load_orders(csv_path, db_path)

    assert stats["loaded"] == 1
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
        assert connection.execute("SELECT customer FROM orders").fetchone()[0] == "Cliente 002"


def test_load_orders_creates_empty_table_for_header_only_csv(tmp_path):
    csv_path = _write_csv(tmp_path / "orders.csv", [])
    db_path = tmp_path / "orders.sqlite"

    stats = clean_orders.load_orders(csv_path, db_path)

    assert stats["read"] == stats["loaded"] == 0
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0


def test_load_orders_rolls_back_delete_when_database_schema_is_incompatible(tmp_path):
    csv_path = _write_csv(tmp_path / "orders.csv", [_base_order()])
    db_path = tmp_path / "orders.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE orders (id INTEGER PRIMARY KEY)")
        connection.execute("INSERT INTO orders (id) VALUES (42)")

    with pytest.raises(sqlite3.OperationalError):
        clean_orders.load_orders(csv_path, db_path)

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT id FROM orders").fetchall() == [(42,)]


def test_load_orders_reports_missing_csv_as_file_not_found(tmp_path):
    with pytest.raises(FileNotFoundError):
        clean_orders.load_orders(tmp_path / "missing.csv", tmp_path / "orders.sqlite")


def test_clean_orders_cli_loads_csv_and_reports_summary(tmp_path, monkeypatch, capsys):
    csv_path = _write_csv(tmp_path / "orders.csv", [_base_order()])
    db_path = tmp_path / "orders.sqlite"
    monkeypatch.setattr(
        "sys.argv",
        ["clean_orders.py", "--csv", str(csv_path), "--db", str(db_path)],
    )

    clean_orders.main()

    output = capsys.readouterr().out
    assert "Leídas: 1" in output
    assert "cargadas: 1" in output
    assert "orders.sqlite" in output


def test_clean_orders_script_entry_point(tmp_path, monkeypatch):
    csv_path = _write_csv(tmp_path / "entrypoint-orders.csv", [_base_order()])
    db_path = tmp_path / "entrypoint-orders.sqlite"
    monkeypatch.setattr(
        "sys.argv",
        ["clean_orders.py", "--csv", str(csv_path), "--db", str(db_path)],
    )

    runpy.run_path(clean_orders.__file__, run_name="__main__")

    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1
