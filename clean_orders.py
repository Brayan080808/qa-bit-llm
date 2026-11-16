import argparse
import csv
import sqlite3
from datetime import date
from decimal import Decimal, InvalidOperation
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = ROOT / "data" / "orders.csv"
DEFAULT_DB_PATH = ROOT / "data" / "orders.sqlite"
FIELDNAMES = ["order_date", "customer", "country", "product", "amount", "status"]


def read_clean_orders(
    csv_path: Path,
) -> tuple[list[tuple[str, str, str, str, int, str]], dict[str, int]]:
    cleaned_rows: list[tuple[str, str, str, str, int, str]] = []
    stats = {
        "read": 0,
        "duplicates": 0,
        "invalid_dates": 0,
        "missing_amounts": 0,
        "invalid_amounts": 0,
        "loaded": 0,
    }
    seen_rows: set[tuple[str, ...]] = set()

    with csv_path.open("r", newline="", encoding="utf-8-sig") as csv_file:
        reader = csv.DictReader(csv_file)
        if reader.fieldnames != FIELDNAMES:
            raise ValueError(
                f"El CSV debe tener estas columnas, en este orden: {', '.join(FIELDNAMES)}."
            )

        for line_number, row in enumerate(reader, start=2):
            stats["read"] += 1
            if None in row or any(value is None for value in row.values()):
                raise ValueError(f"La fila {line_number} no tiene exactamente seis columnas.")

            values = tuple((row[field] or "").strip() for field in FIELDNAMES)
            if values in seen_rows:
                stats["duplicates"] += 1
                continue
            seen_rows.add(values)

            order_date, customer, country, product, amount_text, status = values
            try:
                parsed_date = date.fromisoformat(order_date)
                if parsed_date.isoformat() != order_date:
                    raise ValueError
            except ValueError:
                stats["invalid_dates"] += 1
                continue

            if not amount_text:
                stats["missing_amounts"] += 1
                continue
            try:
                amount = Decimal(amount_text)
                if not amount.is_finite() or amount != amount.quantize(Decimal("0.01")):
                    raise InvalidOperation
            except InvalidOperation:
                stats["invalid_amounts"] += 1
                continue

            if not all((customer, country, product, status)):
                raise ValueError(f"La fila {line_number} tiene campos obligatorios vacíos.")

            amount_cents = int(amount * 100)
            cleaned_rows.append(
                (parsed_date.isoformat(), customer, country, product, amount_cents, status)
            )

    stats["loaded"] = len(cleaned_rows)
    return cleaned_rows, stats


def load_orders(csv_path: Path = DEFAULT_CSV_PATH, db_path: Path = DEFAULT_DB_PATH) -> dict[str, int]:
    rows, stats = read_clean_orders(csv_path)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    connection = sqlite3.connect(db_path)
    try:
        with connection:
            connection.execute(
                """
                CREATE TABLE IF NOT EXISTS orders (
                    order_id INTEGER PRIMARY KEY,
                    order_date TEXT NOT NULL,
                    customer TEXT NOT NULL,
                    country TEXT NOT NULL,
                    product TEXT NOT NULL,
                    amount_cents INTEGER NOT NULL,
                    status TEXT NOT NULL
                )
                """
            )
            connection.execute("DELETE FROM orders")
            connection.executemany(
                """
                INSERT INTO orders (
                    order_date, customer, country, product, amount_cents, status
                ) VALUES (?, ?, ?, ?, ?, ?)
                """,
                rows,
            )
            connection.execute(
                """
                CREATE VIEW IF NOT EXISTS monthly_sales_by_country AS
                SELECT
                    substr(order_date, 1, 7) AS month,
                    country,
                    SUM(amount_cents) / 100.0 AS total_sales,
                    COUNT(*) AS order_count
                FROM orders
                WHERE status IN ('paid', 'shipped')
                GROUP BY substr(order_date, 1, 7), country
                """
            )
            connection.execute(
                """
                CREATE VIEW IF NOT EXISTS top_customers AS
                SELECT
                    customer,
                    COUNT(*) AS order_count,
                    SUM(amount_cents) / 100.0 AS total_spent
                FROM orders
                WHERE status IN ('paid', 'shipped')
                GROUP BY customer
                ORDER BY total_spent DESC, customer ASC
                LIMIT 10
                """
            )
    finally:
        connection.close()

    return stats


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Limpia el CSV de pedidos y reemplaza la tabla orders en SQLite."
    )
    parser.add_argument("--csv", type=Path, default=DEFAULT_CSV_PATH, help="Ruta del CSV.")
    parser.add_argument("--db", type=Path, default=DEFAULT_DB_PATH, help="Ruta de la base SQLite.")
    args = parser.parse_args()

    stats = load_orders(args.csv, args.db)
    print(
        f"Leídas: {stats['read']}; duplicadas: {stats['duplicates']}; "
        f"fechas inválidas: {stats['invalid_dates']}; importes vacíos: "
        f"{stats['missing_amounts']}; importes inválidos: {stats['invalid_amounts']}; "
        f"cargadas: {stats['loaded']}."
    )
    print(f"Base de datos SQLite: {args.db}")


if __name__ == "__main__":
    main()
