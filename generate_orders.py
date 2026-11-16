import argparse
import csv
import random
from datetime import date, timedelta
from pathlib import Path


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = ROOT / "data" / "orders.csv"
FIELDNAMES = ["order_date", "customer", "country", "product", "amount", "status"]
COUNTRIES = ["España", "México", "Colombia", "Argentina", "Chile", "Perú"]
PRODUCTS = [
    "Laptop",
    "Teléfono",
    "Auriculares",
    "Monitor",
    "Teclado",
    "Ratón",
]
STATUSES = ["pending", "paid", "shipped", "cancelled"]
DUPLICATE_COUNT = 12
INVALID_DATE_COUNT = 10
MISSING_AMOUNT_COUNT = 8


def _format_amount(cents: int) -> str:
    return f"{cents // 100}.{cents % 100:02d}"


def generate_orders(
    output_path: Path = DEFAULT_CSV_PATH,
    seed: int = 2026,
    valid_orders: int = 1000,
) -> int:
    if valid_orders < 1:
        raise ValueError("valid_orders debe ser mayor que cero.")

    rng = random.Random(seed)
    first_day = date(2025, 1, 1)
    days_in_year = 365
    valid_rows: list[dict[str, str]] = []
    seen_rows: set[tuple[str, ...]] = set()

    while len(valid_rows) < valid_orders:
        order_day = first_day + timedelta(days=rng.randrange(days_in_year))
        cents = rng.randint(500, 300_000)
        row = {
            "order_date": order_day.isoformat(),
            "customer": f"Cliente {rng.randint(1, 80):03d}",
            "country": rng.choice(COUNTRIES),
            "product": rng.choice(PRODUCTS),
            "amount": _format_amount(cents),
            "status": rng.choice(STATUSES),
        }
        signature = tuple(row[field] for field in FIELDNAMES)
        if signature not in seen_rows:
            seen_rows.add(signature)
            valid_rows.append(row)

    rows = list(valid_rows)
    rows.extend(
        dict(row)
        for row in rng.sample(valid_rows, min(DUPLICATE_COUNT, valid_orders))
    )

    invalid_date_rows = rng.sample(valid_rows, min(INVALID_DATE_COUNT, valid_orders))
    for index, row in enumerate(invalid_date_rows):
        invalid_row = dict(row)
        invalid_row["order_date"] = f"2025-13-{index + 1:02d}"
        rows.append(invalid_row)

    for row in rng.sample(valid_rows, min(MISSING_AMOUNT_COUNT, valid_orders)):
        missing_amount_row = dict(row)
        missing_amount_row["amount"] = ""
        rows.append(missing_amount_row)

    rng.shuffle(rows)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=FIELDNAMES)
        writer.writeheader()
        writer.writerows(rows)

    return len(rows)


def main() -> None:
    parser = argparse.ArgumentParser(description="Genera un CSV reproducible de pedidos.")
    parser.add_argument(
        "--output",
        type=Path,
        default=DEFAULT_CSV_PATH,
        help=f"Ruta del CSV de salida (por defecto: {DEFAULT_CSV_PATH})",
    )
    parser.add_argument("--seed", type=int, default=2026, help="Semilla aleatoria.")
    parser.add_argument(
        "--valid-orders",
        type=int,
        default=1000,
        help="Cantidad de pedidos válidos antes de añadir errores.",
    )
    args = parser.parse_args()
    row_count = generate_orders(args.output, args.seed, args.valid_orders)
    print(f"CSV generado: {args.output} ({row_count} filas de pedidos)")


if __name__ == "__main__":
    main()
