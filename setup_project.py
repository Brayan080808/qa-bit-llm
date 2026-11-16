from pathlib import Path
from typing import Callable

import clean_orders
import generate_orders


ROOT = Path(__file__).resolve().parent
DEFAULT_CSV_PATH = ROOT / "data" / "orders.csv"
DEFAULT_DB_PATH = ROOT / "data" / "orders.sqlite"
CSV_COLUMNS = "order_date,customer,country,product,amount,status"


def initialize_project(
    input_fn: Callable[[str], str] | None = None,
    output_fn: Callable[[str], None] = print,
    csv_path: Path = DEFAULT_CSV_PATH,
    db_path: Path = DEFAULT_DB_PATH,
) -> dict[str, int]:
    read_input = input if input_fn is None else input_fn
    output_fn("Orders CSV required structure (columns in this exact order):")
    output_fn(CSV_COLUMNS)
    output_fn("Example: 2025-03-01,Customer 001,Chile,Laptop,12.34,paid")
    output_fn("1. Generate a sample CSV automatically")
    output_fn("2. Use an existing CSV file")

    choice = read_input("Choose 1 or 2: ").strip()
    if choice == "1":
        source_csv = csv_path
        generated_rows = generate_orders.generate_orders(source_csv)
        output_fn(f"Generated CSV: {source_csv} ({generated_rows} rows).")
    elif choice == "2":
        supplied_path = read_input("Enter the CSV path: ").strip()
        if not supplied_path:
            raise ValueError("A CSV file path is required.")
        source_csv = Path(supplied_path).expanduser()
        if not source_csv.is_absolute():
            source_csv = Path.cwd() / source_csv
    else:
        raise ValueError("Choose 1 to generate a CSV or 2 to provide one.")

    if source_csv.suffix.lower() != ".csv":
        raise ValueError("The input file must use the .csv extension.")
    stats = clean_orders.load_orders(source_csv, db_path)
    output_fn(
        "CSV loaded and cleaned: "
        f"read={stats['read']}, duplicates={stats['duplicates']}, "
        f"invalid_dates={stats['invalid_dates']}, "
        f"missing_amounts={stats['missing_amounts']}, "
        f"invalid_amounts={stats['invalid_amounts']}, loaded={stats['loaded']}."
    )
    output_fn(f"SQLite database: {db_path}")
    output_fn("Next steps:")
    output_fn("1. Set GEMINI_API_KEY in .env (copy .env.example if needed).")
    output_fn("2. Start the API: uvicorn api:app --reload")
    output_fn("3. Open the API docs: http://127.0.0.1:8000/docs")
    return stats


def main() -> None:
    try:
        initialize_project()
    except (FileNotFoundError, ValueError) as exc:
        raise SystemExit(f"Setup failed: {exc}") from exc


if __name__ == "__main__":
    main()
