import csv
import sqlite3
from pathlib import Path

import pytest

import clean_orders
import generate_orders
import setup_project


def _write_orders_csv(path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=clean_orders.FIELDNAMES)
        writer.writeheader()
        writer.writerow(
            {
                "order_date": "2025-03-01",
                "customer": "Customer 001",
                "country": "Chile",
                "product": "Laptop",
                "amount": "12.34",
                "status": "paid",
            }
        )


def test_setup_can_generate_csv_and_initialize_database(tmp_path, monkeypatch):
    csv_path = tmp_path / "generated.csv"
    db_path = tmp_path / "orders.sqlite"
    real_generate = generate_orders.generate_orders
    monkeypatch.setattr(
        generate_orders,
        "generate_orders",
        lambda output_path: real_generate(output_path, valid_orders=2),
    )
    output = []

    stats = setup_project.initialize_project(
        input_fn=lambda _: "1",
        output_fn=output.append,
        csv_path=csv_path,
        db_path=db_path,
    )

    assert csv_path.is_file()
    assert stats["loaded"] == 2
    assert db_path.is_file()
    assert any("order_date,customer,country,product,amount,status" in line for line in output)
    assert any("uvicorn api:app --reload" in line for line in output)


def test_setup_can_load_user_provided_csv(tmp_path):
    csv_path = tmp_path / "user-orders.csv"
    db_path = tmp_path / "orders.sqlite"
    _write_orders_csv(csv_path)
    answers = iter(["2", str(csv_path)])

    stats = setup_project.initialize_project(
        input_fn=lambda _: next(answers),
        output_fn=lambda _: None,
        db_path=db_path,
    )

    assert stats["loaded"] == 1
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT customer, amount_cents FROM orders"
        ).fetchall() == [("Customer 001", 1234)]


def test_setup_rejects_missing_user_csv_path(tmp_path):
    answers = iter(["2", str(tmp_path / "missing.csv")])

    with pytest.raises(FileNotFoundError):
        setup_project.initialize_project(
            input_fn=lambda _: next(answers),
            output_fn=lambda _: None,
            db_path=tmp_path / "orders.sqlite",
        )


def test_setup_rejects_unknown_mode(tmp_path):
    with pytest.raises(ValueError, match="Choose 1 to generate"):
        setup_project.initialize_project(
            input_fn=lambda _: "3",
            output_fn=lambda _: None,
            csv_path=tmp_path / "orders.csv",
            db_path=tmp_path / "orders.sqlite",
        )
