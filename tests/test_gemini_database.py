import csv
import json
import os
import sqlite3
from types import SimpleNamespace

import anyio
import httpx
import pytest

import api
import clean_orders


ORDERS = [
    {
        "order_date": "2025-03-05",
        "customer": "Buyer A",
        "country": "España",
        "product": "Laptop",
        "amount": "10.00",
        "status": "paid",
    },
    {
        "order_date": "2025-03-20",
        "customer": "Buyer A",
        "country": "España",
        "product": "Laptop",
        "amount": "5.00",
        "status": "shipped",
    },
    {
        "order_date": "2025-03-22",
        "customer": "Buyer C",
        "country": "España",
        "product": "Laptop",
        "amount": "99.00",
        "status": "pending",
    },
    {
        "order_date": "2025-03-25",
        "customer": "Buyer C",
        "country": "España",
        "product": "Laptop",
        "amount": "88.00",
        "status": "cancelled",
    },
    {
        "order_date": "2025-03-13",
        "customer": "Buyer B",
        "country": "México",
        "product": "Laptop",
        "amount": "20.25",
        "status": "paid",
    },
    {
        "order_date": "2025-04-04",
        "customer": "Buyer B",
        "country": "México",
        "product": "Laptop",
        "amount": "4.75",
        "status": "shipped",
    },
    {
        "order_date": "2025-04-02",
        "customer": "Buyer A",
        "country": "España",
        "product": "Laptop",
        "amount": "2.50",
        "status": "paid",
    },
    {
        "order_date": "2025-03-02",
        "customer": "Buyer C",
        "country": "Colombia",
        "product": "Laptop",
        "amount": "30.00",
        "status": "paid",
    },
    {
        "order_date": "2025-03-04",
        "customer": "Buyer D",
        "country": "Colombia",
        "product": "Laptop",
        "amount": "1.00",
        "status": "shipped",
    },
    {
        "order_date": "2025-02-14",
        "customer": "Buyer E",
        "country": "Chile",
        "product": "Laptop",
        "amount": "40.00",
        "status": "paid",
    },
    {
        "order_date": "2025-04-01",
        "customer": "Buyer E",
        "country": "Chile",
        "product": "Laptop",
        "amount": "500.00",
        "status": "pending",
    },
]

DATABASE_CASES = [
    (
        "¿Cuánto vendimos en España en marzo de 2025?",
        "SELECT total_sales FROM monthly_sales_by_country "
        "WHERE month = '2025-03' AND country = 'España'",
        [{"total_sales": 15.0}],
    ),
    (
        "¿Cuántos pedidos de venta hubo en España durante marzo de 2025?",
        "SELECT order_count FROM monthly_sales_by_country "
        "WHERE month = '2025-03' AND country = 'España'",
        [{"order_count": 2}],
    ),
    (
        "¿Cuál fue la venta total de México en marzo de 2025?",
        "SELECT total_sales FROM monthly_sales_by_country "
        "WHERE month = '2025-03' AND country = 'México'",
        [{"total_sales": 20.25}],
    ),
    (
        "¿Cuántas ventas tuvimos en México en abril de 2025?",
        "SELECT order_count FROM monthly_sales_by_country "
        "WHERE month = '2025-04' AND country = 'México'",
        [{"order_count": 1}],
    ),
    (
        "¿Cuánto se vendió en España en abril de 2025?",
        "SELECT total_sales FROM monthly_sales_by_country "
        "WHERE month = '2025-04' AND country = 'España'",
        [{"total_sales": 2.5}],
    ),
    (
        "Compara las ventas de todos los países en marzo de 2025.",
        "SELECT country, total_sales FROM monthly_sales_by_country "
        "WHERE month = '2025-03' ORDER BY country",
        [
            {"country": "Colombia", "total_sales": 31.0},
            {"country": "España", "total_sales": 15.0},
            {"country": "México", "total_sales": 20.25},
        ],
    ),
    (
        "¿Qué cliente gastó más y cuánto?",
        "SELECT customer, total_spent FROM top_customers "
        "ORDER BY total_spent DESC LIMIT 1",
        [{"customer": "Buyer E", "total_spent": 40.0}],
    ),
    (
        "Dame los tres clientes que más gastaron con su importe.",
        "SELECT customer, total_spent FROM top_customers "
        "ORDER BY total_spent DESC LIMIT 3",
        [
            {"customer": "Buyer E", "total_spent": 40.0},
            {"customer": "Buyer C", "total_spent": 30.0},
            {"customer": "Buyer B", "total_spent": 25.0},
        ],
    ),
    (
        "¿Cuánto vendimos en Narnia en marzo de 2025?",
        "SELECT total_sales FROM monthly_sales_by_country "
        "WHERE month = '2025-03' AND country = 'Narnia'",
        [],
    ),
    (
        "¿Cuánto vendimos en España en agosto de 2025?",
        "SELECT total_sales FROM monthly_sales_by_country "
        "WHERE month = '2025-08' AND country = 'España'",
        [],
    ),
]

INVALID_GEMINI_SQL = [
    "DELETE FROM orders",
    "SELECT * FROM monthly_sales_by_country; DROP TABLE orders",
    "SELECT 1",
    "SELECT load_extension('extension.dll') FROM top_customers",
    'SELECT * FROM "Usuarios administradores"',
]

SQLITE_PROMPT_EXAMPLES = [
    (
        "SELECT * FROM orders WHERE country = 'Chile' "
        "ORDER BY amount_cents DESC LIMIT 20",
        [
            {
                "order_id": 11,
                "order_date": "2025-04-01",
                "customer": "Buyer E",
                "country": "Chile",
                "product": "Laptop",
                "amount_cents": 50000,
                "status": "pending",
            },
            {
                "order_id": 10,
                "order_date": "2025-02-14",
                "customer": "Buyer E",
                "country": "Chile",
                "product": "Laptop",
                "amount_cents": 4000,
                "status": "paid",
            },
        ],
    ),
    (
        "SELECT customer AS name, amount_cents / 100.0 AS price "
        "FROM orders WHERE country = 'Chile' "
        "ORDER BY amount_cents DESC LIMIT 20",
        [
            {"name": "Buyer E", "price": 500.0},
            {"name": "Buyer E", "price": 40.0},
        ],
    ),
    (
        "SELECT COALESCE(SUM(total_sales), 0) AS total_sales "
        "FROM monthly_sales_by_country "
        "WHERE month = '2025-03' AND country = 'España'",
        [{"total_sales": 15.0}],
    ),
    (
        "SELECT COALESCE(SUM(order_count), 0) AS order_count "
        "FROM monthly_sales_by_country WHERE month = '2026-11'",
        [{"order_count": 0}],
    ),
    (
        "SELECT customer, total_spent FROM top_customers "
        "ORDER BY total_spent DESC LIMIT 3",
        [
            {"customer": "Buyer E", "total_spent": 40.0},
            {"customer": "Buyer C", "total_spent": 30.0},
            {"customer": "Buyer B", "total_spent": 25.0},
        ],
    ),
]

LIVE_GEMINI_ENABLED = (
    os.getenv("RUN_GEMINI_INTEGRATION") == "1"
    and bool(os.getenv("GEMINI_API_KEY"))
)


def _create_database(tmp_path):
    csv_path = tmp_path / "orders.csv"
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=clean_orders.FIELDNAMES)
        writer.writeheader()
        writer.writerows(ORDERS)

    db_path = tmp_path / "orders.sqlite"
    clean_orders.load_orders(csv_path, db_path)
    return db_path


def _install_gemini_stub(monkeypatch, question_to_sql):
    monkeypatch.setenv("GEMINI_API_KEY", "pytest-not-a-real-key")
    requests = []

    class FakeGeminiClient:
        def __init__(self, api_key):
            assert api_key == "pytest-not-a-real-key"
            self.models = SimpleNamespace(generate_content=self.generate_content)

        def __enter__(self):
            return self

        def __exit__(self, exc_type, exc_value, traceback):
            return False

        def generate_content(self, **kwargs):
            requests.append(kwargs)
            payload = json.loads(kwargs["contents"])
            sql = question_to_sql[payload["question"]]
            return SimpleNamespace(
                text=json.dumps(
                    {"responseType": "sql", "modelResponse": sql}
                )
            )

    monkeypatch.setattr(api.genai, "Client", FakeGeminiClient)
    return requests


def _post_question(app, question):
    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.post("/ask", json={"question": question})

    return anyio.run(send_request)


@pytest.mark.parametrize(
    ("question", "sql", "expected_rows"),
    DATABASE_CASES,
    ids=[
        "sales-spain-march",
        "order-count-spain-march",
        "sales-mexico-march",
        "order-count-mexico-april",
        "sales-spain-april",
        "sales-by-country",
        "highest-spending-customer",
        "top-three-customers",
        "unknown-country-no-results",
        "empty-month-no-results",
    ],
)
def test_gemini_question_to_sql_to_database_returns_expected_rows(
    tmp_path, monkeypatch, question, sql, expected_rows
):
    db_path = _create_database(tmp_path)
    gemini_requests = _install_gemini_stub(monkeypatch, {question: sql})
    app = api.create_app(
        db_path,
        answer_generator=lambda question, columns, rows: "Respuesta simulada.",
    )

    response = _post_question(app, question)

    assert response.status_code == 200
    assert response.json()["answer"] == "Respuesta simulada."
    assert response.json()["rows"] == expected_rows
    assert response.json()["truncated"] is False
    assert json.loads(gemini_requests[0]["contents"])["question"] == question
    assert gemini_requests[0]["config"].response_mime_type == "application/json"
    assert gemini_requests[0]["config"].response_schema is api.RespuestaJSON


@pytest.mark.parametrize(
    ("sql", "expected_rows"),
    SQLITE_PROMPT_EXAMPLES,
    ids=[
        "chile-orders",
        "chile-customer-prices",
        "monthly-sales",
        "empty-current-month",
        "top-customers",
    ],
)
def test_sqlite_prompt_examples_are_valid_and_execute(
    tmp_path, sql, expected_rows
):
    db_path = _create_database(tmp_path)

    safe_sql = api.validate_read_query(sql)
    _, rows, truncated = api.execute_read_query(db_path, safe_sql)

    assert rows == expected_rows
    assert truncated is False


def test_read_only_endpoint_allows_sqlite_like_year_filter(tmp_path):
    db_path = _create_database(tmp_path)
    sql = (
        "SELECT country FROM monthly_sales_by_country "
        "WHERE month LIKE '2025-%' "
        "GROUP BY country ORDER BY SUM(total_sales) DESC LIMIT 1"
    )
    app = api.create_app(
        db_path,
        lambda _: sql,
        lambda question, columns, rows: "Chile tuvo las mayores ventas.",
    )

    response = _post_question(app, "¿Qué país vendió más en 2025?")

    assert response.status_code == 200
    assert response.json()["rows"] == [{"country": "Chile"}]
    assert response.json()["answer"] == "Chile tuvo las mayores ventas."


@pytest.mark.parametrize("generated_sql", INVALID_GEMINI_SQL)
def test_gemini_invalid_or_unsafe_sql_gets_explained_without_execution(
    tmp_path, monkeypatch, generated_sql
):
    db_path = _create_database(tmp_path)
    question = "prueba de consulta"
    _install_gemini_stub(monkeypatch, {question: generated_sql})
    app = api.create_app(
        db_path,
        answer_generator=lambda question, columns, rows: "Respuesta simulada.",
        unavailable_answer_generator=lambda question, limitation, history: (
            "Solo puedo consultar los análisis de pedidos permitidos."
        ),
    )

    response = _post_question(app, question)

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "Solo puedo consultar los análisis de pedidos permitidos."
    )
    assert response.json()["sql"] is None
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == len(
            ORDERS
        )


@pytest.mark.parametrize(
    ("question", "expected_value", "expected_answer_value"),
    [
        (
            "¿Cuánto vendimos en España en marzo de 2025?",
            15.0,
            "15",
        ),
        (
            "¿Cuántos pedidos de venta hubo en España en marzo de 2025?",
            2,
            "2",
        ),
    ],
    ids=["monthly-sales-value", "monthly-order-count"],
)
@pytest.mark.skipif(
    not LIVE_GEMINI_ENABLED,
    reason="Activa RUN_GEMINI_INTEGRATION=1 para llamar al servicio Gemini.",
)
def test_live_gemini_returns_correct_answer_from_sqlite(
    tmp_path, question, expected_value, expected_answer_value
):
    db_path = _create_database(tmp_path)
    app = api.create_app(db_path)

    response = _post_question(app, question)

    assert response.status_code == 200, response.json()
    answer = response.json()["answer"].lower()
    assert answer.strip()
    assert expected_answer_value in answer
    numeric_values = [
        value
        for row in response.json()["rows"]
        for value in row.values()
        if isinstance(value, (int, float))
    ]
    assert any(value == pytest.approx(expected_value) for value in numeric_values)
    assert response.json()["truncated"] is False


@pytest.mark.skipif(
    not LIVE_GEMINI_ENABLED,
    reason="Activa RUN_GEMINI_INTEGRATION=1 para llamar al servicio Gemini.",
)
def test_live_gemini_finds_the_top_spending_customers(tmp_path):
    db_path = _create_database(tmp_path)
    app = api.create_app(db_path)

    response = _post_question(
        app,
        "Dame los tres clientes con mayor gasto y sus importes, ordenados de mayor a menor.",
    )

    assert response.status_code == 200, response.json()
    answer = response.json()["answer"].lower()
    assert answer.strip()
    assert all(
        customer.lower() in answer
        for customer in ("Buyer E", "Buyer C", "Buyer B")
    )
    assert response.json()["rows"] == [
        {"customer": "Buyer E", "total_spent": 40.0},
        {"customer": "Buyer C", "total_spent": 30.0},
        {"customer": "Buyer B", "total_spent": 25.0},
    ]
