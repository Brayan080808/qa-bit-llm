import csv
import json
import logging
import sqlite3
from pathlib import Path
from types import SimpleNamespace

import anyio
import httpx
import pytest

import api
import clean_orders


def _order(
    order_date="2025-03-01",
    customer="Cliente 001",
    country="España",
    amount="12.34",
    status="paid",
):
    return {
        "order_date": order_date,
        "customer": customer,
        "country": country,
        "product": "Laptop",
        "amount": amount,
        "status": status,
    }


def _client_for_orders(
    tmp_path, orders, query_generator, answer_generator=None
):
    csv_path = tmp_path / "input" / "orders.csv"
    csv_path.parent.mkdir(parents=True, exist_ok=True)
    with csv_path.open("w", newline="", encoding="utf-8") as csv_file:
        writer = csv.DictWriter(csv_file, fieldnames=clean_orders.FIELDNAMES)
        writer.writeheader()
        writer.writerows(orders)

    db_path = tmp_path / "db" / "orders.sqlite"
    clean_orders.load_orders(csv_path, db_path)
    if answer_generator is None:
        answer_generator = lambda question, columns, rows: "Respuesta de prueba."
    return api.create_app(db_path, query_generator, answer_generator), db_path


def _post(app, question):
    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.post("/ask", json={"question": question})

    return anyio.run(send_request)


class _FakeGeminiClient:
    def __init__(self, api_key, generate_content):
        self.api_key = api_key
        self.models = SimpleNamespace(generate_content=generate_content)
        self.closed = False

    def __enter__(self):
        return self

    def __exit__(self, exc_type, exc_value, traceback):
        self.closed = True


def _patch_gemini_client(monkeypatch, generate_content):
    clients = []

    def client_factory(api_key):
        client = _FakeGeminiClient(api_key, generate_content)
        clients.append(client)
        return client

    monkeypatch.setattr(api.genai, "Client", client_factory)
    return clients


@pytest.mark.parametrize(
    "sql",
    [
        "SELECT month, total_sales FROM monthly_sales_by_country",
        " SELECT customer FROM top_customers ORDER BY total_spent DESC; ",
    ],
)
def test_validate_read_query_accepts_selects_from_allowed_views(sql):
    normalized = api.validate_read_query(sql)

    assert "SELECT" in normalized.upper()
    assert any(view in normalized for view in api.ALLOWED_VIEWS)


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "INSERT INTO orders VALUES (1)",
        "DELETE FROM orders",
        "DROP TABLE orders",
        "SELECT * FROM orders",
        "SELECT * FROM sqlite_master",
        "SELECT * FROM main.monthly_sales_by_country",
        "SELECT 1",
        "SELECT * FROM monthly_sales_by_country; DELETE FROM orders",
        "WITH recent AS (SELECT * FROM monthly_sales_by_country) SELECT * FROM recent",
        "SELECT FROM monthly_sales_by_country",
        "SELECT * FROM",
    ],
)
def test_validate_read_query_rejects_unsafe_or_invalid_sql(sql):
    with pytest.raises(api.QueryValidationError):
        api.validate_read_query(sql)


def test_validate_read_query_rejects_non_string_input():
    with pytest.raises(api.QueryValidationError, match="vacía"):
        api.validate_read_query(None)


def test_cleaning_load_creates_monthly_sales_view_with_paid_orders_only(tmp_path):
    orders = [
        _order(amount="10.00"),
        _order(order_date="2025-03-26", amount="4.50", status="shipped"),
        _order(customer="Cliente pending", amount="999.00", status="pending"),
        _order(order_date="2025-04-01", amount="888.00", status="cancelled"),
        _order(customer="Cliente MX", country="México", amount="1.25"),
    ]
    client, db_path = _client_for_orders(
        tmp_path, orders, lambda _: "SELECT month, country, total_sales, order_count "
        "FROM monthly_sales_by_country ORDER BY country"
    )

    response = _post(client, "ventas por país en marzo")

    assert response.status_code == 200
    result_by_country = {row["country"]: row for row in response.json()["rows"]}
    assert result_by_country["España"] == {
        "month": "2025-03",
        "country": "España",
        "total_sales": 14.5,
        "order_count": 2,
    }
    assert result_by_country["México"]["total_sales"] == 1.25
    assert set(result_by_country) == {"España", "México"}
    assert db_path.is_file()


def test_cleaning_load_creates_top_ten_customers_view(tmp_path):
    orders = [
        _order(customer=f"Cliente {index:02d}", amount=f"{index}.00")
        for index in range(1, 13)
    ]
    client, _ = _client_for_orders(
        tmp_path,
        orders,
        lambda _: "SELECT customer, order_count, total_spent FROM top_customers",
    )

    response = _post(client, "clientes con mayor gasto")

    assert response.status_code == 200
    customers = response.json()["rows"]
    assert len(customers) == 10
    assert customers[0]["customer"] == "Cliente 12"
    assert customers[-1]["customer"] == "Cliente 03"


def test_endpoint_calls_injected_llm_and_returns_query_results(tmp_path):
    received_questions = []
    received_answer_context = []

    def fake_llm(question):
        received_questions.append(question)
        return "SELECT country, total_sales FROM monthly_sales_by_country"

    def fake_answer(question, columns, rows):
        received_answer_context.append((question, columns, rows))
        return "En España vendimos 12.34."

    client, _ = _client_for_orders(
        tmp_path, [_order()], fake_llm, answer_generator=fake_answer
    )

    response = _post(client, "  ¿Cuánto vendimos?  ")

    assert response.status_code == 200
    assert received_questions == ["¿Cuánto vendimos?"]
    assert response.json()["question"] == "¿Cuánto vendimos?"
    assert response.json()["answer"] == "En España vendimos 12.34."
    assert received_answer_context == [
        (
            "¿Cuánto vendimos?",
            ["country", "total_sales"],
            [{"country": "España", "total_sales": 12.34}],
        )
    ]
    assert response.json()["rows"] == [{"country": "España", "total_sales": 12.34}]
    assert response.json()["truncated"] is False


def test_endpoint_returns_clear_error_when_answer_generation_fails(tmp_path):
    def failing_answer(question, columns, rows):
        raise api.LLMQueryError("Gemini no pudo generar la respuesta textual.")

    client, _ = _client_for_orders(
        tmp_path,
        [_order()],
        lambda _: "SELECT country FROM monthly_sales_by_country",
        answer_generator=failing_answer,
    )

    response = _post(client, "ventas")

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "Gemini no pudo generar la respuesta textual."
    )


def test_endpoint_returns_clear_error_when_llm_fails(tmp_path):
    def failing_llm(_):
        raise api.LLMQueryError("Fallo simulado del proveedor.")

    client, _ = _client_for_orders(tmp_path, [_order()], failing_llm)

    response = _post(client, "ventas")

    assert response.status_code == 502
    assert response.json()["detail"] == "Fallo simulado del proveedor."


def test_endpoint_rejects_llm_generated_write_query_without_changing_database(tmp_path):
    client, db_path = _client_for_orders(
        tmp_path, [_order()], lambda _: "DELETE FROM orders"
    )

    response = _post(client, "borra los pedidos")

    assert response.status_code == 502
    assert "no permitida" in response.json()["detail"]
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1


def test_endpoint_returns_422_for_blank_question_without_calling_llm(tmp_path):
    calls = []
    client, _ = _client_for_orders(
        tmp_path, [_order()], lambda question: calls.append(question) or "SELECT 1"
    )

    response = _post(client, "   ")

    assert response.status_code == 422
    assert calls == []


def test_endpoint_returns_service_unavailable_when_database_is_missing(tmp_path):
    app = api.create_app(
        tmp_path / "missing.sqlite",
        lambda _: "SELECT * FROM monthly_sales_by_country",
    )

    response = _post(app, "ventas")

    assert response.status_code == 503
    assert "clean_orders.py" in response.json()["detail"]


def test_endpoint_returns_setup_error_when_analytic_views_are_missing(tmp_path):
    db_path = tmp_path / "orders.sqlite"
    with sqlite3.connect(db_path) as connection:
        connection.execute("CREATE TABLE orders (order_id INTEGER PRIMARY KEY)")
    app = api.create_app(
        db_path,
        lambda _: "SELECT * FROM monthly_sales_by_country",
    )

    response = _post(app, "ventas por país")

    assert response.status_code == 503
    assert "vistas analíticas" in response.json()["detail"]
    assert "clean_orders.py" in response.json()["detail"]


def test_endpoint_returns_bad_gateway_for_sqlite_operational_errors(tmp_path):
    app, _ = _client_for_orders(
        tmp_path,
        [_order()],
        lambda _: "SELECT unknown_column FROM monthly_sales_by_country",
    )

    response = _post(app, "ventas")

    assert response.status_code == 502
    assert response.json()["detail"] == (
        "La consulta generada no se pudo ejecutar en SQLite."
    )


def test_endpoint_logs_validated_sql_and_database_path(tmp_path, caplog):
    sql = "SELECT country FROM monthly_sales_by_country"
    app, db_path = _client_for_orders(
        tmp_path, [_order()], lambda _: sql
    )

    with caplog.at_level(logging.INFO, logger="uvicorn.error"):
        response = _post(app, "ventas por país")

    assert response.status_code == 200
    assert any(
        sql in record.getMessage() and str(db_path.resolve()) in record.getMessage()
        for record in caplog.records
    )


def test_execute_read_query_denies_direct_access_to_orders_table(tmp_path):
    _, db_path = _client_for_orders(tmp_path, [_order()], lambda _: "")

    with pytest.raises(sqlite3.DatabaseError):
        api.execute_read_query(db_path, "SELECT * FROM orders")


def test_read_only_authorizer_denies_unapproved_sql_functions():
    decision = api._read_only_authorizer(
        sqlite3.SQLITE_FUNCTION,
        None,
        "load_extension",
        "main",
        None,
    )

    assert decision == sqlite3.SQLITE_DENY


def test_read_only_authorizer_denies_unrecognized_sqlite_actions():
    decision = api._read_only_authorizer(
        sqlite3.SQLITE_PRAGMA,
        "user_version",
        None,
        "main",
        None,
    )

    assert decision == sqlite3.SQLITE_DENY


def test_endpoint_reports_when_result_rows_are_truncated(tmp_path):
    orders = [
        _order(customer=f"Cliente {index}", country=f"País {index}")
        for index in range(api.MAX_RESULT_ROWS + 1)
    ]
    client, _ = _client_for_orders(
        tmp_path,
        orders,
        lambda _: "SELECT country FROM monthly_sales_by_country ORDER BY country",
    )

    response = _post(client, "ventas por país")

    assert response.status_code == 200
    assert len(response.json()["rows"]) == api.MAX_RESULT_ROWS
    assert response.json()["truncated"] is True


def test_endpoint_denies_unapproved_sql_functions(tmp_path):
    client, _ = _client_for_orders(
        tmp_path,
        [_order()],
        lambda _: "SELECT load_extension('extension.dll') "
        "FROM monthly_sales_by_country",
    )

    response = _post(client, "ejecuta una extension")

    assert response.status_code == 502
    assert "no se pudo ejecutar" in response.json()["detail"]


def test_gemini_generator_returns_sql_from_json_response(monkeypatch):
    request_options = {}

    def generate_content(**kwargs):
        request_options.update(kwargs)
        return SimpleNamespace(text=json.dumps({"sql": "SELECT * FROM top_customers"}))

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    clients = _patch_gemini_client(monkeypatch, generate_content)

    sql = api.generate_sql_with_gemini("clientes principales")

    assert sql == "SELECT * FROM top_customers"
    assert clients[0].api_key == "test-key"
    assert clients[0].closed is True
    assert request_options["model"] == "test-model"
    assert request_options["contents"] == "clientes principales"
    assert request_options["config"].response_mime_type == "application/json"


def test_gemini_generator_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(api.LLMQueryError, match="GEMINI_API_KEY"):
        api.generate_sql_with_gemini("ventas")


def test_gemini_generator_reports_empty_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch, lambda **_: SimpleNamespace(text=None)
    )

    with pytest.raises(api.LLMQueryError, match="vacía"):
        api.generate_sql_with_gemini("ventas")


def test_gemini_generator_reports_provider_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch,
        lambda **_: (_ for _ in ()).throw(
            api.errors.APIError(503, {"message": "simulated provider error"})
        ),
    )

    with pytest.raises(api.LLMQueryError, match="Gemini"):
        api.generate_sql_with_gemini("ventas")


def test_gemini_generator_reports_network_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch,
        lambda **_: (_ for _ in ()).throw(
            httpx.ConnectError("simulated network error")
        ),
    )

    with pytest.raises(api.LLMQueryError, match="Gemini"):
        api.generate_sql_with_gemini("ventas")


@pytest.mark.parametrize(
    ("content", "error"),
    [
        ("not json", "JSON válido"),
        ('{"query":"SELECT 1"}', "no contiene"),
        ("[]", "no contiene"),
    ],
)
def test_gemini_generator_reports_malformed_responses(monkeypatch, content, error):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch, lambda **_: SimpleNamespace(text=content)
    )

    with pytest.raises(api.LLMQueryError, match=error):
        api.generate_sql_with_gemini("ventas")


def test_gemini_answer_generator_uses_question_and_sqlite_rows(monkeypatch):
    request_options = {}

    def generate_content(**kwargs):
        request_options.update(kwargs)
        return SimpleNamespace(text="En España se vendieron 12.34.")

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    clients = _patch_gemini_client(monkeypatch, generate_content)

    answer = api.generate_answer_with_gemini(
        "¿Cuánto vendimos?",
        ["country", "total_sales"],
        [{"country": "España", "total_sales": 12.34}],
    )

    assert answer == "En España se vendieron 12.34."
    assert clients[0].api_key == "test-key"
    assert clients[0].closed is True
    assert request_options["model"] == "test-model"
    assert json.loads(request_options["contents"]) == {
        "question": "¿Cuánto vendimos?",
        "columns": ["country", "total_sales"],
        "rows": [{"country": "España", "total_sales": 12.34}],
    }


def test_gemini_answer_generator_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(api.LLMQueryError, match="GEMINI_API_KEY"):
        api.generate_answer_with_gemini("ventas", [], [])


def test_gemini_answer_generator_reports_empty_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch, lambda **_: SimpleNamespace(text="   ")
    )

    with pytest.raises(api.LLMQueryError, match="respuesta textual vacía"):
        api.generate_answer_with_gemini("ventas", [], [])


def test_gemini_answer_generator_reports_provider_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch,
        lambda **_: (_ for _ in ()).throw(
            api.errors.APIError(503, {"message": "simulated provider error"})
        ),
    )

    with pytest.raises(api.LLMQueryError, match="respuesta textual"):
        api.generate_answer_with_gemini("ventas", [], [])
