import csv
import json
import logging
import sqlite3
from io import StringIO
from datetime import datetime
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
    tmp_path,
    orders,
    query_generator,
    answer_generator=None,
    unavailable_answer_generator=None,
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
    return (
        api.create_app(
            db_path,
            query_generator,
            answer_generator,
            unavailable_answer_generator,
        ),
        db_path,
    )


def _post(app, question, conversation_id=None):
    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            payload = {"question": question}
            if conversation_id is not None:
                payload["conversation_id"] = conversation_id
            return await client.post("/ask", json=payload)

    return anyio.run(send_request)


def _delete_conversation(app, conversation_id):
    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.delete(f"/conversations/{conversation_id}")

    return anyio.run(send_request)


def _upload_csv(app, filename, content):
    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.post(
                "/orders/upload",
                files={"file": (filename, content, "text/csv")},
            )

    return anyio.run(send_request)


def _delete_orders(app, confirm):
    async def send_request():
        async with httpx.AsyncClient(
            transport=httpx.ASGITransport(app=app),
            base_url="http://test",
        ) as client:
            return await client.request(
                "DELETE", "/orders", json={"confirm": confirm}
            )

    return anyio.run(send_request)


def _csv_bytes(orders):
    buffer = StringIO(newline="")
    writer = csv.DictWriter(buffer, fieldnames=clean_orders.FIELDNAMES)
    writer.writeheader()
    writer.writerows(orders)
    return buffer.getvalue().encode("utf-8")


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
        "SELECT * FROM orders WHERE country = 'Chile' ORDER BY amount_cents DESC",
    ],
)
def test_validate_read_query_accepts_selects_from_allowed_sources(sql):
    normalized = api.validate_read_query(sql)

    assert "SELECT" in normalized.upper()
    assert any(source in normalized for source in api.ALLOWED_QUERY_SOURCES)


@pytest.mark.parametrize(
    "sql",
    [
        "",
        "INSERT INTO orders VALUES (1)",
        "DELETE FROM orders",
        "DROP TABLE orders",
        'SELECT * FROM "Usuarios administradores"',
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


def test_prompt_loader_reports_missing_file(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "PROMPTS_DIR", tmp_path)

    with pytest.raises(RuntimeError, match="No se pudo cargar el prompt"):
        api._read_prompt("missing.txt")


def test_prompt_loader_rejects_empty_file(monkeypatch, tmp_path):
    monkeypatch.setattr(api, "PROMPTS_DIR", tmp_path)
    (tmp_path / "empty.txt").write_text("", encoding="utf-8")

    with pytest.raises(RuntimeError, match="archivo de prompt está vacío"):
        api._read_prompt("empty.txt")


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


def test_endpoint_reuses_context_for_follow_up_questions(tmp_path, monkeypatch):
    query_histories = []
    answer_histories = []

    def fake_sql(question, history):
        query_histories.append(history)
        return "SELECT country, total_sales FROM monthly_sales_by_country"

    def fake_answer(question, columns, rows, history):
        answer_histories.append(history)
        return f"Respuesta {len(answer_histories)}."

    monkeypatch.setattr(api, "generate_sql_with_gemini", fake_sql)
    monkeypatch.setattr(api, "generate_answer_with_gemini", fake_answer)
    _, db_path = _client_for_orders(tmp_path, [_order()], None)
    app = api.create_app(db_path)

    first_response = _post(app, "Ventas en España")
    conversation_id = first_response.json()["conversation_id"]
    second_response = _post(app, "¿Y en marzo?", conversation_id)

    assert first_response.status_code == 200
    assert second_response.status_code == 200
    assert second_response.json()["conversation_id"] == conversation_id
    expected_history = [
        {
            "question": "Ventas en España",
            "sql": first_response.json()["sql"],
            "answer": first_response.json()["answer"],
        }
    ]
    assert query_histories == [[], expected_history]
    assert answer_histories == [[], expected_history]


def test_delete_conversation_clears_its_context(tmp_path, monkeypatch):
    received_histories = []

    def fake_sql(question, history):
        received_histories.append(history)
        return "SELECT country FROM monthly_sales_by_country"

    monkeypatch.setattr(api, "generate_sql_with_gemini", fake_sql)
    monkeypatch.setattr(
        api,
        "generate_answer_with_gemini",
        lambda question, columns, rows, history: "Respuesta de prueba.",
    )
    _, db_path = _client_for_orders(tmp_path, [_order()], None)
    app = api.create_app(db_path)

    first_response = _post(app, "Ventas en España")
    conversation_id = first_response.json()["conversation_id"]
    delete_response = _delete_conversation(app, conversation_id)
    next_response = _post(app, "¿Y en marzo?", conversation_id)

    assert delete_response.status_code == 200
    assert delete_response.json() == {
        "conversation_id": conversation_id,
        "cleared": True,
    }
    assert next_response.status_code == 200
    assert received_histories == [[], []]


def test_delete_conversation_returns_404_for_unknown_context(tmp_path):
    _, db_path = _client_for_orders(tmp_path, [_order()], None)
    app = api.create_app(db_path)

    response = _delete_conversation(
        app, "a5bcbb64-8d90-4ab6-8ca1-c7b48a3ef118"
    )

    assert response.status_code == 404


def test_conversation_cache_expires_after_idle_timeout(monkeypatch):
    now = [100.0]
    monkeypatch.setattr(api.time, "monotonic", lambda: now[0])
    cache = api.ConversationCache()
    cache.add_turn(
        "conversation",
        {"question": "ventas", "sql": "SELECT 1", "answer": "Respuesta."},
    )

    now[0] += api.CONVERSATION_TTL_SECONDS

    assert cache.get_history("conversation") == []


def test_conversation_cache_keeps_only_the_most_recent_turns():
    cache = api.ConversationCache()
    for index in range(api.MAX_CONVERSATION_TURNS + 2):
        cache.add_turn(
            "conversation",
            {"question": str(index), "sql": "SELECT 1", "answer": "Respuesta."},
        )

    history = cache.get_history("conversation")

    assert len(history) == api.MAX_CONVERSATION_TURNS
    assert history[0]["question"] == "2"
    assert history[-1]["question"] == str(api.MAX_CONVERSATION_TURNS + 1)


def test_conversation_cache_evicts_least_recently_used_session(monkeypatch):
    monkeypatch.setattr(api, "MAX_CONVERSATIONS", 2)
    cache = api.ConversationCache()
    turn = {"question": "ventas", "sql": "SELECT 1", "answer": "Respuesta."}
    cache.add_turn("first", turn)
    cache.add_turn("second", turn)
    cache.get_history("first")
    cache.add_turn("third", turn)

    assert cache.get_history("second") == []
    assert cache.get_history("first") == [turn]
    assert cache.get_history("third") == [turn]


def test_endpoint_executes_sqlite_coalesce_for_month_without_orders(tmp_path):
    sqlite_sql = (
        "SELECT COALESCE(SUM(order_count), 0) AS order_count "
        "FROM monthly_sales_by_country WHERE month = '2026-11'"
    )
    client, _ = _client_for_orders(
        tmp_path, [_order(order_date="2025-03-01")], lambda _: sqlite_sql
    )

    response = _post(client, "¿Cuántas ventas hubo este mes?")

    assert response.status_code == 200
    assert response.json()["rows"] == [{"order_count": 0}]


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


def test_endpoint_explains_rejected_write_query_without_changing_database(tmp_path):
    fallback_calls = []

    def fallback(question, limitation, history):
        fallback_calls.append((question, limitation, history))
        return "No puedo borrar pedidos. Puedo ayudarte a analizar las ventas."

    client, db_path = _client_for_orders(
        tmp_path,
        [_order()],
        lambda _: "DELETE FROM orders",
        unavailable_answer_generator=fallback,
    )

    response = _post(client, "borra los pedidos")

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "No puedo borrar pedidos. Puedo ayudarte a analizar las ventas."
    )
    assert response.json()["sql"] is None
    assert fallback_calls[0][0] == "borra los pedidos"
    assert "consulta segura" in fallback_calls[0][1]
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1


def test_endpoint_asks_gemini_to_explain_unsafe_query(tmp_path, monkeypatch):
    fallback_calls = []

    def fallback(question, limitation, history):
        fallback_calls.append((question, limitation, history))
        return "No puedo acceder a ese detalle; puedo resumir ventas por país."

    monkeypatch.setattr(api, "generate_unavailable_answer_with_gemini", fallback)
    client, _ = _client_for_orders(
        tmp_path,
        [_order()],
        lambda _: 'SELECT * FROM "Usuarios administradores"',
    )

    response = _post(client, "Dame cada pedido individual")

    assert response.status_code == 200
    assert response.json()["answer"].startswith("No puedo acceder")
    assert response.json()["sql"] is None
    assert fallback_calls[0][0] == "Dame cada pedido individual"
    assert "consulta segura" in fallback_calls[0][1]


def test_endpoint_reports_error_if_gemini_cannot_explain_unsafe_query(
    tmp_path, monkeypatch
):
    def failing_fallback(question, limitation, history):
        raise api.LLMQueryError("Gemini no pudo explicar la limitación.")

    monkeypatch.setattr(
        api, "generate_unavailable_answer_with_gemini", failing_fallback
    )
    client, _ = _client_for_orders(
        tmp_path,
        [_order()],
        lambda _: 'SELECT * FROM "Usuarios administradores"',
    )

    response = _post(client, "Dame cada pedido individual")

    assert response.status_code == 502
    assert response.json()["detail"] == "Gemini no pudo explicar la limitación."


def test_endpoint_returns_422_for_blank_question_without_calling_llm(tmp_path):
    calls = []
    client, _ = _client_for_orders(
        tmp_path, [_order()], lambda question: calls.append(question) or "SELECT 1"
    )

    response = _post(client, "   ")

    assert response.status_code == 422
    assert calls == []


def test_upload_endpoint_cleans_and_replaces_database_from_csv(tmp_path):
    _, db_path = _client_for_orders(tmp_path, [_order()], lambda _: "")
    app = api.create_app(db_path)
    valid_order = _order(customer="Cliente Chile", country="Chile", amount="20.50")
    invalid_date = _order(order_date="16/11/2026", country="Chile")
    missing_amount = _order(amount="", country="Chile")
    csv_content = _csv_bytes(
        [valid_order, valid_order.copy(), invalid_date, missing_amount]
    )

    response = _upload_csv(app, "orders.csv", csv_content)

    assert response.status_code == 200
    assert response.json()["stats"] == {
        "read": 4,
        "duplicates": 1,
        "invalid_dates": 1,
        "missing_amounts": 1,
        "invalid_amounts": 0,
        "loaded": 1,
    }
    assert response.json()["filename"] == "orders.csv"
    with sqlite3.connect(db_path) as connection:
        assert connection.execute(
            "SELECT customer, country, amount_cents FROM orders"
        ).fetchall() == [("Cliente Chile", "Chile", 2050)]


def test_upload_endpoint_rejects_bad_csv_without_changing_database(tmp_path):
    app, db_path = _client_for_orders(tmp_path, [_order()], lambda _: "")
    invalid_csv = b"date,customer\n2025-03-01,Cliente"

    response = _upload_csv(app, "bad.csv", invalid_csv)

    assert response.status_code == 422
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1


def test_upload_endpoint_rejects_non_csv_file(tmp_path):
    app, _ = _client_for_orders(tmp_path, [_order()], lambda _: "")

    response = _upload_csv(app, "orders.txt", b"not a CSV")

    assert response.status_code == 415


def test_upload_endpoint_rejects_files_over_size_limit(tmp_path):
    app, _ = _client_for_orders(tmp_path, [_order()], lambda _: "")

    response = _upload_csv(
        app,
        "large.csv",
        b"x" * (api.MAX_CSV_UPLOAD_BYTES + 1),
    )

    assert response.status_code == 413


def test_clear_orders_endpoint_requires_explicit_confirmation(tmp_path):
    app, db_path = _client_for_orders(tmp_path, [_order()], lambda _: "")

    response = _delete_orders(app, confirm=False)

    assert response.status_code == 400
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 1


def test_clear_orders_endpoint_deletes_rows_and_keeps_analytic_views(tmp_path):
    orders = [_order(), _order(customer="Cliente 002", amount="20.00")]
    app, db_path = _client_for_orders(tmp_path, orders, lambda _: "")

    response = _delete_orders(app, confirm=True)

    assert response.status_code == 200
    assert response.json() == {"deleted_count": 2, "cleared": True}
    with sqlite3.connect(db_path) as connection:
        assert connection.execute("SELECT COUNT(*) FROM orders").fetchone()[0] == 0
        assert connection.execute(
            "SELECT name FROM sqlite_master WHERE type = 'view' "
            "AND name IN ('monthly_sales_by_country', 'top_customers') "
            "ORDER BY name"
        ).fetchall() == [
            ("monthly_sales_by_country",),
            ("top_customers",),
        ]


def test_clear_orders_endpoint_reports_missing_database(tmp_path):
    app = api.create_app(tmp_path / "missing.sqlite")

    response = _delete_orders(app, confirm=True)

    assert response.status_code == 503


def test_clear_orders_endpoint_reports_uninitialized_database(tmp_path):
    db_path = tmp_path / "empty.sqlite"
    sqlite3.connect(db_path).close()
    app = api.create_app(db_path)

    response = _delete_orders(app, confirm=True)

    assert response.status_code == 503


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


def test_execute_read_query_allows_direct_orders_and_sorts_by_amount(tmp_path):
    orders = [
        _order(customer="Cliente bajo", country="Chile", amount="10.00"),
        _order(customer="Cliente alto", country="Chile", amount="25.50"),
        _order(customer="Cliente MX", country="México", amount="100.00"),
    ]
    _, db_path = _client_for_orders(tmp_path, orders, lambda _: "")
    sql = (
        "SELECT customer, amount_cents FROM orders "
        "WHERE country = 'Chile' ORDER BY amount_cents DESC"
    )

    safe_sql = api.validate_read_query(sql)
    columns, rows, truncated = api.execute_read_query(db_path, safe_sql)

    assert columns == ["customer", "amount_cents"]
    assert rows == [
        {"customer": "Cliente alto", "amount_cents": 2550},
        {"customer": "Cliente bajo", "amount_cents": 1000},
    ]
    assert truncated is False


def test_endpoint_returns_order_details_in_a_readable_table(tmp_path):
    orders = [
        _order(customer="Cliente bajo", country="Chile", amount="10.00"),
        _order(customer="Cliente alto", country="Chile", amount="25.50"),
    ]
    sql = (
        "SELECT customer AS name, amount_cents / 100.0 AS price "
        "FROM orders WHERE country = 'Chile' "
        "ORDER BY amount_cents DESC LIMIT 20"
    )
    markdown_answer = "| Nombre | Precio |\n|---|---:|\n| Cliente alto | 25.50 |"
    app, _ = _client_for_orders(
        tmp_path,
        orders,
        lambda _: sql,
        answer_generator=lambda question, columns, rows: markdown_answer,
    )

    response = _post(app, "Muestra nombres y precios de pedidos de Chile")

    assert response.status_code == 200
    assert response.json()["rows"] == [
        {"name": "Cliente alto", "price": 25.5},
        {"name": "Cliente bajo", "price": 10.0},
    ]
    assert response.json()["answer"] == markdown_answer


def test_read_only_authorizer_denies_unapproved_sql_functions():
    decision = api._read_only_authorizer(
        sqlite3.SQLITE_FUNCTION,
        None,
        "load_extension",
        "main",
        None,
    )

    assert decision == sqlite3.SQLITE_DENY


def test_read_only_authorizer_denies_restricted_admin_table():
    decision = api._read_only_authorizer(
        sqlite3.SQLITE_READ,
        "Usuarios administradores",
        "nombre",
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


def test_endpoint_explains_sqlite_authorization_errors(tmp_path):
    client, _ = _client_for_orders(
        tmp_path,
        [_order()],
        lambda _: "SELECT load_extension('extension.dll') "
        "FROM monthly_sales_by_country",
        unavailable_answer_generator=lambda question, limitation, history: (
            "Solo puedo consultar los análisis de pedidos disponibles."
        ),
    )

    response = _post(client, "ejecuta una extension")

    assert response.status_code == 200
    assert response.json()["answer"] == (
        "Solo puedo consultar los análisis de pedidos disponibles."
    )
    assert response.json()["sql"] is None


def test_endpoint_returns_direct_llm_response_without_query(tmp_path):
    app = api.create_app(
        tmp_path / "missing.sqlite",
        lambda _: api.LLMDirectResponse(
            "Puedo ayudarte con ventas mensuales por país o clientes principales."
        ),
    )

    response = _post(app, "¿Qué tiempo hace hoy?")

    assert response.status_code == 200
    assert response.json()["answer"].startswith("Puedo ayudarte")
    assert response.json()["sql"] is None
    assert response.json()["rows"] == []


def test_gemini_generator_returns_sql_from_json_response(monkeypatch):
    request_options = {}
    conversation_history = [
        {
            "question": "Ventas en España",
            "sql": "SELECT country FROM monthly_sales_by_country",
            "answer": "En España hubo ventas.",
        }
    ]

    def generate_content(**kwargs):
        request_options.update(kwargs)
        return SimpleNamespace(
            text=json.dumps(
                {
                    "responseType": "sql",
                    "modelResponse": "SELECT * FROM top_customers",
                }
            )
        )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    clients = _patch_gemini_client(monkeypatch, generate_content)

    sql = api.generate_sql_with_gemini(
        "clientes principales", conversation_history
    )

    assert sql == "SELECT * FROM top_customers"
    assert clients[0].api_key == "test-key"
    assert clients[0].closed is True
    assert request_options["model"] == "test-model"
    sql_context = json.loads(request_options["contents"])
    assert sql_context["question"] == "clientes principales"
    assert sql_context["conversation_history"] == conversation_history
    assert (
        datetime.fromisoformat(sql_context["current_datetime"]).utcoffset()
        is not None
    )
    assert request_options["config"].response_mime_type == "application/json"
    assert request_options["config"].response_schema is api.RespuestaJSON
    assert request_options["config"].system_instruction == api.SQL_GENERATION_PROMPT
    assert "SQLite" in request_options["config"].system_instruction
    assert "amount_cents INTEGER NOT NULL" in (
        request_options["config"].system_instruction
    )


def test_gemini_generator_returns_direct_response_from_json(monkeypatch):
    direct_answer = (
        "Puedo ayudar con ventas por mes y país o clientes principales."
    )
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch,
        lambda **_: SimpleNamespace(
            text=json.dumps(
                {"responseType": "answer", "modelResponse": direct_answer}
            )
        ),
    )

    result = api.generate_sql_with_gemini("¿Qué tiempo hace hoy?")

    assert result == api.LLMDirectResponse(direct_answer)


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
        ('{"responseType":"sql"}', "esquema JSON esperado"),
        ("[]", "esquema JSON esperado"),
        (
            '{"responseType":"maybe","modelResponse":"SELECT 1"}',
            "esquema JSON esperado",
        ),
        (
            '{"responseType":"sql","modelResponse":"   "}',
            "campo modelResponse vacío",
        ),
        (
            '{"responseType":"sql","modelResponse":"SELECT 1","extra":"value"}',
            "esquema JSON esperado",
        ),
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
    conversation_history = [
        {
            "question": "Ventas en España",
            "sql": "SELECT country FROM monthly_sales_by_country",
            "answer": "En España hubo ventas.",
        }
    ]

    def generate_content(**kwargs):
        request_options.update(kwargs)
        return SimpleNamespace(
            text=json.dumps(
                {
                    "responseType": "answer",
                    "modelResponse": "En España se vendieron 12.34.",
                }
            )
        )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    clients = _patch_gemini_client(monkeypatch, generate_content)

    answer = api.generate_answer_with_gemini(
        "¿Cuánto vendimos?",
        ["country", "total_sales"],
        [{"country": "España", "total_sales": 12.34}],
        conversation_history,
    )

    assert answer == "En España se vendieron 12.34."
    assert clients[0].api_key == "test-key"
    assert clients[0].closed is True
    assert request_options["model"] == "test-model"
    answer_context = json.loads(request_options["contents"])
    assert answer_context["question"] == "¿Cuánto vendimos?"
    assert answer_context["conversation_history"] == conversation_history
    assert (
        datetime.fromisoformat(answer_context["current_datetime"]).utcoffset()
        is not None
    )
    assert answer_context["columns"] == ["country", "total_sales"]
    assert answer_context["rows"] == [{"country": "España", "total_sales": 12.34}]
    assert request_options["config"].system_instruction == api.ANSWER_GENERATION_PROMPT
    assert request_options["config"].response_mime_type == "application/json"
    assert request_options["config"].response_schema is api.RespuestaJSON
    assert "readable Markdown table" in (
        request_options["config"].system_instruction
    )


def test_gemini_answer_generator_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(api.LLMQueryError, match="GEMINI_API_KEY"):
        api.generate_answer_with_gemini("ventas", [], [])


def test_gemini_answer_generator_reports_empty_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch, lambda **_: SimpleNamespace(text="   ")
    )

    with pytest.raises(api.LLMQueryError, match="respuesta vacía"):
        api.generate_answer_with_gemini("ventas", [], [])


def test_gemini_answer_generator_rejects_sql_response_type(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch,
        lambda **_: SimpleNamespace(
            text=json.dumps(
                {"responseType": "sql", "modelResponse": "SELECT 1"}
            )
        ),
    )

    with pytest.raises(api.LLMQueryError, match="respuesta en modelResponse"):
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


def test_gemini_unavailable_answer_uses_explanation_prompt(monkeypatch):
    request_options = {}
    conversation_history = [
        {"question": "ventas", "sql": None, "answer": "No hay datos visibles."}
    ]

    def generate_content(**kwargs):
        request_options.update(kwargs)
        return SimpleNamespace(
            text=json.dumps(
                {
                    "responseType": "answer",
                    "modelResponse": (
                        "Solo puedo consultar las métricas de pedidos disponibles."
                    ),
                }
            )
        )

    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    monkeypatch.setenv("GEMINI_MODEL", "test-model")
    clients = _patch_gemini_client(monkeypatch, generate_content)

    answer = api.generate_unavailable_answer_with_gemini(
        "Muéstrame los productos vendidos.",
        "La información no está disponible en las vistas permitidas.",
        conversation_history,
    )

    assert answer == "Solo puedo consultar las métricas de pedidos disponibles."
    assert clients[0].closed is True
    context = json.loads(request_options["contents"])
    assert context["question"] == "Muéstrame los productos vendidos."
    assert context["limitation"] == (
        "La información no está disponible en las vistas permitidas."
    )
    assert context["conversation_history"] == conversation_history
    assert request_options["config"].system_instruction == (
        api.UNAVAILABLE_ANSWER_PROMPT
    )
    assert request_options["config"].response_mime_type == "application/json"
    assert request_options["config"].response_schema is api.RespuestaJSON


def test_gemini_unavailable_answer_requires_api_key(monkeypatch):
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(api.LLMQueryError, match="GEMINI_API_KEY"):
        api.generate_unavailable_answer_with_gemini("ventas", "sin acceso")


def test_gemini_unavailable_answer_reports_empty_response(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch, lambda **_: SimpleNamespace(text="   ")
    )

    with pytest.raises(api.LLMQueryError, match="respuesta vacía"):
        api.generate_unavailable_answer_with_gemini("ventas", "sin acceso")


def test_gemini_unavailable_answer_rejects_sql_response_type(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch,
        lambda **_: SimpleNamespace(
            text=json.dumps(
                {"responseType": "sql", "modelResponse": "SELECT 1"}
            )
        ),
    )

    with pytest.raises(api.LLMQueryError, match="explicación en modelResponse"):
        api.generate_unavailable_answer_with_gemini("ventas", "sin acceso")


def test_gemini_unavailable_answer_reports_provider_errors(monkeypatch):
    monkeypatch.setenv("GEMINI_API_KEY", "test-key")
    _patch_gemini_client(
        monkeypatch,
        lambda **_: (_ for _ in ()).throw(
            api.errors.APIError(503, {"message": "simulated provider error"})
        ),
    )

    with pytest.raises(api.LLMQueryError, match="explicar"):
        api.generate_unavailable_answer_with_gemini("ventas", "sin acceso")
