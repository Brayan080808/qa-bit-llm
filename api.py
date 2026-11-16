import json
import logging
import os
import sqlite3
from datetime import datetime
from pathlib import Path
from typing import Any, Callable

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import sqlglot
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field
from sqlglot import exp
from sqlglot.errors import ParseError

from clean_orders import DEFAULT_DB_PATH


LOGGER = logging.getLogger("uvicorn.error")
APP_DIR = Path(__file__).resolve().parent
load_dotenv(APP_DIR / ".env")
PROMPTS_DIR = APP_DIR / "prompts"


def _read_prompt(filename: str) -> str:
    prompt_path = PROMPTS_DIR / filename
    try:
        prompt = prompt_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        raise RuntimeError(f"No se pudo cargar el prompt: {prompt_path}") from exc
    if not prompt:
        raise RuntimeError(f"El archivo de prompt está vacío: {prompt_path}")
    return prompt


SQL_GENERATION_PROMPT = _read_prompt("sql_generation.txt")
ANSWER_GENERATION_PROMPT = _read_prompt("answer_generation.txt")

ALLOWED_VIEWS = frozenset({"monthly_sales_by_country", "top_customers"})
ALLOWED_SQL_FUNCTIONS = frozenset(
    {
        "avg",
        "coalesce",
        "count",
        "ifnull",
        "like",
        "lower",
        "max",
        "min",
        "round",
        "strftime",
        "substr",
        "sum",
        "upper",
    }
)
MAX_RESULT_ROWS = 500


class LLMQueryError(RuntimeError):
    """Raised when the model cannot return usable SQL or a textual answer."""


class QueryValidationError(ValueError):
    """Raised when SQL is not a single read-only query over approved views."""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)


class AskResponse(BaseModel):
    question: str
    answer: str
    sql: str
    columns: list[str]
    rows: list[dict[str, Any]]
    truncated: bool


QueryGenerator = Callable[[str], str]
AnswerGenerator = Callable[[str, list[str], list[dict[str, Any]]], str]


def _current_datetime_context() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def generate_sql_with_gemini(question: str) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMQueryError("Falta configurar GEMINI_API_KEY.")

    contents = json.dumps(
        {"question": question, "current_datetime": _current_datetime_context()},
        ensure_ascii=False,
    )
    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=SQL_GENERATION_PROMPT,
                    response_mime_type="application/json",
                    temperature=0,
                    max_output_tokens=256,
                ),
            )
    except (errors.APIError, httpx.RequestError) as exc:
        raise LLMQueryError("Gemini no pudo completar la solicitud.") from exc

    response_text = (response.text or "").strip()
    if not response_text:
        raise LLMQueryError("Gemini devolvió una respuesta vacía.")
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise LLMQueryError("Gemini no devolvió un objeto JSON válido.") from exc
    if not isinstance(payload, dict) or not isinstance(payload.get("sql"), str):
        raise LLMQueryError("La respuesta de Gemini no contiene una consulta SQL.")
    return payload["sql"].strip()


def generate_answer_with_gemini(
    question: str, columns: list[str], rows: list[dict[str, Any]]
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMQueryError("Falta configurar GEMINI_API_KEY.")

    contents = json.dumps(
        {
            "question": question,
            "current_datetime": _current_datetime_context(),
            "columns": columns,
            "rows": rows,
        },
        ensure_ascii=False,
    )
    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                contents=contents,
                config=types.GenerateContentConfig(
                    system_instruction=ANSWER_GENERATION_PROMPT,
                    temperature=0,
                    max_output_tokens=256,
                ),
            )
    except (errors.APIError, httpx.RequestError) as exc:
        raise LLMQueryError("Gemini no pudo generar la respuesta textual.") from exc

    answer = (response.text or "").strip()
    if not answer:
        raise LLMQueryError("Gemini devolvió una respuesta textual vacía.")
    return answer


def validate_read_query(sql: str) -> str:
    if not isinstance(sql, str) or not sql.strip():
        raise QueryValidationError("La consulta está vacía.")
    try:
        statements = sqlglot.parse(sql, read="sqlite")
    except ParseError as exc:
        raise QueryValidationError("La sintaxis SQL no es válida.") from exc

    if len(statements) != 1 or statements[0] is None:
        raise QueryValidationError("Se permite una sola sentencia SQL.")
    statement = statements[0]
    if not isinstance(statement, exp.Select):
        raise QueryValidationError("Solo se permiten consultas SELECT.")
    if not statement.expressions:
        raise QueryValidationError("La consulta SELECT debe incluir columnas.")

    tables = list(statement.find_all(exp.Table))
    if not tables:
        raise QueryValidationError("La consulta debe leer una vista permitida.")
    for table in tables:
        if (
            table.catalog
            or table.db
            or table.name.lower() not in ALLOWED_VIEWS
        ):
            raise QueryValidationError(
                "Solo se permite consultar monthly_sales_by_country y top_customers."
            )

    return statement.sql(dialect="sqlite")


def _read_only_authorizer(action, argument_one, argument_two, database, source):
    if action in (sqlite3.SQLITE_SELECT, getattr(sqlite3, "SQLITE_RECURSIVE", -1)):
        return sqlite3.SQLITE_OK
    if action == sqlite3.SQLITE_READ:
        table_name = (argument_one or "").lower()
        view_source = (source or "").lower()
        if table_name in ALLOWED_VIEWS:
            return sqlite3.SQLITE_OK
        if table_name == "orders" and view_source in ALLOWED_VIEWS:
            return sqlite3.SQLITE_OK
        return sqlite3.SQLITE_DENY
    if action == sqlite3.SQLITE_FUNCTION:
        function_name = (argument_two or argument_one or "").lower()
        if function_name in ALLOWED_SQL_FUNCTIONS:
            return sqlite3.SQLITE_OK
    return sqlite3.SQLITE_DENY


def execute_read_query(
    db_path: Path, sql: str
) -> tuple[list[str], list[dict[str, Any]], bool]:
    if not db_path.is_file():
        raise FileNotFoundError(db_path)

    database_uri = f"{db_path.resolve().as_uri()}?mode=ro"
    connection = sqlite3.connect(database_uri, uri=True)
    try:
        connection.row_factory = sqlite3.Row
        connection.execute("PRAGMA query_only = ON")
        connection.set_authorizer(_read_only_authorizer)
        cursor = connection.execute(sql)
        columns = [column[0] for column in cursor.description or ()]
        fetched_rows = cursor.fetchmany(MAX_RESULT_ROWS + 1)
        truncated = len(fetched_rows) > MAX_RESULT_ROWS
        rows = [dict(row) for row in fetched_rows[:MAX_RESULT_ROWS]]
        return columns, rows, truncated
    finally:
        connection.close()


def create_app(
    db_path: Path = DEFAULT_DB_PATH,
    query_generator: QueryGenerator | None = None,
    answer_generator: AnswerGenerator | None = None,
) -> FastAPI:
    app = FastAPI(title="Orders Analytics API", version="1.0.0")
    generate_query = (
        query_generator if query_generator is not None else generate_sql_with_gemini
    )
    generate_answer = (
        answer_generator
        if answer_generator is not None
        else generate_answer_with_gemini
    )

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="La pregunta no puede estar vacía.")

        try:
            generated_sql = generate_query(question)
        except LLMQueryError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        LOGGER.info("SQL generado para la pregunta: %r", generated_sql)

        try:
            safe_sql = validate_read_query(generated_sql)
        except QueryValidationError as exc:
            raise HTTPException(
                status_code=502,
                detail=f"El modelo devolvió una consulta no permitida: {exc}",
            ) from exc

        LOGGER.info(
            "Ejecutando SQL validado en SQLite (%s): %r",
            db_path.resolve(),
            safe_sql,
        )
        try:
            columns, rows, truncated = execute_read_query(db_path, safe_sql)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="La base de datos no existe. Ejecuta primero clean_orders.py.",
            ) from exc
        except sqlite3.Error as exc:
            LOGGER.exception(
                "No se pudo ejecutar SQL validado en SQLite (%s): %r",
                db_path.resolve(),
                safe_sql,
            )
            if (
                isinstance(exc, sqlite3.OperationalError)
                and "no such table:" in str(exc).lower()
            ):
                raise HTTPException(
                    status_code=503,
                    detail=(
                        "La base SQLite no contiene las vistas analíticas. Ejecuta "
                        "`python clean_orders.py` para regenerarlas; esto recarga "
                        "la tabla orders desde el CSV."
                    ),
                ) from exc
            raise HTTPException(
                status_code=502,
                detail="La consulta generada no se pudo ejecutar en SQLite.",
            ) from exc

        try:
            answer = generate_answer(question, columns, rows)
        except LLMQueryError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        return AskResponse(
            question=question,
            answer=answer,
            sql=safe_sql,
            columns=columns,
            rows=rows,
            truncated=truncated,
        )

    return app


app = create_app()
