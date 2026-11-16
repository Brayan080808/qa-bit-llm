import json
import logging
import os
import sqlite3
import tempfile
import threading
import time
from collections import OrderedDict
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Callable, Literal
from uuid import UUID, uuid4

import httpx
from dotenv import load_dotenv
from google import genai
from google.genai import errors, types
import sqlglot
from fastapi import FastAPI, File, HTTPException, UploadFile
from pydantic import BaseModel, Field, ValidationError
from sqlglot import exp
from sqlglot.errors import ParseError

import clean_orders
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


DATABASE_SCHEMA_PROMPT = _read_prompt("database_schema.txt")
SQL_GENERATION_PROMPT = (
    f"{_read_prompt('sql_generation.txt')}\n\n{DATABASE_SCHEMA_PROMPT}"
)
ANSWER_GENERATION_PROMPT = _read_prompt("answer_generation.txt")
UNAVAILABLE_ANSWER_PROMPT = _read_prompt("unavailable_answer.txt")

ALLOWED_VIEWS = frozenset({"monthly_sales_by_country", "top_customers"})
ALLOWED_QUERY_SOURCES = ALLOWED_VIEWS | {"orders"}
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
MAX_CSV_UPLOAD_BYTES = 10 * 1024 * 1024
CONVERSATION_TTL_SECONDS = 30 * 60
MAX_CONVERSATIONS = 1000
MAX_CONVERSATION_TURNS = 10


class LLMQueryError(RuntimeError):
    """Raised when the model cannot return usable SQL or a textual answer."""


class QueryValidationError(ValueError):
    """Raised when SQL is not a single read-only query over approved views."""


class AskRequest(BaseModel):
    question: str = Field(min_length=1, max_length=1000)
    conversation_id: UUID | None = None


class ClearOrdersRequest(BaseModel):
    confirm: bool = False


class RespuestaJSON(BaseModel):
    responseType: Literal["sql", "answer"]
    modelResponse: str


class AskResponse(BaseModel):
    conversation_id: UUID
    question: str
    answer: str
    sql: str | None
    columns: list[str]
    rows: list[dict[str, Any]]
    truncated: bool


class OrderUploadResponse(BaseModel):
    filename: str
    stats: dict[str, int]
    message: str


class ClearOrdersResponse(BaseModel):
    deleted_count: int
    cleared: bool


@dataclass(frozen=True)
class LLMDirectResponse:
    text: str


QueryGenerator = Callable[[str], str | LLMDirectResponse]
AnswerGenerator = Callable[[str, list[str], list[dict[str, Any]]], str]
ConversationTurn = dict[str, Any]
ConversationHistory = list[ConversationTurn]
UnavailableAnswerGenerator = Callable[[str, str, ConversationHistory], str]


class ConversationCache:
    def __init__(self) -> None:
        self._conversations: OrderedDict[
            str, tuple[float, ConversationHistory]
        ] = OrderedDict()
        self._lock = threading.Lock()

    def _remove_expired(self, now: float) -> None:
        expired = [
            conversation_id
            for conversation_id, (last_access, _) in self._conversations.items()
            if now - last_access >= CONVERSATION_TTL_SECONDS
        ]
        for conversation_id in expired:
            del self._conversations[conversation_id]

    def get_history(self, conversation_id: str) -> ConversationHistory:
        with self._lock:
            now = time.monotonic()
            self._remove_expired(now)
            entry = self._conversations.get(conversation_id)
            if entry is None:
                return []
            history = entry[1]
            self._conversations[conversation_id] = (now, history)
            self._conversations.move_to_end(conversation_id)
            return [turn.copy() for turn in history]

    def add_turn(self, conversation_id: str, turn: ConversationTurn) -> None:
        with self._lock:
            now = time.monotonic()
            self._remove_expired(now)
            entry = self._conversations.get(conversation_id)
            history = entry[1] if entry is not None else []
            history.append(turn.copy())
            del history[:-MAX_CONVERSATION_TURNS]
            self._conversations[conversation_id] = (now, history)
            self._conversations.move_to_end(conversation_id)
            while len(self._conversations) > MAX_CONVERSATIONS:
                self._conversations.popitem(last=False)

    def clear(self, conversation_id: str) -> bool:
        with self._lock:
            return self._conversations.pop(conversation_id, None) is not None


def _current_datetime_context() -> str:
    return datetime.now().astimezone().isoformat(timespec="seconds")


def _gemini_json_config(
    system_instruction: str, max_output_tokens: int = 256
) -> types.GenerateContentConfig:
    return types.GenerateContentConfig(
        system_instruction=system_instruction,
        response_mime_type="application/json",
        response_schema=RespuestaJSON,
        temperature=0,
        max_output_tokens=max_output_tokens,
    )


def _parse_gemini_response(response_text: str | None) -> RespuestaJSON:
    if not response_text or not response_text.strip():
        raise LLMQueryError("Gemini devolvió una respuesta vacía.")
    try:
        payload = json.loads(response_text)
    except json.JSONDecodeError as exc:
        raise LLMQueryError("Gemini no devolvió un objeto JSON válido.") from exc
    if not isinstance(payload, dict) or set(payload) != {
        "responseType",
        "modelResponse",
    }:
        raise LLMQueryError(
            "Gemini no devolvió el esquema JSON esperado con responseType y modelResponse."
        )
    try:
        result = RespuestaJSON.model_validate(payload)
    except ValidationError as exc:
        raise LLMQueryError(
            "Gemini no devolvió el esquema JSON esperado con responseType y modelResponse."
        ) from exc
    if not result.modelResponse.strip():
        raise LLMQueryError("Gemini devolvió un campo modelResponse vacío.")
    return result


def generate_sql_with_gemini(
    question: str, conversation_history: ConversationHistory | None = None
) -> str | LLMDirectResponse:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMQueryError("Falta configurar GEMINI_API_KEY.")

    contents = json.dumps(
        {
            "question": question,
            "current_datetime": _current_datetime_context(),
            "conversation_history": conversation_history or [],
        },
        ensure_ascii=False,
    )
    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                contents=contents,
                config=_gemini_json_config(SQL_GENERATION_PROMPT),
            )
    except (errors.APIError, httpx.RequestError) as exc:
        raise LLMQueryError("Gemini no pudo completar la solicitud.") from exc

    result = _parse_gemini_response(response.text)
    if result.responseType == "sql":
        return result.modelResponse.strip()
    return LLMDirectResponse(result.modelResponse.strip())


def generate_answer_with_gemini(
    question: str,
    columns: list[str],
    rows: list[dict[str, Any]],
    conversation_history: ConversationHistory | None = None,
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
            "conversation_history": conversation_history or [],
        },
        ensure_ascii=False,
    )
    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                contents=contents,
                config=_gemini_json_config(ANSWER_GENERATION_PROMPT),
            )
    except (errors.APIError, httpx.RequestError) as exc:
        raise LLMQueryError("Gemini no pudo generar la respuesta textual.") from exc

    result = _parse_gemini_response(response.text)
    if result.responseType != "answer":
        raise LLMQueryError("Gemini no devolvió una respuesta en modelResponse.")
    return result.modelResponse.strip()


def generate_unavailable_answer_with_gemini(
    question: str,
    limitation: str,
    conversation_history: ConversationHistory | None = None,
) -> str:
    api_key = os.getenv("GEMINI_API_KEY")
    if not api_key:
        raise LLMQueryError("Falta configurar GEMINI_API_KEY.")

    contents = json.dumps(
        {
            "question": question,
            "current_datetime": _current_datetime_context(),
            "conversation_history": conversation_history or [],
            "limitation": limitation,
        },
        ensure_ascii=False,
    )
    try:
        with genai.Client(api_key=api_key) as client:
            response = client.models.generate_content(
                model=os.getenv("GEMINI_MODEL", "gemini-3.5-flash-lite"),
                contents=contents,
                config=_gemini_json_config(UNAVAILABLE_ANSWER_PROMPT),
            )
    except (errors.APIError, httpx.RequestError) as exc:
        raise LLMQueryError(
            "Gemini no pudo explicar por qué no fue posible consultar esos datos."
        ) from exc

    result = _parse_gemini_response(response.text)
    if result.responseType != "answer":
        raise LLMQueryError("Gemini no devolvió una explicación en modelResponse.")
    return result.modelResponse.strip()


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
            or table.name.lower() not in ALLOWED_QUERY_SOURCES
        ):
            raise QueryValidationError(
                "The query references a database object that is not allowed."
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
        if table_name == "orders" and (
            not view_source or view_source in ALLOWED_VIEWS
        ):
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
    unavailable_answer_generator: UnavailableAnswerGenerator | None = None,
) -> FastAPI:
    app = FastAPI(title="Orders Analytics API", version="1.0.0")
    conversation_cache = ConversationCache()

    @app.post("/orders/upload", response_model=OrderUploadResponse)
    async def upload_orders(file: UploadFile = File(...)) -> OrderUploadResponse:
        filename = file.filename or ""
        if Path(filename).suffix.lower() != ".csv":
            await file.close()
            raise HTTPException(status_code=415, detail="Debes subir un archivo .csv.")

        try:
            contents = await file.read(MAX_CSV_UPLOAD_BYTES + 1)
        finally:
            await file.close()
        if len(contents) > MAX_CSV_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail="El archivo CSV supera el límite de 10 MB.",
            )

        temporary_path: Path | None = None
        try:
            with tempfile.NamedTemporaryFile(suffix=".csv", delete=False) as temp_file:
                temp_file.write(contents)
                temporary_path = Path(temp_file.name)
            stats = clean_orders.load_orders(temporary_path, db_path)
        except ValueError as exc:
            raise HTTPException(status_code=422, detail=str(exc)) from exc
        except sqlite3.Error as exc:
            LOGGER.exception("No se pudo cargar el archivo CSV en SQLite.")
            raise HTTPException(
                status_code=500, detail="No se pudo cargar el CSV en la base de datos."
            ) from exc
        finally:
            if temporary_path is not None:
                temporary_path.unlink(missing_ok=True)

        return OrderUploadResponse(
            filename=Path(filename).name,
            stats=stats,
            message="CSV validado, limpiado y cargado correctamente.",
        )

    @app.delete("/orders", response_model=ClearOrdersResponse)
    def delete_orders(request: ClearOrdersRequest) -> ClearOrdersResponse:
        if not request.confirm:
            raise HTTPException(
                status_code=400,
                detail="Confirma la eliminación enviando {\"confirm\": true}.",
            )
        try:
            deleted_count = clean_orders.clear_orders(db_path)
        except FileNotFoundError as exc:
            raise HTTPException(
                status_code=503,
                detail="La base de datos no existe. Sube primero un CSV.",
            ) from exc
        except sqlite3.OperationalError as exc:
            LOGGER.exception("No se pudieron vaciar los pedidos de SQLite.")
            if "no such table: orders" in str(exc).lower():
                raise HTTPException(
                    status_code=503,
                    detail="La base de datos todavía no contiene la tabla de pedidos.",
                ) from exc
            raise HTTPException(
                status_code=500, detail="No se pudieron vaciar los pedidos."
            ) from exc
        except sqlite3.Error as exc:
            LOGGER.exception("No se pudieron vaciar los pedidos de SQLite.")
            raise HTTPException(
                status_code=500, detail="No se pudieron vaciar los pedidos."
            ) from exc
        return ClearOrdersResponse(deleted_count=deleted_count, cleared=True)

    def generate_query(
        question: str, history: ConversationHistory
    ) -> str | LLMDirectResponse:
        if query_generator is None:
            return generate_sql_with_gemini(question, history)
        return query_generator(question)

    def generate_answer(
        question: str,
        columns: list[str],
        rows: list[dict[str, Any]],
        history: ConversationHistory,
    ) -> str:
        if answer_generator is None:
            return generate_answer_with_gemini(question, columns, rows, history)
        return answer_generator(question, columns, rows)

    def generate_unavailable_answer(
        question: str, history: ConversationHistory
    ) -> str:
        limitation = (
            "La solicitud no se pudo responder con una consulta segura sobre los "
            "datos de pedidos disponibles."
        )
        if unavailable_answer_generator is None:
            return generate_unavailable_answer_with_gemini(
                question, limitation, history
            )
        return unavailable_answer_generator(question, limitation, history)

    def response_without_query(
        question: str, conversation_id: UUID, history: ConversationHistory
    ) -> AskResponse:
        try:
            answer = generate_unavailable_answer(question, history)
        except LLMQueryError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        conversation_cache.add_turn(
            str(conversation_id),
            {"question": question, "sql": None, "answer": answer},
        )
        return AskResponse(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            sql=None,
            columns=[],
            rows=[],
            truncated=False,
        )

    @app.delete("/conversations/{conversation_id}")
    def clear_conversation(conversation_id: UUID) -> dict[str, Any]:
        if not conversation_cache.clear(str(conversation_id)):
            raise HTTPException(
                status_code=404, detail="No hay contexto activo para esa conversación."
            )
        return {"conversation_id": conversation_id, "cleared": True}

    @app.post("/ask", response_model=AskResponse)
    def ask(request: AskRequest) -> AskResponse:
        question = request.question.strip()
        if not question:
            raise HTTPException(status_code=422, detail="La pregunta no puede estar vacía.")
        conversation_id = request.conversation_id or uuid4()
        conversation_key = str(conversation_id)
        history = conversation_cache.get_history(conversation_key)

        try:
            generated_result = generate_query(question, history)
        except LLMQueryError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc
        if isinstance(generated_result, LLMDirectResponse):
            conversation_cache.add_turn(
                conversation_key,
                {
                    "question": question,
                    "sql": None,
                    "answer": generated_result.text,
                },
            )
            return AskResponse(
                conversation_id=conversation_id,
                question=question,
                answer=generated_result.text,
                sql=None,
                columns=[],
                rows=[],
                truncated=False,
            )
        LOGGER.info("SQL generado para la pregunta: %r", generated_result)

        try:
            safe_sql = validate_read_query(generated_result)
        except QueryValidationError:
            LOGGER.warning(
                "Gemini generó una consulta que no coincide con el esquema permitido."
            )
            return response_without_query(question, conversation_id, history)

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
            if "not authorized" in str(exc).lower():
                return response_without_query(question, conversation_id, history)
            raise HTTPException(
                status_code=502,
                detail="La consulta generada no se pudo ejecutar en SQLite.",
            ) from exc

        try:
            answer = generate_answer(question, columns, rows, history)
        except LLMQueryError as exc:
            raise HTTPException(status_code=502, detail=str(exc)) from exc

        conversation_cache.add_turn(
            conversation_key,
            {"question": question, "sql": safe_sql, "answer": answer},
        )
        return AskResponse(
            conversation_id=conversation_id,
            question=question,
            answer=answer,
            sql=safe_sql,
            columns=columns,
            rows=rows,
            truncated=truncated,
        )

    return app


app = create_app()
