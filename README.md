# qa-bit-llm

## Datos de pedidos y SQLite

El proyecto genera un CSV determinista con 1.000 pedidos válidos y 30 filas
con errores intencionales (12 duplicados, 10 fechas inválidas y 8 importes
vacíos). No requiere dependencias externas; necesita Python 3.10 o superior.

```powershell
python generate_orders.py
python clean_orders.py
```

Los archivos se crean en `data/orders.csv` y `data/orders.sqlite`. Para cargar
otra vez o usar rutas distintas:

```powershell
python clean_orders.py --csv data/orders.csv --db data/orders.sqlite
```

La carga elimina duplicados exactos y descarta filas con fecha no ISO o importe
vacío/no válido. Los importes se guardan como centavos enteros (`amount_cents`)
para evitar errores de redondeo. La tabla `orders` se vacía y recarga dentro de
una transacción, por lo que ejecutar el script varias veces no duplica pedidos.

## Pruebas y cobertura

Instala las dependencias y ejecuta las pruebas con cobertura de líneas y ramas:

```powershell
python -m pip install -r requirements.txt -r requirements-dev.txt
python -m pytest --cov=api --cov=clean_orders --cov=generate_orders --cov-branch --cov-report=term-missing
```

Las pruebas estándar simulan Gemini. Para probar tres consultas reales contra
Gemini y SQLite, con `GEMINI_API_KEY` definido en `.env`:

```powershell
$env:RUN_GEMINI_INTEGRATION = "1"
python -m pytest tests/test_gemini_database.py -k live_gemini
Remove-Item Env:RUN_GEMINI_INTEGRATION
```

Esta ejecución realiza llamadas reales a Gemini; puede consumir cuota de la API.

## API FastAPI

La carga de pedidos crea las vistas `monthly_sales_by_country` y
`top_customers`. Las ventas y el gasto de clientes cuentan únicamente pedidos
`paid` o `shipped`; las vistas exponen importes en unidades monetarias.

```powershell
python -m pip install -r requirements.txt
python generate_orders.py
python clean_orders.py
Copy-Item .env.example .env
# Edita .env y define GEMINI_API_KEY con tu clave, sin compartirla ni subirla.
uvicorn api:app --reload
```

Envía una pregunta a `POST /ask`:

```json
{"question": "¿Cuánto vendimos en España en marzo de 2025?"}
```

La respuesta incluye `answer`, un resumen textual redactado por Gemini usando
solo las filas que devolvió SQLite, además del SQL, las columnas, las filas y si
el resultado fue truncado. Antes de ejecutarlo, exige una única consulta `SELECT`
sobre las dos vistas permitidas; además, abre SQLite en modo de solo lectura y
limita la respuesta a 500 filas. Configura `GEMINI_MODEL` para elegir otro modelo.
`.env` está excluido de Git; las pruebas simulan Gemini y no requieren una clave
real. Si Gemini no puede redactar la respuesta, la API devuelve un error claro.

El SQL generado y el SQL validado (junto con la ruta de SQLite) aparecen en los
logs de Uvicorn. Si la API indica que faltan las vistas analíticas, detén el
servidor y ejecuta `python clean_orders.py`; esto recrea las vistas, pero también
recarga la tabla `orders` desde `data/orders.csv`. Luego vuelve a iniciar Uvicorn.

## Decisiones y siguientes pasos

- Se eligió SQLite para no requerir un servicio externo. El CSV usa fechas ISO;
  se eliminan duplicados exactos y se descartan filas con fechas inválidas o
  importes vacíos/no válidos. Los importes se persisten en centavos enteros.
- Las vistas consideran ventas solo los pedidos `paid` o `shipped`; los 10
  clientes se ordenan por gasto total, con desempate por nombre.
- Con más tiempo agregaría migraciones, autenticación y límites de uso para la
  API, observabilidad de errores del LLM y pruebas de carga.

## Uso de IA

Se pidió ayuda para crear datos reproducibles, limpieza y carga en SQLite,
vistas SQL, una API que convierta preguntas a consultas y pruebas de casos
válidos e inválidos. Se aceptaron la semilla fija, los importes en centavos,
la lista cerrada de vistas permitidas y el uso de un LLM simulado en tests.
Se corrigieron expectativas de pruebas para datasets pequeños y se endureció
la validación después de comprobar que un parser SQL podía aceptar un `SELECT`
sin columnas. No se incluyó ninguna clave en el código ni en Git.