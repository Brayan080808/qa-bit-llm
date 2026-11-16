# qa-bit-llm

## Instalacion resumen
- python generate_orders.py
- python clean_orders.py --csv data/orders.csv --db data/orders.sqlite
- python -m pip install -r requirements.txt -r requirements-dev.txt
- colocar tu api key en el .env (GEMINI_API_KEY)
- python setup_project.py
uvicorn api:app --reload

## Test resumen
- python -m pytest
- python -m pytest --cov=api --cov=clean_orders --cov=generate_orders --cov-branch --cov-report=term-missing
- $env:RUN_GEMINI_INTEGRATION = "1"
python -m pytest tests/test_gemini_database.py -k live_gemini
Remove-Item Env:RUN_GEMINI_INTEGRATION (Nota puede llegar a fallar por los limites de la api)

## Decisiones y siguientes pasos

- Se eligió SQLite para no requerir un servicio externo. El CSV usa fechas ISO;
  se eliminan duplicados exactos y se descartan filas con fechas inválidas o
  importes vacíos/no válidos. Los importes se persisten en centavos enteros.
- Las vistas consideran ventas solo los pedidos `paid` o `shipped`; los 10
  clientes se ordenan por gasto total, con desempate por nombre.
- Con más tiempo agregaría migraciones, autenticación y límites de uso para la
  API, observabilidad de errores del LLM y pruebas de carga.
- Se implementarian semaforos para limitar las consultar al mismo tiempo y evitar saturar el sistema 
- Impleria cache con Redis para las preguntas mas frecuentes
- Anadira mas documentacion para mayor contexto respecto a logica de negocio


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
También se incluyen 20 casos de integración (consultas válidas y solicitudes no
permitidas). Para ejecutar solo esos casos y mostrar cada salida JSON:

```powershell
$env:RUN_GEMINI_INTEGRATION = "1"
python -m pytest -s tests/test_gemini_database.py -k live_gemini_prompt
Remove-Item Env:RUN_GEMINI_INTEGRATION
```

Cada uno de estos 20 casos realiza una llamada real a Gemini y puede consumir
cuota de la API.

## API FastAPI

La base incluye la tabla `orders` y las vistas `monthly_sales_by_country` y
`top_customers`. Se permiten consultas `SELECT` de solo lectura sobre esos tres
orígenes. Las vistas de ventas y clientes cuentan únicamente pedidos `paid` o
`shipped`; la tabla contiene pedidos individuales y guarda `amount_cents` en
centavos enteros. La tabla administrativa que aparece en el esquema está
explícitamente bloqueada.

```powershell
python -m pip install -r requirements.txt
Copy-Item .env.example .env
# Edita .env y define GEMINI_API_KEY con tu clave, sin compartirla ni subirla.
python setup_project.py
uvicorn api:app --reload
```

`setup_project.py` pregunta si deseas generar un CSV de ejemplo o cargar uno
propio. La estructura requerida, en este orden, es:

```text
order_date,customer,country,product,amount,status
2025-03-01,Customer 001,Chile,Laptop,12.34,paid
```

La carga limpia duplicados, fechas inválidas e importes vacíos o inválidos y
reemplaza los pedidos de SQLite en una transacción.

También puedes cargar el CSV desde `POST /orders/upload` como `multipart/form-data`
con un campo `file`; la API devuelve el resumen de filas leídas, descartadas y
cargadas. El endpoint rechaza CSV con una estructura incorrecta y archivos
mayores a 10 MB.

Para vaciar la tabla desde `DELETE /orders`, envía explícitamente
`{"confirm": true}`. El endpoint solo borra pedidos; conserva la base y las
vistas analíticas.

Envía una pregunta a `POST /ask`:

```json
{"question": "¿Cuánto vendimos en España en marzo de 2025?"}
```

La respuesta devuelve un `conversation_id`. Envíalo en las siguientes
solicitudes para conservar el contexto de esa conversación:

```json
{
  "question": "¿Y en abril?",
  "conversation_id": "UUID-devuelto-por-la-primera-respuesta"
}
```

Para borrar el contexto de una conversación, llama a
`DELETE /conversations/{conversation_id}`. El contexto vive en memoria: expira
tras 30 minutos de inactividad, conserva hasta 10 turnos por conversación, limita
la caché a 1.000 conversaciones y se pierde al reiniciar la aplicación. No se
comparte entre varios procesos de Uvicorn. Mantén privado el `conversation_id`;
no hay autenticación en esta API.

Si la pregunta se puede responder con los datos, Gemini genera una consulta que
se valida antes de ejecutarla. Si queda fuera del análisis de pedidos o pide
información que no está en las vistas, Gemini responde directamente sin SQL.
Si una consulta generada no es segura o no se ajusta al esquema, no se ejecuta y
Gemini explica la limitación en lenguaje natural. Cuando no hay filas, la
respuesta también lo indica sin inventar resultados. En respuestas directas,
`sql` es `null` y `columns` y `rows` están vacíos.

La respuesta incluye `answer`, redactada por Gemini usando el resultado de
SQLite y el contexto reciente, además del SQL (si aplica), las columnas, las
filas y si el resultado fue truncado. Cuando hay filas, Gemini las presenta en
una tabla Markdown legible. La API solo permite una consulta `SELECT` sobre los
orígenes autorizados; abre SQLite en modo de solo lectura y limita la respuesta
a 500 filas. Configura `GEMINI_MODEL` para elegir otro modelo.
`.env` está excluido de Git; las pruebas simulan Gemini y no requieren una clave
real. Si Gemini no puede redactar la respuesta, la API devuelve un error claro.

Las instrucciones del modelo están en
`prompts/sql_generation.txt`, `prompts/answer_generation.txt` y
`prompts/unavailable_answer.txt`. El esquema convertido de JSON a texto está en
`prompts/database_schema.txt` y se añade al prompt SQL al iniciar la API. Los
prompts y las propiedades estructuradas enviadas al modelo usan inglés; las
respuestas al usuario se mantienen en español salvo que pida otro idioma. Todas
las llamadas a Gemini solicitan JSON estructurado mediante `response_schema`;
la salida lleva el tipo en `responseType` y el SQL o mensaje en `modelResponse`.
Cada petición recibe la fecha y hora local actual en ISO 8601 con zona horaria.

El SQL generado y el SQL validado (junto con la ruta de SQLite) aparecen en los
logs de Uvicorn. Si la API indica que faltan las vistas analíticas, detén el
servidor y ejecuta `python clean_orders.py`; esto recrea las vistas, pero también
recarga la tabla `orders` desde `data/orders.csv`. Luego vuelve a iniciar Uvicorn.


## Uso de IA

Se pidió ayuda para crear datos reproducibles, limpieza y carga en SQLite,
vistas SQL, una API que convierta preguntas a consultas y pruebas de casos
válidos e inválidos. Se aceptaron la semilla fija, los importes en centavos,
la lista cerrada de vistas permitidas y el uso de un LLM simulado en tests.
Se corrigieron expectativas de pruebas para datasets pequeños y se endureció
la validación después de comprobar que un parser SQL podía aceptar un `SELECT`
sin columnas. No se incluyó ninguna clave en el código ni en Git.