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

Instala las dependencias de desarrollo y ejecuta las pruebas con cobertura de
líneas y ramas:

```powershell
python -m pip install -r requirements-dev.txt
python -m pytest --cov=clean_orders --cov=generate_orders --cov-branch --cov-report=term-missing
```