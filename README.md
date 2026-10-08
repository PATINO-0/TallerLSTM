# Energy · LSTM para predicción de demanda energética

Aplicación completa en español: backend FastAPI y frontend HTML/CSS/JavaScript, en un solo proyecto desplegable en Vercel. Utiliza **el modelo entrenado existente**, sin reentrenarlo ni modificar sus artefactos.

La interfaz usa textos más grandes, colores de mayor contraste y un flujo de carga de historial → comprobación de la hora → predicción. La edición manual queda en un panel desplegable; cargar un CSV completo permite predecir directamente.

El modelo predice la **demanda eléctrica de la siguiente hora, en MW**, a partir de **24 observaciones horarias consecutivas con 16 variables cada una**. Una fila aislada no es suficiente para este LSTM.

## Inicio local

Necesitas Python **3.12 o 3.13** de 64 bits. No necesitas Node.js para ejecutar la aplicación. TensorFlow se ejecuta en CPU; no necesitas GPU. La primera instalación descarga varios cientos de MB.

Desde la raíz del proyecto, en PowerShell:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Abre **http://127.0.0.1:8000**. La documentación interactiva de la API está en **http://127.0.0.1:8000/docs**.

También puedes iniciar todo con:

```powershell
powershell -ExecutionPolicy Bypass -File .\scripts\start.ps1
```

En Linux/macOS:

```bash
python3 -m venv .venv
.venv/bin/python -m pip install -r requirements.txt
.venv/bin/python -m uvicorn app:app --host 127.0.0.1 --port 8000
```

Si Windows muestra un error de DLL al importar TensorFlow, instala el [Microsoft Visual C++ Redistributable de 64 bits](https://learn.microsoft.com/en-us/cpp/windows/latest-supported-vc-redist) y reinicia el servidor.

## Usar la interfaz

1. Espera a que aparezca **Modelo conectado**. La primera carga puede tardar; puedes pulsar el indicador para reintentar.
2. Selecciona **Cargar ejemplo** para probar con datos sintéticos, **Importar CSV** para cargar tus mediciones o completa manualmente las 24 horas.
3. Comprueba la fecha destacada en **Hora que se estimará**: siempre es una hora después de la última medición del historial. Con un ejemplo o CSV completo puedes predecir directamente, sin abrir el editor.
4. Para ingresar datos a mano, pulsa **Ingresar a mano**, selecciona la última hora observada en Colombia y completa cada hora con **Guardar y continuar**. También puedes abrir **Revisar o editar las mediciones**. Se generan las variables de calendario; el período del día y los festivos son editables.
5. Pulsa **Predecir demanda**. Verás el resultado en MW, su variación frente a la última hora y el gráfico del historial con la predicción.
6. **Descargar resultado y entradas** guarda un JSON con las observaciones exactas utilizadas y la respuesta. Al cambiar una entrada, la predicción anterior se descarta para evitar resultados desactualizados.

El ejemplo **no contiene mediciones reales**. Usa datos observados y las unidades del entrenamiento para una predicción operativa. La API nunca devuelve una predicción simulada si falla el modelo.

### ¿Basta con hora futura, clima previsto, precio previsto y festivo?

**No para el modelo guardado en este proyecto.** El archivo Keras tiene entrada `(batch, 24, 16)` e incluye `demanda_mw` histórica. Fue entrenado para estimar la siguiente hora usando 24 observaciones anteriores. Hora futura, temperatura, humedad, viento, radiación, precipitación, precio y festivo de una sola hora no reemplazan ese historial; tampoco corresponde sustituir el clima histórico por pronósticos futuros sin reentrenar.

La interfaz simplifica la captura mediante un CSV de **fecha + siete mediciones + festivo** por hora. Las otras ocho variables se generan desde la fecha. Se conserva el uso de 24 horas reales y de la demanda histórica. Un formulario que utilice únicamente las ocho entradas futuras requeriría otro entrenamiento y un conjunto de datos que relacione esos pronósticos con la demanda objetivo.

El escalador del precio tiene promedio aproximado **24,91** y desviación estándar **5,85**. Un precio de **810** requiere comprobar la moneda y escala originales. La interfaz muestra un aviso junto al campo cuando está alejado de esa distribución; conserva el valor ingresado y no aplica una conversión monetaria inventada.

## Variables independientes X y salida Y

El orden se conserva explícitamente a partir de `feature_columns` y se verifica contra `metadata.json` y el escalador:

| Variable | Descripción / unidad |
| --- | --- |
| `demanda_mw` | Demanda **histórica observada** en MW; entrada de cada hora |
| `temperatura_c` | Temperatura en °C |
| `humedad_pct` | Humedad entre 0 y 100 % |
| `viento_kmh` | Velocidad del viento, km/h |
| `radiacion_wm2` | Radiación solar, W/m² |
| `precipitacion_mm` | Precipitación, mm |
| `precio_kwh` | Precio por kWh, **en la moneda y escala usadas al entrenar** |
| `hora` | Entero 0–23 |
| `dia_semana` | Entero 0–6: lunes=0, domingo=6 |
| `fin_semana` | 1 para sábado/domingo, 0 para los demás días |
| `festivo` | 1 si ese día es festivo, 0 si no lo es |
| `mes` | Entero 1–12 |
| `periodo_dia_madrugada` | Indicador 0/1 |
| `periodo_dia_manana` | Indicador 0/1 |
| `periodo_dia_tarde` | Indicador 0/1 |
| `periodo_dia_noche` | Indicador 0/1 |

Exactamente un indicador de período debe ser 1 por fila. La salida se llama `demanda_objetivo` en el escalador guardado y se entrega como **`predicted_demand_mw`**.

### Convenciones que debes comprobar con el entrenamiento

Los artefactos incluyen los nombres de las categorías, pero **no incluyen el código de ingeniería de calendario ni la moneda del precio**. La aplicación usa lunes=0, sábado/domingo como fin de semana y America/Bogota. Como valores iniciales del selector usa madrugada 00–05, mañana 06–11, tarde 12–17 y noche 18–23. **Estos cortes son una convención de la interfaz, no una regla verificada del entrenamiento.** El período puede editarse en cada hora; el CSV/API conserva los cuatro indicadores que suministres, sin recalcularlos. Verifica también la codificación del día de semana en el notebook original antes de usar mediciones operativas; si difiere, adapta la validación y el generador de calendario.

`precio_kwh` no lleva una etiqueta de moneda inventada. Por ejemplo, introducir 800 cuando el entrenamiento utilizó una escala alrededor de 25 puede producir una entrada fuera de distribución. La API avisa cuando alguna de las siete mediciones se aleja más de tres desviaciones estándar de los datos de entrenamiento; esto no garantiza detectar todos los casos problemáticos.

## Importar CSV

Descarga **Descargar plantilla CSV** desde la interfaz o `GET /api/template.csv`. Contiene una cabecera y 24 filas: las siete mediciones están vacías para que ingreses tus datos, las fechas se inicializan para las últimas horas y `festivo` vale 0. Puedes modificar las fechas manteniendo la cronología. El frontend genera las variables de calendario con las convenciones descritas arriba.

La **plantilla sencilla** tiene estas nueve columnas:

```csv
timestamp,demanda_mw,temperatura_c,humedad_pct,viento_kmh,radiacion_wm2,precipitacion_mm,precio_kwh,festivo
```

También se conserva el **formato completo** para quienes ya tengan las variables de calendario calculadas. Puedes descargarlo en `GET /api/template.csv?layout=full`:

```csv
timestamp,demanda_mw,temperatura_c,humedad_pct,viento_kmh,radiacion_wm2,precipitacion_mm,precio_kwh,hora,dia_semana,fin_semana,festivo,mes,periodo_dia_madrugada,periodo_dia_manana,periodo_dia_tarde,periodo_dia_noche
```

- Exactamente **24 filas**, de la más antigua a la más reciente.
- El formato sencillo requiere **exactamente sus nueve columnas**; el completo requiere **las 16 variables** y admite `timestamp`. El orden de las columnas puede variar.
- `timestamp` es **obligatorio en la plantilla sencilla** y opcional en el formato completo. Si lo incluyes, úsalo en todas las filas, en ISO 8601 con zona horaria: `2026-10-07T14:00:00-05:00`. Las horas deben ser exactas y estar separadas por una hora. En el formato completo, los campos de calendario deben coincidir con la fecha en Colombia y los períodos suministrados se conservan.
- Sin `timestamp`, se conservan las variables de calendario ingresadas y se devuelve la siguiente hora sin inventar una fecha absoluta.
- Se admite coma con decimal punto, o punto y coma con decimal coma. No uses separadores de miles. Se admiten BOM de Excel, CRLF y campos entre comillas.
- Tamaño máximo de CSV: **256 KB**. Se rechazan celdas vacías, columnas desconocidas/duplicadas, valores no finitos, fechas inconsistentes y períodos inválidos.

En `examples/` encontrarás una petición JSON y un CSV **sintéticos** con 24 filas para probar el flujo.

## API

| Método y ruta | Función |
| --- | --- |
| `GET /` | Interfaz web |
| `GET /api/health` | Carga/verifica el modelo y devuelve su estado real |
| `GET /api/metadata` | Variables, forma de entrada, métricas y convenciones |
| `GET /api/example` | 24 horas sintéticas listas para probar |
| `GET /api/template.csv` | Plantilla sencilla con mediciones vacías |
| `GET /api/template.csv?layout=full` | Plantilla con las 16 variables explícitas |
| `POST /api/predict` | Predicción real del modelo entrenado |
| `GET /docs` | Swagger para explorar y probar la API |
| `GET /redoc` | Documentación alternativa |

Formato de `POST /api/predict`:

```json
{
  "source": "manual",
  "observations": [
    {
      "timestamp": "2026-10-07T00:00:00-05:00",
      "demanda_mw": 650,
      "temperatura_c": 19,
      "humedad_pct": 78,
      "viento_kmh": 11,
      "radiacion_wm2": 0,
      "precipitacion_mm": 1,
      "precio_kwh": 25,
      "hora": 0,
      "dia_semana": 2,
      "fin_semana": 0,
      "festivo": 0,
      "mes": 10,
      "periodo_dia_madrugada": 1,
      "periodo_dia_manana": 0,
      "periodo_dia_tarde": 0,
      "periodo_dia_noche": 0
    }
  ]
}
```

El bloque anterior muestra **una observación** para explicar sus campos. La petición real requiere 24; usa el archivo completo `examples/peticion_demo.json`. `source` puede ser `manual`, `csv` o `demo`; si se omite, vale `manual`.

Prueba rápida contra el servidor iniciado:

```powershell
.\.venv\Scripts\python.exe .\scripts\predict_example.py
```

Para comprobar las rutas y hacer una predicción desde PowerShell, abre **otra terminal** mientras el servidor sigue ejecutándose:

```powershell
$apiBaseUrl = 'http://127.0.0.1:8000'

# Estado real del modelo (la primera carga puede tardar).
Invoke-RestMethod -Uri "$apiBaseUrl/api/health" -TimeoutSec 120 | ConvertTo-Json

# Variables y métricas guardadas.
Invoke-RestMethod -Uri "$apiBaseUrl/api/metadata" -TimeoutSec 120 | ConvertTo-Json -Depth 10

# Obtener 24 observaciones sintéticas y enviarlas al modelo.
$apiExample = Invoke-RestMethod -Uri "$apiBaseUrl/api/example" -TimeoutSec 120
$apiBody = $apiExample | ConvertTo-Json -Depth 10
Invoke-RestMethod -Uri "$apiBaseUrl/api/predict" -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($apiBody)) -TimeoutSec 120 | ConvertTo-Json -Depth 10
```

O envía el JSON completo de ejemplo incluido en el proyecto:

```powershell
$taskBody = Get-Content -Raw -Encoding UTF8 .\examples\peticion_demo.json
Invoke-RestMethod -Uri "$apiBaseUrl/api/predict" -Method Post -ContentType 'application/json; charset=utf-8' -Body ([System.Text.Encoding]::UTF8.GetBytes($taskBody)) -TimeoutSec 120 | ConvertTo-Json -Depth 10
```

También puedes usar `curl.exe`, incluso en Windows PowerShell donde `curl` puede ser un alias de otro comando:

```powershell
curl.exe --fail-with-body http://127.0.0.1:8000/api/health
curl.exe --fail-with-body -X POST http://127.0.0.1:8000/api/predict -H 'Content-Type: application/json' --data-binary '@examples/peticion_demo.json'
```

Después de publicar, cambia `$apiBaseUrl` por tu dominio de Vercel, por ejemplo `https://tu-proyecto.vercel.app`, y ejecuta las mismas peticiones. Para el script Python también puedes pasar el dominio:

```powershell
.\.venv\Scripts\python.exe .\scripts\predict_example.py https://tu-proyecto.vercel.app
```

En el navegador, usa `/docs` para probar `POST /api/predict` con **Try it out** y el contenido completo de `examples/peticion_demo.json`.

La respuesta incluye `predicted_demand_mw`, `prediction_timestamp`, `next_hour`, `last_demand_mw`, `change_mw`, `change_pct`, `model`, `source`, `warnings` e `inference_ms`. Si la última demanda es cero, `change_pct` es `null`. Si la entrada no incluye fechas, `prediction_timestamp` es `null`. HTTP **422** indica datos inválidos; HTTP **503** indica modelo no disponible. El tiempo de inferencia no incluye la carga inicial de TensorFlow ni la latencia de red.

`POST /api/predict` conserva su contrato de 24 observaciones completas con 16 variables cada una. El frontend expande la plantilla sencilla antes de llamar a la API. `GET /api/metadata` incluye `price_training_reference` con el promedio y la desviación estándar del precio, leídos del escalador original.

## Desplegar en Vercel desde GitHub

La configuración utiliza el soporte nativo de [FastAPI en Vercel](https://vercel.com/docs/frameworks/backend/fastapi). `app.py` es el punto de entrada; `frontend/` contiene los recursos montados en `/assets`. El frontend llama a `/api/...` en el mismo dominio, por lo que no necesitas configurar una URL de backend ni CORS para este despliegue integrado.

1. Crea un repositorio en **GitHub** y sube el proyecto, incluyendo **`production_model.keras`, `preprocessing.pkl`, `metadata.json`, `app.py`, `backend/`, `frontend/`, `requirements.txt`, `.python-version` y `vercel.json`**. Excluye `.venv` y `.env` (ya están en los archivos de exclusión).
2. En el panel web de Vercel, elige **Add New → Project → Import Git Repository**, conecta tu cuenta de **GitHub**, autoriza el acceso al repositorio y pulsa **Import** sobre él.
3. Usa la **raíz de este proyecto** como Root Directory y el preset **FastAPI**. Deja Build Command y Output Directory con sus valores automáticos; no configures una carpeta de frontend separada.
4. Se solicita Python **3.12** mediante `.python-version`. Vercel instalará `requirements.txt`.
5. Activa **Fluid Compute** y agrega la variable **`VERCEL_SUPPORT_LARGE_FUNCTIONS=1`** para Production y Preview. TensorFlow y sus dependencias pueden superar el límite estándar de 500 MB para Python. Vercel documenta [Large Functions hasta 5 GB, en beta](https://vercel.com/changelog/vercel-functions-can-now-be-up-to-5-gb-in-package-size); su disponibilidad depende del proyecto. Si no está habilitada para tu proyecto, necesitarás alojar el backend en un servicio compatible con contenedores y configurar el frontend para esa API.
6. Pulsa **Deploy**. Abre el dominio publicado, espera **Modelo conectado**, carga el ejemplo y ejecuta una predicción. Revisa también `/api/health` y `/docs`.

### Subir el código a GitHub con Git

Si todavía no has inicializado el repositorio local, crea primero un repositorio vacío en GitHub y ejecuta desde la carpeta del proyecto:

```powershell
git init
git add .
git commit -m "Crear aplicación de predicción de demanda energética"
git branch -M main
git remote add origin https://github.com/TU_USUARIO/TU_REPOSITORIO.git
git push -u origin main
```

Reemplaza `TU_USUARIO` y `TU_REPOSITORIO` por los tuyos. Los comandos anteriores son para un repositorio nuevo; si ya tienes Git configurado, conserva tu rama y remoto existentes. También puedes usar GitHub Desktop para subir el proyecto.

Después de conectar el repositorio en Vercel, los nuevos commits enviados a la rama de producción generan un nuevo despliegue. Los cambios de configuración o variables de entorno pueden requerir **Redeploy** desde el panel. Todo el despliegue se administra desde la integración GitHub–Vercel.

La publicación requiere tu cuenta y la disponibilidad de Large Functions; **los archivos de configuración no crean ni publican un proyecto por sí solos**. La verificación local no reemplaza una prueba del dominio publicado. Se configuró una duración máxima de 120 segundos por función para permitir la primera carga del modelo.

## Estructura

```text
app.py                       # FastAPI y rutas públicas
backend/
  config.py                  # Rutas, variables y metadatos
  schemas.py                 # Validación de entradas y respuesta
  inference.py               # Modelo original y escaladores originales
  examples.py                # Ejemplo sintético y plantilla vacía
frontend/
  index.html                 # Interfaz en español
  styles.css                 # Diseño adaptable, sin fuentes externas
  app.mjs                    # Edición, API, gráfico y descarga
  data.mjs                   # Calendario, validación y parser CSV
  favicon.svg
examples/                    # JSON y CSV de prueba sintéticos
scripts/                     # Inicio y prueba contra servidor real
tests/                       # Pruebas de API, inferencia y CSV
production_model.keras       # Artefacto original: LSTM → Dense → Dense
preprocessing.pkl            # Artefacto original: escaladores X e Y
metadata.json                # Artefacto original: variables y métricas
requirements.txt
requirements-dev.txt
.python-version
vercel.json
.env.example
```

La API forma un tensor **(1, 24, 16)** después de aplicar `feature_scaler.transform`. Carga Keras con `compile=False` y `safe_mode=True`, ejecuta con `training=False` y aplica `target_scaler.inverse_transform` a la salida. El modelo se carga una vez por proceso de forma diferida. Se mantiene **scikit-learn 1.6.1**, la versión registrada por el archivo de preprocesamiento, y **Keras 3.13.2**, la versión que guardó el modelo. Los artefactos originales se conservan.

No se necesita base de datos. El historial permanece en la memoria de la pestaña y se pierde al recargar; descárgalo junto con la predicción si deseas conservarlo. La aplicación recibe CSV de mediciones; no permite subir modelos o archivos pickle.

## Verificación

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-dev.txt
.\.venv\Scripts\python.exe -m pytest -q
node --test tests/frontend.test.mjs
```

Node.js 22+ solo es necesario para estas pruebas del parser CSV. Las pruebas Python ejecutan el modelo real y comparan la API contra inferencia directa con los escaladores guardados; también cubren datos inválidos, cronología, cruces de fecha, predicciones repetibles y errores del servidor.

Verificación visual opcional con Chrome instalado y el servidor iniciado:

```powershell
.\.venv\Scripts\python.exe -m pip install selenium==4.40.0
.\.venv\Scripts\python.exe .\scripts\verify_ui.py
```

Usa un perfil de prueba independiente, comprueba escritorio y un viewport móvil de 390 px, y guarda capturas en `test-results/`.

La verificación visual comprueba también la importación sencilla, el aviso de precio y el contraste de los textos visibles sobre fondos sólidos, con una referencia mínima de 4,5:1 según [W3C: contraste mínimo](https://www.w3.org/WAI/WCAG21/Understanding/contrast-minimum.html). Esa comprobación concreta no equivale a una auditoría completa de accesibilidad.

Las métricas existentes del modelo son MAE ≈ **23,04 MW**, RMSE ≈ **37,67 MW**, MAPE ≈ **4,17 %** y R² ≈ **0,81**. Provienen de `metadata.json`; no constituyen un intervalo de confianza de cada resultado ni una evaluación nueva sobre datos reales.
#   T a l l e r L S T M  
 
