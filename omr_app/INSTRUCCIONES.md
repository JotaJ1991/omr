# 📋 Guía de Instalación y Uso — Calificador OMR

## ¿Qué hace esta app?
Toma fotos de hojas de respuestas de 125 preguntas, detecta automáticamente
cuál opción (A/B/C/D) fue marcada en cada pregunta, y guarda los resultados
en Google Sheets para análisis estadístico.

---

## 📁 Estructura de archivos

```
omr_app/
├── app.py               ← Servidor web principal (Flask)
├── omr_processor.py     ← Motor de reconocimiento de burbujas (OMR)
├── sheets_connector.py  ← Integración con Google Sheets
├── calibrar.py          ← Herramienta de calibración
├── requirements.txt     ← Librerías necesarias
├── credentials.json     ← (TÚ LO CREAS) Credenciales de Google
├── templates/
│   └── index.html       ← Interfaz web del celular
└── uploads/             ← Carpeta temporal de imágenes
```

---

## 🚀 INSTALACIÓN EN TU COMPUTADOR (Para empezar a probar)

### Paso 1: Instalar Python
Si no lo tienes: https://www.python.org/downloads/
Descarga Python 3.10 o superior.

### Paso 2: Instalar las librerías
Abre la terminal (cmd en Windows / terminal en Mac/Linux) dentro de la carpeta del proyecto:

```bash
pip install -r requirements.txt
```

### Paso 3: Configurar Google Sheets (ver sección abajo)

### Paso 4: Iniciar el servidor
```bash
python app.py
```

Verás algo como:
```
 * Running on http://0.0.0.0:5000
```

### Paso 5: Abrir desde el celular
Tu computador y celular deben estar en el **mismo WiFi**.
En el celular, abre el navegador y escribe:
```
http://[IP-DE-TU-PC]:5000
```
Para ver la IP de tu PC:
- Windows: `ipconfig` en cmd → busca "IPv4 Address"
- Mac/Linux: `ifconfig` → busca "inet"

---

## 🔑 CONFIGURAR GOOGLE SHEETS

### Paso 1: Crear proyecto en Google Cloud
1. Ve a: https://console.cloud.google.com/
2. Clic en "Crear proyecto" → dale un nombre (ej: "CalificadorOMR")
3. Selecciona el proyecto recién creado

### Paso 2: Activar APIs
1. Menú lateral → "APIs y servicios" → "Habilitar APIs"
2. Busca y habilita: **"Google Sheets API"**
3. Busca y habilita: **"Google Drive API"**

### Paso 3: Crear cuenta de servicio
1. Menú → "APIs y servicios" → "Credenciales"
2. Clic en "Crear credenciales" → "Cuenta de servicio"
3. Dale un nombre (ej: "omr-bot")
4. En "Rol" selecciona "Editor"
5. Clic en la cuenta creada → pestaña "Claves"
6. "Agregar clave" → "Crear nueva clave" → JSON
7. Se descarga un archivo JSON → **cópialo a la carpeta del proyecto** y renómbralo `credentials.json`

### Paso 4: Crear el Google Sheets
1. Ve a https://sheets.google.com
2. Crea una hoja nueva (cualquier nombre)
3. Copia el **ID** de la URL:
   `https://docs.google.com/spreadsheets/d/`**`ESTE_ES_EL_ID`**`/edit`

### Paso 5: Compartir el Sheets con la cuenta de servicio
1. Abre tu `credentials.json` y copia el valor de `"client_email"`
   (tiene formato: `algo@tu-proyecto.iam.gserviceaccount.com`)
2. En el Sheets, clic en "Compartir" → pega ese email → "Editor"

### Paso 6: Configurar el ID en la app
Abre `sheets_connector.py` y cambia esta línea:
```python
SPREADSHEET_ID = 'TU_ID_DE_SHEETS_AQUÍ'
```
por el ID que copiaste en el Paso 4.

---

## ☁️ DESPLIEGUE EN LA NUBE (Para acceso desde cualquier lugar)

### Opción A: Render.com (Gratis, recomendado)

1. Crea cuenta en https://render.com
2. Crea nuevo "Web Service" → conecta con tu repositorio de GitHub
   (sube los archivos a GitHub primero)
3. Configuración:
   - Build Command: `pip install -r requirements.txt`
   - Start Command: `python app.py`
4. Variables de entorno (en el panel de Render):
   - `SPREADSHEET_ID` = tu ID del Sheets
   - `GOOGLE_CREDENTIALS_JSON` = contenido completo del credentials.json
     (copia y pega todo el JSON como una sola línea)
5. Render te da una URL pública tipo `https://tu-app.onrender.com`

### Opción B: Railway.app (También gratis)
Similar a Render, busca "Deploy from GitHub" en railway.app

---

## 🎯 CALIBRACIÓN (¡IMPORTANTE!)

La primera vez que uses la app con hojas impresas reales, debes calibrar.

### Por qué es necesario
El OMR necesita saber exactamente dónde están las burbujas en la imagen.
Esto depende de TU impresora, el tamaño de la foto, etc.

### Cómo calibrar
1. Imprime una hoja de respuestas en blanco (sin marcar)
2. Tómale una foto con el celular
3. Copia la foto a la carpeta del proyecto
4. Ejecuta:
   ```bash
   python calibrar.py mi_foto.jpg
   ```
5. Revisa la imagen `mi_foto_calibracion.jpg` generada
6. Si la grilla azul/verde no coincide con las burbujas,
   ajusta los valores en `omr_processor.py`:

```python
GRID_TOP    = 0.285   # Sube si la grilla empieza muy abajo
GRID_BOTTOM = 0.975   # Baja si se pasa del último renglón
GRID_LEFT   = 0.030   # Mueve a la derecha si las celdas quedan a la izq
GRID_RIGHT  = 0.975   # Mueve a la izquierda si se pasan a la derecha
```

### Ajuste del umbral de detección
Si detecta burbujas no marcadas como marcadas (falsos positivos):
```python
FILL_THRESHOLD = 0.25  # Súbelo (era 0.18)
```
Si no detecta burbujas que sí están marcadas (falsos negativos):
```python
FILL_THRESHOLD = 0.12  # Bájalo (era 0.18)
```

---

## 📊 ANÁLISIS EN GOOGLE SHEETS

Cada examen procesado agrega una fila con:
| Columna | Contenido |
|---------|-----------|
| A | Fecha |
| B | Hora |
| C | Nombre del estudiante |
| D | ID / Cédula |
| E-EF | P1 a P125 (respuestas: A,B,C,D o ?) |
| EG | Total respondidas |

### Fórmulas útiles para análisis
```
=COUNTIF(E2:EF2,"A")     → Cuántas A marcó este estudiante
=COUNTIF(E:E,"B")         → Cuántos marcaron B en la pregunta 1
=AVERAGE(EG:EG)           → Promedio de preguntas respondidas
```

Para comparar con clave de respuestas, puedes crear una segunda hoja
"Clave" con las respuestas correctas y usar COUNTIFS para calcular puntajes.

---

## 🐛 Solución de problemas

**"No se detectaron marcadores"**
→ La foto está muy oscura, muy torcida (>30°), o los marcadores están cortados.
→ Prueba con mejor iluminación y encuadrando toda la hoja.

**Respuestas incorrectas sistemáticamente**
→ Necesitas calibrar. Sigue la sección de Calibración.

**Error de Google Sheets**
→ Verifica que el email de la cuenta de servicio tenga acceso al Sheets.
→ Verifica que el SPREADSHEET_ID sea correcto.

**La app abre pero no procesa**
→ Revisa la terminal donde corriste `python app.py` para ver el error.
