"""
sheets_connector.py
-------------------
Conecta con Google Sheets usando la API de Google.

CONFIGURACIÓN INICIAL (solo se hace una vez):
1. Ve a https://console.cloud.google.com/
2. Crea un proyecto nuevo
3. Activa la API "Google Sheets API"
4. Crea una "Cuenta de servicio" (Service Account)
5. Descarga el archivo de credenciales JSON → guárdalo como 'credentials.json'
6. Crea un Google Sheets nuevo y copia el ID de la URL
7. Comparte ese Sheets con el email de la cuenta de servicio (con permiso de edición)
8. Pon el ID en la variable SPREADSHEET_ID abajo
"""

import os
import json
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN — edita estos valores
# ─────────────────────────────────────────────────────────────────────────────

# ID del Google Sheets (lo encuentras en la URL: docs.google.com/spreadsheets/d/ESTE_ID/edit)
SPREADSHEET_ID = os.environ.get('SPREADSHEET_ID', '15erGbl2O6j7SGORYl2FbOO2wVV_r9vFzE4c304y-PUs')

# Nombre de la hoja dentro del Sheets
SHEET_NAME = 'Respuestas'

# Ruta al archivo de credenciales JSON descargado de Google Cloud
CREDENTIALS_FILE = os.environ.get('GOOGLE_CREDENTIALS_FILE', 'credentials.json')

# Si tienes las credenciales como variable de entorno (para despliegue en la nube)
CREDENTIALS_JSON = os.environ.get('GOOGLE_CREDENTIALS_JSON', None)

# ─────────────────────────────────────────────────────────────────────────────


def _get_sheets_client():
    """Inicializa y retorna el cliente de Google Sheets."""
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        raise ImportError(
            "Faltan librerías. Ejecuta: pip install gspread google-auth"
        )

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]

    # Credenciales desde variable de entorno (preferido para producción)
    if CREDENTIALS_JSON:
        creds_dict = json.loads(CREDENTIALS_JSON)
        creds = Credentials.from_service_account_info(creds_dict, scopes=scopes)
    elif os.path.exists(CREDENTIALS_FILE):
        creds = Credentials.from_service_account_file(CREDENTIALS_FILE, scopes=scopes)
    else:
        raise FileNotFoundError(
            f"No se encontró el archivo de credenciales '{CREDENTIALS_FILE}'. "
            "Sigue las instrucciones en sheets_connector.py para configurarlo."
        )

    client = gspread.authorize(creds)
    return client


def _ensure_header(worksheet):
    """Verifica que la hoja tenga encabezado. Si no, lo crea."""
    existing = worksheet.row_values(1)
    if existing:
        return  # Ya tiene encabezado

    # Crear encabezado
    header = ['Fecha', 'Hora', 'Nombre', 'ID_Estudiante']
    header += [f'P{i}' for i in range(1, 126)]  # P1, P2, ... P125
    header += ['Total_Respondidas']
    worksheet.append_row(header, value_input_option='RAW')

    # Formatear encabezado (negrita, fondo azul)
    try:
        worksheet.format('A1:EF1', {
            'backgroundColor': {'red': 0.2, 'green': 0.4, 'blue': 0.8},
            'textFormat': {'bold': True, 'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}},
            'horizontalAlignment': 'CENTER'
        })
    except Exception:
        pass  # El formato es opcional, no romper si falla


def save_to_sheets(student_name: str, exam_id: str, answers: list) -> dict:
    """
    Guarda las respuestas de un examen en Google Sheets.

    Args:
        student_name: Nombre del estudiante
        exam_id: ID/cédula del estudiante
        answers: Lista de 125 strings ('A','B','C','D' o '?')

    Returns:
        dict con 'row' (número de fila insertada) y 'url' del sheets
    """
    client = _get_sheets_client()

    try:
        spreadsheet = client.open_by_key(SPREADSHEET_ID)
    except Exception:
        raise ValueError(
            f"No se pudo abrir el Sheets con ID '{SPREADSHEET_ID}'. "
            "Verifica el ID y que hayas compartido el Sheets con la cuenta de servicio."
        )

    # Obtener o crear hoja 'Respuestas'
    try:
        worksheet = spreadsheet.worksheet(SHEET_NAME)
    except Exception:
        worksheet = spreadsheet.add_worksheet(title=SHEET_NAME, rows=1000, cols=130)

    _ensure_header(worksheet)

    # Construir fila
    now = datetime.now()
    total_answered = len([a for a in answers if a != '?'])

    row_data = [
        now.strftime('%Y-%m-%d'),        # Fecha
        now.strftime('%H:%M:%S'),        # Hora
        student_name,                     # Nombre
        exam_id,                          # ID
    ]
    row_data += answers                   # P1..P125
    row_data += [total_answered]          # Total respondidas

    # Insertar fila
    worksheet.append_row(row_data, value_input_option='RAW')

    # Número de la fila insertada
    row_num = len(worksheet.col_values(1))

    return {
        'row': row_num,
        'url': f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit'
    }


def get_sheet_data() -> list:
    """
    Retorna todos los datos del Sheets como lista de diccionarios.
    Útil para mostrar resultados en la app.
    """
    client = _get_sheets_client()
    spreadsheet = client.open_by_key(SPREADSHEET_ID)

    try:
        worksheet = spreadsheet.worksheet(SHEET_NAME)
        records = worksheet.get_all_records()
        return records
    except Exception as e:
        return []


# ─────────────────────────────────────────────────────────────────────────────
# MODO OFFLINE (para pruebas sin credenciales de Google)
# ─────────────────────────────────────────────────────────────────────────────

def save_to_csv_fallback(student_name: str, exam_id: str, answers: list) -> dict:
    """
    Alternativa local: guarda en CSV si no hay Google Sheets configurado.
    Útil para pruebas.
    """
    import csv
    filepath = 'resultados_local.csv'
    now = datetime.now()
    file_exists = os.path.exists(filepath)

    with open(filepath, 'a', newline='', encoding='utf-8') as f:
        writer = csv.writer(f)
        if not file_exists:
            header = ['Fecha', 'Hora', 'Nombre', 'ID']
            header += [f'P{i}' for i in range(1, 126)]
            header += ['Total']
            writer.writerow(header)

        row = [now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'), student_name, exam_id]
        row += answers
        row += [len([a for a in answers if a != '?'])]
        writer.writerow(row)

    return {'row': None, 'url': filepath}
