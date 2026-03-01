"""
sheets_connector.py  —  v2  (soporte multi-hoja)
-------------------------------------------------
Permite crear/seleccionar pestañas dentro del Spreadsheet.
La hoja activa se pasa como parámetro en cada llamada.
"""

import os
import json
from datetime import datetime

# ─────────────────────────────────────────────────────────────────────────────
# CONFIGURACIÓN
# ─────────────────────────────────────────────────────────────────────────────
SPREADSHEET_ID   = os.environ.get('SPREADSHEET_ID', 'TU_ID_DE_SHEETS_AQUÍ')
CREDENTIALS_FILE = os.environ.get('GOOGLE_CREDENTIALS_FILE', 'credentials.json')
CREDENTIALS_JSON = os.environ.get('GOOGLE_CREDENTIALS_JSON', None)
# ─────────────────────────────────────────────────────────────────────────────


def _get_sheets_client():
    try:
        import gspread
        from google.oauth2.service_account import Credentials
    except ImportError:
        raise ImportError("Ejecuta: pip install gspread google-auth")

    scopes = [
        'https://www.googleapis.com/auth/spreadsheets',
        'https://www.googleapis.com/auth/drive'
    ]
    if CREDENTIALS_JSON:
        creds = Credentials.from_service_account_info(
            json.loads(CREDENTIALS_JSON), scopes=scopes)
    elif os.path.exists(CREDENTIALS_FILE):
        creds = Credentials.from_service_account_file(
            CREDENTIALS_FILE, scopes=scopes)
    else:
        raise FileNotFoundError(
            f"No se encontró '{CREDENTIALS_FILE}'. "
            "Configura las credenciales de Google.")

    return gspread.authorize(creds)


def _open_spreadsheet():
    client = _get_sheets_client()
    try:
        return client.open_by_key(SPREADSHEET_ID)
    except Exception:
        raise ValueError(
            f"No se pudo abrir el Sheets '{SPREADSHEET_ID}'. "
            "Verifica el ID y los permisos de la cuenta de servicio.")


def _ensure_header(worksheet):
    """Crea encabezado si la hoja está vacía."""
    if worksheet.row_values(1):
        return
    header = ['Fecha', 'Hora', 'Nombre', 'ID_Estudiante']
    header += [f'P{i}' for i in range(1, 126)]
    header += ['Total_Respondidas']
    worksheet.append_row(header, value_input_option='RAW')
    try:
        worksheet.format('A1:EF1', {
            'backgroundColor': {'red': 0.2, 'green': 0.4, 'blue': 0.8},
            'textFormat': {
                'bold': True,
                'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}
            },
            'horizontalAlignment': 'CENTER'
        })
    except Exception:
        pass


# ─────────────────────────────────────────────────────────────────────────────
# API MULTI-HOJA
# ─────────────────────────────────────────────────────────────────────────────

def list_sheets() -> list:
    """
    Retorna la lista de nombres de todas las pestañas del Spreadsheet.
    """
    spreadsheet = _open_spreadsheet()
    return [ws.title for ws in spreadsheet.worksheets()]


def create_sheet(name: str) -> dict:
    """
    Crea una nueva pestaña con el nombre dado.
    Si ya existe, retorna error.
    """
    name = name.strip()
    if not name:
        return {'success': False, 'error': 'El nombre no puede estar vacío.'}

    spreadsheet = _open_spreadsheet()
    existing = [ws.title for ws in spreadsheet.worksheets()]

    if name in existing:
        return {'success': False, 'error': f'La hoja "{name}" ya existe.'}

    try:
        ws = spreadsheet.add_worksheet(title=name, rows=1000, cols=130)
        _ensure_header(ws)
        return {'success': True, 'name': name}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def save_to_sheets(student_name: str, exam_id: str,
                   answers: list, sheet_name: str) -> dict:
    """
    Guarda una fila de respuestas en la pestaña indicada.
    """
    spreadsheet = _open_spreadsheet()

    # Obtener o crear la hoja
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except Exception:
        worksheet = spreadsheet.add_worksheet(
            title=sheet_name, rows=1000, cols=130)

    _ensure_header(worksheet)

    now  = datetime.now()
    total = len([a for a in answers if a != '?'])

    row_data  = [now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'),
                 student_name, exam_id]
    row_data += answers
    row_data += [total]

    worksheet.append_row(row_data, value_input_option='RAW')
    row_num = len(worksheet.col_values(1))

    return {
        'row': row_num,
        'url': f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit'
    }


def get_sheet_data(sheet_name: str) -> list:
    """Retorna todos los registros de una pestaña."""
    spreadsheet = _open_spreadsheet()
    try:
        return spreadsheet.worksheet(sheet_name).get_all_records()
    except Exception:
        return []
