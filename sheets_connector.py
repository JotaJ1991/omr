"""
sheets_connector.py  —  v3  (clave de respuestas + conteo horizontal y vertical)
---------------------------------------------------------------------------------
Estructura de cada pestaña:
  Fila 1  — Encabezado de columnas
  Fila 2  — CLAVE (respuestas correctas) — nombre fijo "*** CLAVE ***"
  Fila 3+ — Estudiantes
  Última  — TOTALES por pregunta (% aciertos vertical)

Columnas:
  A  Fecha
  B  Hora
  C  Nombre
  D  ID_Estudiante
  E…EX  P1…P125  (respuesta del estudiante, o — si no detectada)
  EY  Respondidas   (total burbujas detectadas)
  EZ  Correctas     (vs clave, solo si clave existe)
  FA  Porcentaje    (Correctas / total preguntas activas × 100)
"""

import os
import json
from datetime import datetime

SPREADSHEET_ID   = os.environ.get('SPREADSHEET_ID', 'TU_ID_DE_SHEETS_AQUÍ')
CREDENTIALS_FILE = os.environ.get('GOOGLE_CREDENTIALS_FILE', 'credentials.json')
CREDENTIALS_JSON = os.environ.get('GOOGLE_CREDENTIALS_JSON', None)

# Columna donde empieza P1 (base-0: col 4 = columna E)
COL_OFFSET = 4
KEY_ROW_NAME = '*** CLAVE ***'


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
            "Verifica el ID y los permisos.")


def _build_header(n=125):
    """Fila 1: encabezados."""
    h = ['Fecha', 'Hora', 'Nombre', 'ID_Estudiante']
    h += [f'P{i}' for i in range(1, n + 1)]
    h += ['Respondidas', 'Correctas', 'Porcentaje']
    return h


def _ensure_structure(worksheet, n_questions=125):
    """
    Crea encabezado (fila 1) y fila de CLAVE (fila 2) si la hoja está vacía.
    Si ya existe no la toca.
    """
    all_rows = worksheet.get_all_values()

    if not all_rows:
        # Hoja vacía: crear encabezado + clave vacía
        header = _build_header(n_questions)
        worksheet.append_row(header, value_input_option='RAW')

        key_row = [datetime.now().strftime('%Y-%m-%d'),
                   datetime.now().strftime('%H:%M:%S'),
                   KEY_ROW_NAME, '']
        key_row += [''] * n_questions        # respuestas correctas vacías
        key_row += ['', '', '']              # Respondidas / Correctas / %
        worksheet.append_row(key_row, value_input_option='RAW')

        # Formato encabezado — azul oscuro
        end_col = _col_letter(COL_OFFSET + n_questions + 3)
        try:
            worksheet.format(f'A1:{end_col}1', {
                'backgroundColor': {'red': 0.13, 'green': 0.27, 'blue': 0.53},
                'textFormat': {'bold': True,
                               'foregroundColor': {'red':1,'green':1,'blue':1}},
                'horizontalAlignment': 'CENTER'
            })
            # Formato fila clave — verde oscuro
            worksheet.format(f'A2:{end_col}2', {
                'backgroundColor': {'red': 0.12, 'green': 0.39, 'blue': 0.19},
                'textFormat': {'bold': True,
                               'foregroundColor': {'red':1,'green':1,'blue':1}},
                'horizontalAlignment': 'CENTER'
            })
        except Exception:
            pass


def _col_letter(idx):
    """Convierte índice base-0 a letra(s) de columna (A, B, … Z, AA, AB…)."""
    result = ''
    idx += 1
    while idx:
        idx, rem = divmod(idx - 1, 26)
        result = chr(65 + rem) + result
    return result


def _get_answer_key(worksheet) -> list:
    """
    Devuelve la lista de respuestas correctas desde la fila 2.
    Retorna lista vacía si la fila de clave está vacía.
    """
    try:
        row2 = worksheet.row_values(2)
        if not row2 or row2[2] != KEY_ROW_NAME:
            return []
        answers = row2[COL_OFFSET:]
        # Quitar columnas de resumen al final
        while answers and answers[-1] in ('', 'Respondidas', 'Correctas', 'Porcentaje'):
            answers.pop()
        return answers
    except Exception:
        return []


def _count_correct(student_answers: list, key: list) -> int:
    """Cuenta cuántas respuestas coinciden con la clave."""
    if not key:
        return 0
    return sum(
        1 for i, ans in enumerate(student_answers)
        if i < len(key) and key[i] not in ('', '?') and ans == key[i]
    )


def _update_totals_row(worksheet, n_questions):
    """
    Actualiza (o crea) la fila de TOTALES al final de la hoja.
    Calcula % de aciertos por pregunta (columna vertical).
    Solo cuenta filas de estudiantes reales (excluye CLAVE y TOTALES).
    """
    try:
        all_rows = worksheet.get_all_values()
        if len(all_rows) < 3:
            return  # solo encabezado + clave, nada que totalizar

        # Filas de estudiantes: desde fila índice 2 en adelante,
        # excluyendo la última si ya es la fila de totales
        student_rows = []
        for r in all_rows[2:]:
            if r and r[2] not in (KEY_ROW_NAME, '--- TOTALES ---'):
                student_rows.append(r)

        if not student_rows:
            return

        key = _get_answer_key(worksheet)

        totals_label = ['', '', '--- TOTALES ---', '']
        col_totals = []
        for q in range(n_questions):
            col_idx = COL_OFFSET + q
            answers_for_q = [r[col_idx] if col_idx < len(r) else ''
                             for r in student_rows]
            total_students = len(answers_for_q)
            if key and q < len(key) and key[q] not in ('', '?'):
                correct = sum(1 for a in answers_for_q if a == key[q])
                pct = round(correct / total_students * 100) if total_students else 0
                col_totals.append(f'{correct}/{total_students} ({pct}%)')
            else:
                detected = sum(1 for a in answers_for_q if a not in ('', '?', '—'))
                col_totals.append(f'{detected}/{total_students}')

        totals_label += col_totals
        totals_label += ['', '', '']

        # Buscar si ya existe fila de totales (última fila)
        last_row_idx = len(all_rows)
        last_row = all_rows[-1] if all_rows else []
        if last_row and last_row[2] == '--- TOTALES ---':
            # Actualizar en lugar de agregar
            worksheet.update(f'A{last_row_idx}', [totals_label],
                             value_input_option='RAW')
        else:
            worksheet.append_row(totals_label, value_input_option='RAW')
            last_row_idx = len(worksheet.col_values(1))

        # Formato fila totales — gris oscuro
        end_col = _col_letter(COL_OFFSET + n_questions + 3)
        try:
            worksheet.format(f'A{last_row_idx}:{end_col}{last_row_idx}', {
                'backgroundColor': {'red': 0.25, 'green': 0.25, 'blue': 0.25},
                'textFormat': {'bold': True,
                               'foregroundColor': {'red':1,'green':1,'blue':1}},
                'horizontalAlignment': 'CENTER'
            })
        except Exception:
            pass
    except Exception:
        pass  # no interrumpir el guardado si falla el resumen


# ─────────────────────────────────────────────────────────────────────────────
# API PUBLICA
# ─────────────────────────────────────────────────────────────────────────────

def list_sheets() -> list:
    spreadsheet = _open_spreadsheet()
    return [ws.title for ws in spreadsheet.worksheets()]


def create_sheet(name: str) -> dict:
    name = name.strip()
    if not name:
        return {'success': False, 'error': 'El nombre no puede estar vacío.'}

    spreadsheet = _open_spreadsheet()
    existing = [ws.title for ws in spreadsheet.worksheets()]
    if name in existing:
        return {'success': False, 'error': f'La hoja "{name}" ya existe.'}

    try:
        ws = spreadsheet.add_worksheet(title=name, rows=1000, cols=135)
        _ensure_structure(ws)
        return {'success': True, 'name': name}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def save_to_sheets(student_name: str, exam_id: str,
                   answers: list, sheet_name: str) -> dict:
    """
    Guarda una fila de respuestas, calcula correctas vs clave,
    y actualiza la fila de totales verticales.
    """
    spreadsheet  = _open_spreadsheet()

    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except Exception:
        worksheet = spreadsheet.add_worksheet(
            title=sheet_name, rows=1000, cols=135)

    n_questions = len(answers)
    _ensure_structure(worksheet, n_questions)

    # Obtener clave y calcular correctas
    key       = _get_answer_key(worksheet)
    correct   = _count_correct(answers, key)
    detected  = len([a for a in answers if a not in ('?', '')])
    pct_str   = f'{round(correct / n_questions * 100)}%' if key else ''

    now      = datetime.now()
    row_data = [now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'),
                student_name, exam_id]
    row_data += [a if a != '?' else '—' for a in answers]
    row_data += [detected,
                 correct if key else '',
                 pct_str]

    worksheet.append_row(row_data, value_input_option='RAW')
    row_num = len(worksheet.col_values(1))

    # Actualizar totales verticales
    _update_totals_row(worksheet, n_questions)

    return {
        'row':     row_num,
        'correct': correct if key else None,
        'pct':     pct_str,
        'url':     f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit'
    }


def save_answer_key(key_answers: list, sheet_name: str) -> dict:
    """
    Guarda o actualiza la fila de clave en fila 2.
    key_answers: lista de hasta 125 respuestas ('A','B','C','D' o '')
    """
    spreadsheet = _open_spreadsheet()
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except Exception:
        return {'success': False, 'error': f'Hoja "{sheet_name}" no encontrada.'}

    n = len(key_answers)
    _ensure_structure(worksheet, n)

    now = datetime.now()
    key_row = [now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'),
               KEY_ROW_NAME, '']
    key_row += [a.upper() if a in ('a','b','c','d','A','B','C','D') else ''
                for a in key_answers]
    key_row += ['', '', '']

    # Siempre actualizar fila 2
    end_col = _col_letter(len(key_row) - 1)
    worksheet.update(f'A2:{end_col}2', [key_row], value_input_option='RAW')

    return {'success': True}


def get_sheet_data(sheet_name: str) -> list:
    spreadsheet = _open_spreadsheet()
    try:
        return spreadsheet.worksheet(sheet_name).get_all_records()
    except Exception:
        return []
