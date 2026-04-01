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
    Si ya existe pero tiene menos columnas de las necesarias, extiende el encabezado.
    Si ya tiene suficientes columnas no la toca.
    """
    all_rows = worksheet.get_all_values()

    if not all_rows:
        # Hoja vacía: crear encabezado + clave vacía
        header = _build_header(n_questions)
        worksheet.append_row(header, value_input_option='RAW')

        key_row = [datetime.now().strftime('%Y-%m-%d'),
                   datetime.now().strftime('%H:%M:%S'),
                   KEY_ROW_NAME, '']
        key_row += [''] * n_questions
        key_row += ['', '', '']
        worksheet.append_row(key_row, value_input_option='RAW')

        end_col = _col_letter(COL_OFFSET + n_questions + 2)
        try:
            worksheet.format(f'A1:{end_col}1', {
                'backgroundColor': {'red': 0.13, 'green': 0.27, 'blue': 0.53},
                'textFormat': {'bold': True,
                               'foregroundColor': {'red':1,'green':1,'blue':1}},
                'horizontalAlignment': 'CENTER'
            })
            worksheet.format(f'A2:{end_col}2', {
                'backgroundColor': {'red': 0.12, 'green': 0.39, 'blue': 0.19},
                'textFormat': {'bold': True,
                               'foregroundColor': {'red':1,'green':1,'blue':1}},
                'horizontalAlignment': 'CENTER'
            })
        except Exception:
            pass
        return

    # Hoja ya existe — verificar que el encabezado cubra n_questions
    header_row = all_rows[0] if all_rows else []
    existing_q_cols = len(header_row) - COL_OFFSET - 3  # descontar fecha/hora/nombre/id + 3 resumen
    if existing_q_cols < n_questions:
        # Extender encabezado con las columnas que faltan
        new_header = _build_header(n_questions)
        end_col = _col_letter(len(new_header) - 1)
        worksheet.update(f'A1:{end_col}1', [new_header], value_input_option='RAW')
        try:
            worksheet.format(f'A1:{end_col}1', {
                'backgroundColor': {'red': 0.13, 'green': 0.27, 'blue': 0.53},
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
    """
    Cuenta correctas solo hasta la última pregunta con clave definida.
    Si la clave tiene ['A','B','','C',''], el último definido es índice 3
    y solo se evalúan las preguntas 1-4.
    """
    if not key:
        return 0
    # Encontrar el último índice con clave definida
    last_defined = -1
    for i in range(len(key) - 1, -1, -1):
        if key[i] not in ('', '?'):
            last_defined = i
            break
    if last_defined == -1:
        return 0
    return sum(
        1 for i, ans in enumerate(student_answers)
        if i <= last_defined and i < len(key)
        and key[i] not in ('', '?') and ans == key[i]
    )


def _key_total(key: list) -> int:
    """Número de preguntas con clave definida (hasta la última)."""
    for i in range(len(key) - 1, -1, -1):
        if key[i] not in ('', '?'):
            return i + 1   # preguntas 1..(i+1)
    return 0


def _update_totals_row(worksheet, n_questions):
    """
    Escribe (o sobreescribe) la fila de TOTALES al final de la hoja.
    Siempre es la última fila — nunca se duplica.
    """
    try:
        all_rows = worksheet.get_all_values()
        if len(all_rows) < 3:
            return

        # Filas de estudiantes reales (excluir encabezado, clave y totales)
        student_rows = [
            r for r in all_rows[2:]
            if r and r[2] not in (KEY_ROW_NAME, '--- TOTALES ---')
        ]
        if not student_rows:
            return

        key   = _get_answer_key(worksheet)
        key_n = _key_total(key)   # hasta qué pregunta hay clave definida

        totals_label = ['', '', '--- TOTALES ---', '']
        col_totals   = []

        for q in range(n_questions):
            col_idx        = COL_OFFSET + q
            answers_for_q  = [r[col_idx] if col_idx < len(r) else ''
                               for r in student_rows]
            total_students = len(answers_for_q)

            if key and q < len(key) and key[q] not in ('', '?'):
                correct = sum(1 for a in answers_for_q if a == key[q])
                pct     = round(correct / total_students * 100) if total_students else 0
                col_totals.append(f'{correct}/{total_students} ({pct}%)')
            elif q < key_n:
                # Dentro del rango de la clave pero sin respuesta definida
                detected = sum(1 for a in answers_for_q if a not in ('', '?', '—'))
                col_totals.append(f'{detected}/{total_students}')
            else:
                detected = sum(1 for a in answers_for_q if a not in ('', '?', '—'))
                col_totals.append(f'{detected}/{total_students}')

        totals_label += col_totals
        totals_label += ['', '', '']

        # Buscar si ya existe fila de totales (debe ser la última)
        last_row_idx = len(all_rows)
        last_row     = all_rows[-1] if all_rows else []

        if last_row and last_row[2] == '--- TOTALES ---':
            worksheet.update(f'A{last_row_idx}', [totals_label],
                             value_input_option='RAW')
        else:
            worksheet.append_row(totals_label, value_input_option='RAW')
            last_row_idx = len(worksheet.col_values(1))

        end_col = _col_letter(COL_OFFSET + n_questions + 2)
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
        pass


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
    y mantiene UNA SOLA fila de totales al final.
    """
    spreadsheet  = _open_spreadsheet()

    try:
        worksheet = spreadsheet.worksheet(sheet_name)
    except Exception:
        worksheet = spreadsheet.add_worksheet(
            title=sheet_name, rows=1000, cols=135)

    n_questions = len(answers)
    _ensure_structure(worksheet, n_questions)

    # Obtener clave
    key      = _get_answer_key(worksheet)
    correct  = _count_correct(answers, key)
    detected = len([a for a in answers if a not in ('?', '')])

    # Porcentaje sobre el total de preguntas con clave definida
    key_n   = _key_total(key)
    pct_str = f'{round(correct / key_n * 100)}%' if key_n > 0 else ''

    # ── Eliminar fila de TOTALES existente antes de agregar el estudiante ─────
    # Esto evita que los totales queden duplicados o en posición incorrecta
    all_rows = worksheet.get_all_values()
    for i, row in enumerate(all_rows):
        if row and len(row) > 2 and row[2] == '--- TOTALES ---':
            worksheet.delete_rows(i + 1)   # gspread usa índice 1-based
            break

    # ── Agregar fila del estudiante ───────────────────────────────────────────
    now      = datetime.now()
    row_data = [now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'),
                student_name, exam_id]
    row_data += [a if a != '?' else '—' for a in answers]
    row_data += [detected,
                 correct if key_n > 0 else '',
                 pct_str]

    worksheet.append_row(row_data, value_input_option='RAW')
    row_num = len(worksheet.col_values(1))

    # ── Agregar TOTALES al final (siempre una sola vez) ───────────────────────
    _update_totals_row(worksheet, n_questions)

    return {
        'row':     row_num,
        'correct': correct if key_n > 0 else None,
        'pct':     pct_str,
        'url':     f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit'
    }


def get_answer_key(sheet_name: str) -> dict:
    """Retorna la clave de respuestas guardada en la fila 2 de la hoja."""
    spreadsheet = _open_spreadsheet()
    try:
        worksheet = spreadsheet.worksheet(sheet_name)
        key = _get_answer_key(worksheet)
        return {'success': True, 'answers': key}
    except Exception as e:
        return {'success': False, 'error': str(e), 'answers': []}


def save_answer_key(key_answers: list, sheet_name: str) -> dict:
    """
    Guarda o actualiza la fila de clave en fila 2.
    key_answers: lista de hasta 200 respuestas (A-H o '')
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
    # Aceptar cualquier letra A-H (mayúscula o minúscula)
    valid = set('ABCDEFGHabcdefgh')
    key_row += [a.upper() if a in valid else ''
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


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADOS SIPAGRE — combina 1S y 2S por ID de estudiante
# ─────────────────────────────────────────────────────────────────────────────

# Definición de materias: (nombre, [(sesion, p_inicio_0based, p_fin_0based)], puntos)
SIPAGRE_SUBJECTS = [
    ('Matematica',       [('1S', 0, 24),  ('2S', 25, 49)],  2.0),
    ('Lectura Critica',  [('1S', 25, 65)],                   2.4),
    ('Sociales',         [('1S', 66, 90), ('2S', 0, 24)],    2.0),
    ('Naturales',        [('1S', 91, 119),('2S', 50, 78)],   1.7),
    ('Ingles',           [('2S', 79, 133)],                   1.8),
]


def _count_correct_range(answers, key, start, end):
    """Cuenta respuestas correctas en rango [start, end] (0-based, inclusivo)."""
    count = 0
    for i in range(start, min(end + 1, len(answers), len(key))):
        a = answers[i].strip() if i < len(answers) else ''
        k = key[i].strip() if i < len(key) else ''
        if k not in ('', '?', '\u2014') and a == k:
            count += 1
    return count


def _extract_students(worksheet):
    """
    Extrae dict {id_estudiante: {'name': str, 'answers': list}} de una hoja.
    Salta fila 1 (encabezado), fila 2 (clave) y fila TOTALES.
    """
    all_rows = worksheet.get_all_values()
    students = {}
    for row in all_rows[2:]:
        if not row or len(row) < 5:
            continue
        name = row[2].strip() if len(row) > 2 else ''
        sid  = row[3].strip() if len(row) > 3 else ''
        if not sid or name in (KEY_ROW_NAME, '--- TOTALES ---'):
            continue
        answers = row[COL_OFFSET:]
        # Limpiar: quitar columnas de resumen al final si las tiene
        # (buscar donde empiezan las no-letra)
        clean = []
        for a in answers:
            a = a.strip()
            if a in ('', '?', '\u2014', '-'):
                clean.append('')
            elif len(a) == 1 and a.upper() in 'ABCDEFGH':
                clean.append(a.upper())
            else:
                break  # llegamos a Respondidas/Correctas/Porcentaje
        students[sid] = {'name': name, 'answers': clean}
    return students


def generate_sipagre_results(sheet_1s: str, sheet_2s: str,
                             results_sheet: str = 'Resultados SIPAGRE') -> dict:
    """
    Combina las sesiones 1S y 2S SIPAGRE por ID de estudiante.
    Calcula puntajes por materia y puntaje general.
    Escribe los resultados en una hoja nueva/existente.
    """
    spreadsheet = _open_spreadsheet()

    # Leer hojas fuente
    try:
        ws_1s = spreadsheet.worksheet(sheet_1s)
    except Exception:
        return {'success': False, 'error': f'Hoja "{sheet_1s}" no encontrada.'}
    try:
        ws_2s = spreadsheet.worksheet(sheet_2s)
    except Exception:
        return {'success': False, 'error': f'Hoja "{sheet_2s}" no encontrada.'}

    # Leer claves
    key_1s = _get_answer_key(ws_1s)
    key_2s = _get_answer_key(ws_2s)
    if not key_1s:
        return {'success': False,
                'error': f'La hoja "{sheet_1s}" no tiene clave de respuestas.'}
    if not key_2s:
        return {'success': False,
                'error': f'La hoja "{sheet_2s}" no tiene clave de respuestas.'}

    # Extraer estudiantes
    students_1s = _extract_students(ws_1s)
    students_2s = _extract_students(ws_2s)

    # Unir por ID — incluir estudiantes que estén en al menos una sesión
    all_ids = set(students_1s.keys()) | set(students_2s.keys())
    if not all_ids:
        return {'success': False, 'error': 'No se encontraron estudiantes.'}

    # Calcular resultados
    results = []
    for sid in sorted(all_ids):
        s1 = students_1s.get(sid)
        s2 = students_2s.get(sid)
        name = (s1 or s2)['name']
        ans_1s = s1['answers'] if s1 else []
        ans_2s = s2['answers'] if s2 else []

        scores = {}
        for subj_name, ranges, points in SIPAGRE_SUBJECTS:
            correct = 0
            total_q = 0
            for session, start, end in ranges:
                if session == '1S':
                    correct += _count_correct_range(ans_1s, key_1s, start, end)
                else:
                    correct += _count_correct_range(ans_2s, key_2s, start, end)
                total_q += (end - start + 1)
            scores[subj_name] = {
                'correct': correct,
                'total':   total_q,
                'points':  round(correct * points, 2),
            }

        mat  = int(round(scores['Matematica']['points']))
        lect = int(round(scores['Lectura Critica']['points']))
        soc  = int(round(scores['Sociales']['points']))
        nat  = int(round(scores['Naturales']['points']))
        ing  = int(round(scores['Ingles']['points']))
        general = int(round(5 * ((mat*3 + lect*3 + soc*3 + nat*3) / 13)))

        results.append({
            'id':       sid,
            'name':     name,
            'mat':      mat,
            'lect':     lect,
            'soc':      soc,
            'nat':      nat,
            'ing':      ing,
            'general':  general,
        })

    # Crear/sobrescribir hoja de resultados
    try:
        ws_res = spreadsheet.worksheet(results_sheet)
        ws_res.clear()
    except Exception:
        ws_res = spreadsheet.add_worksheet(
            title=results_sheet, rows=len(results) + 5, cols=20)

    # Encabezado
    header = [
        'ID Estudiante', 'Nombre',
        'Matematica', 'Lectura Critica', 'Sociales',
        'Naturales', 'Ingles', 'Puntaje General'
    ]
    ws_res.update('A1:H1', [header], value_input_option='RAW')

    # Datos
    rows = []
    for r in results:
        rows.append([
            r['id'], r['name'],
            r['mat'], r['lect'], r['soc'],
            r['nat'], r['ing'], r['general'],
        ])

    if rows:
        end_col = _col_letter(len(header) - 1)
        end_row = len(rows) + 1
        ws_res.update(f'A2:{end_col}{end_row}', rows, value_input_option='RAW')

    # Formato encabezado
    try:
        ws_res.format('A1:H1', {
            'backgroundColor': {'red': 0.13, 'green': 0.27, 'blue': 0.53},
            'textFormat': {'bold': True,
                           'foregroundColor': {'red': 1, 'green': 1, 'blue': 1}},
            'horizontalAlignment': 'CENTER'
        })
    except Exception:
        pass

    return {
        'success':  True,
        'students': len(results),
        'results':  results,
        'url':      f'https://docs.google.com/spreadsheets/d/{SPREADSHEET_ID}/edit',
    }
