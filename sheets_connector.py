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

SPREADSHEET_ID   = os.environ.get('SPREADSHEET_ID', '15erGbl2O6j7SGORYl2FbOO2wVV_r9vFzE4c304y-PUs')
CREDENTIALS_FILE = os.environ.get('GOOGLE_CREDENTIALS_FILE', 'credentials.json')
CREDENTIALS_JSON = os.environ.get('GOOGLE_CREDENTIALS_JSON', None)

# Columna donde empieza P1 (base-0: col 4 = columna E)
COL_OFFSET = 4
KEY_ROW_NAME = '*** CLAVE ***'
COURSES_SHEET = 'Cursos'
SIMULACROS_SHEET = 'Simulacros'

# Tipos de simulacro
SIM_COMPLETO = 'completo'   # 2 sesiones (1S + 2S)
SIM_MEDIA    = 'media'      # 1 sesión (M)


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
    h += ['Respondidas', 'Correctas', 'Porcentaje', 'Curso']
    return h


def _find_curso_col(worksheet) -> int:
    """Devuelve el índice base-0 de la columna 'Curso' o -1 si no existe."""
    try:
        header = worksheet.row_values(1)
        for i, h in enumerate(header):
            if h.strip().lower() == 'curso':
                return i
    except Exception:
        pass
    return -1


def _ensure_curso_column(worksheet, n_questions=125) -> int:
    """Garantiza que exista la columna 'Curso' al final. Devuelve su indice 0-based."""
    idx = _find_curso_col(worksheet)
    if idx >= 0:
        return idx
    try:
        header = worksheet.row_values(1) or []
        new_idx = len(header)
        col_letter = _col_letter(new_idx)
        worksheet.update(f'{col_letter}1', [['Curso']], value_input_option='RAW')
        return new_idx
    except Exception:
        return -1


# ── Gestión de cursos ─────────────────────────────────────────────────────
def _grade_from_course_name(name: str) -> str:
    """Deriva el grado de un nombre de curso numérico tipo '701', '1001'.
       701  -> '7'  (3 digitos: primer digito)
       1001 -> '10' (4 digitos: primeros 2)
       1101 -> '11'
       Si no es numérico, retorna ''."""
    n = (name or '').strip()
    if not n.isdigit():
        return ''
    if len(n) == 4:
        return n[:2]
    if len(n) == 3:
        return n[:1]
    return ''


def _ensure_courses_sheet():
    """Crea/verifica la hoja 'Cursos'. Asegura columnas: Curso, Grado."""
    ss = _open_spreadsheet()
    try:
        ws = ss.worksheet(COURSES_SHEET)
        # Verificar header
        header = ws.row_values(1) or []
        if len(header) < 2 or header[1].strip().lower() != 'grado':
            ws.update('A1:B1', [['Curso', 'Grado']], value_input_option='RAW')
    except Exception:
        ws = ss.add_worksheet(title=COURSES_SHEET, rows=200, cols=3)
        ws.update('A1:B1', [['Curso', 'Grado']], value_input_option='RAW')
        try:
            ws.format('A1:B1', {
                'backgroundColor': {'red': 0.39, 'green': 0.40, 'blue': 0.95},
                'textFormat': {'bold': True,
                               'foregroundColor': {'red':1,'green':1,'blue':1}},
                'horizontalAlignment': 'CENTER'
            })
        except Exception:
            pass
    return ws


def list_courses() -> list:
    """Lista los cursos guardados como [{name, grado}, ...] sin duplicados, ordenados."""
    try:
        ws = _ensure_courses_sheet()
        rows = ws.get_all_values()[1:]  # saltar encabezado
        seen_names = set()
        result = []
        for r in rows:
            name = (r[0] if len(r) > 0 else '').strip()
            grado = (r[1] if len(r) > 1 else '').strip()
            if not name or name in seen_names:
                continue
            seen_names.add(name)
            # Si no tiene grado guardado, intentar derivarlo del nombre
            if not grado:
                grado = _grade_from_course_name(name)
            result.append({'name': name, 'grado': grado})
        # Ordenar por (grado numérico, nombre)
        def _key(c):
            g = c['grado']
            try:
                gi = int(g)
            except Exception:
                gi = 99
            return (gi, c['name'])
        result.sort(key=_key)
        return result
    except Exception:
        return []


def add_course(name: str, grado: str = '') -> dict:
    """Agrega un curso si no existe."""
    name  = (name or '').strip()
    grado = (grado or '').strip()
    if not name:
        return {'success': False, 'error': 'El nombre no puede estar vacío.'}
    if len(name) > 50:
        return {'success': False, 'error': 'Nombre demasiado largo (máx 50).'}
    if not grado:
        grado = _grade_from_course_name(name)
    try:
        existing = list_courses()
        if any(c['name'] == name for c in existing):
            return {'success': False, 'error': f'El curso "{name}" ya existe.'}
        ws = _ensure_courses_sheet()
        ws.append_row([name, grado], value_input_option='RAW')
        return {'success': True, 'name': name, 'grado': grado,
                'courses': list_courses()}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def delete_course(name: str) -> dict:
    """Elimina un curso por nombre."""
    name = (name or '').strip()
    if not name:
        return {'success': False, 'error': 'Nombre vacío.'}
    try:
        ws = _ensure_courses_sheet()
        col = ws.col_values(1)
        for i, c in enumerate(col):
            if i == 0:
                continue
            if (c or '').strip() == name:
                ws.delete_rows(i + 1)
                return {'success': True, 'courses': list_courses()}
        return {'success': False, 'error': 'Curso no encontrado.'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ── Simulacros ────────────────────────────────────────────────────────────
def _ensure_simulacros_sheet():
    """Crea/verifica la hoja 'Simulacros'. Columnas: Nombre, Fecha, Tipo, Grados."""
    ss = _open_spreadsheet()
    try:
        ws = ss.worksheet(SIMULACROS_SHEET)
        header = ws.row_values(1) or []
        if len(header) < 4 or header[0].strip().lower() != 'nombre':
            ws.update('A1:D1', [['Nombre', 'Fecha', 'Tipo', 'Grados']],
                      value_input_option='RAW')
    except Exception:
        ws = ss.add_worksheet(title=SIMULACROS_SHEET, rows=200, cols=5)
        ws.update('A1:D1', [['Nombre', 'Fecha', 'Tipo', 'Grados']],
                  value_input_option='RAW')
        try:
            ws.format('A1:D1', {
                'backgroundColor': {'red': 0.39, 'green': 0.40, 'blue': 0.95},
                'textFormat': {'bold': True,
                               'foregroundColor': {'red':1,'green':1,'blue':1}},
                'horizontalAlignment': 'CENTER'
            })
        except Exception:
            pass
    return ws


def simulacro_sheet_names(nombre: str, tipo: str) -> dict:
    """Devuelve los nombres de las hojas asociadas a un simulacro.
       Convención:
         Completo -> '{nombre} - 1S' y '{nombre} - 2S'
         Media    -> '{nombre} M'
       Hoja de resultados:
         Completo -> 'Resultados {nombre}'
         Media    -> 'Resultados {nombre} M'
    """
    nombre = (nombre or '').strip()
    if tipo == SIM_COMPLETO:
        return {
            'sheets':  [f'{nombre} - 1S', f'{nombre} - 2S'],
            'results': f'Resultados {nombre}',
        }
    else:
        return {
            'sheets':  [f'{nombre} M'],
            'results': f'Resultados {nombre} M',
        }


def list_simulacros() -> list:
    """Lista los simulacros como [{nombre, fecha, tipo, grados, sheets, results}, ...]."""
    try:
        ws = _ensure_simulacros_sheet()
        rows = ws.get_all_values()[1:]
        out = []
        seen = set()
        for r in rows:
            nombre = (r[0] if len(r) > 0 else '').strip()
            fecha  = (r[1] if len(r) > 1 else '').strip()
            tipo   = (r[2] if len(r) > 2 else '').strip().lower()
            grados = (r[3] if len(r) > 3 else '').strip()
            if not nombre or nombre in seen:
                continue
            seen.add(nombre)
            if tipo not in (SIM_COMPLETO, SIM_MEDIA):
                tipo = SIM_COMPLETO
            grados_list = [g.strip() for g in grados.split(',') if g.strip()]
            sn = simulacro_sheet_names(nombre, tipo)
            out.append({
                'nombre':  nombre,
                'fecha':   fecha,
                'tipo':    tipo,
                'grados':  grados_list,
                'sheets':  sn['sheets'],
                'results': sn['results'],
            })
        # Ordenar por fecha desc (mas recientes primero)
        out.sort(key=lambda s: s['fecha'], reverse=True)
        return out
    except Exception:
        return []


def add_simulacro(nombre: str, fecha: str, tipo: str, grados: list) -> dict:
    """Crea un nuevo simulacro y sus hojas asociadas vacías (con estructura)."""
    nombre = (nombre or '').strip()
    fecha  = (fecha or '').strip()
    tipo   = (tipo or '').strip().lower()
    if not nombre:
        return {'success': False, 'error': 'El nombre es obligatorio.'}
    if tipo not in (SIM_COMPLETO, SIM_MEDIA):
        return {'success': False, 'error': 'Tipo inválido (completo o media).'}
    if not isinstance(grados, list):
        grados = []
    grados = [str(g).strip() for g in grados if str(g).strip()]
    try:
        existing = list_simulacros()
        if any(s['nombre'] == nombre for s in existing):
            return {'success': False, 'error': f'El simulacro "{nombre}" ya existe.'}

        ws = _ensure_simulacros_sheet()
        ws.append_row([nombre, fecha, tipo, ','.join(grados)],
                      value_input_option='RAW')

        # Crear las hojas asociadas (vacías con estructura)
        sn = simulacro_sheet_names(nombre, tipo)
        spreadsheet = _open_spreadsheet()
        n_q = 125  # default
        for sname in sn['sheets']:
            try:
                spreadsheet.worksheet(sname)
            except Exception:
                w = spreadsheet.add_worksheet(title=sname, rows=1000, cols=140)
                _ensure_structure(w, n_q)
                _ensure_curso_column(w, n_q)

        return {'success': True, 'simulacro': {
            'nombre':  nombre, 'fecha': fecha, 'tipo': tipo,
            'grados':  grados,
            'sheets':  sn['sheets'],
            'results': sn['results'],
        }, 'simulacros': list_simulacros()}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def delete_simulacro(nombre: str) -> dict:
    """Elimina del catálogo (NO borra las hojas con datos)."""
    nombre = (nombre or '').strip()
    if not nombre:
        return {'success': False, 'error': 'Nombre vacío.'}
    try:
        ws = _ensure_simulacros_sheet()
        col = ws.col_values(1)
        for i, c in enumerate(col):
            if i == 0:
                continue
            if (c or '').strip() == nombre:
                ws.delete_rows(i + 1)
                return {'success': True, 'simulacros': list_simulacros()}
        return {'success': False, 'error': 'Simulacro no encontrado.'}
    except Exception as e:
        return {'success': False, 'error': str(e)}


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
                   answers: list, sheet_name: str,
                   curso: str = '') -> dict:
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
    # Asegurar columna Curso
    curso_col = _ensure_curso_column(worksheet, n_questions)

    # Obtener clave
    key      = _get_answer_key(worksheet)
    correct  = _count_correct(answers, key)
    detected = len([a for a in answers if a not in ('?', '')])

    # Porcentaje sobre el total de preguntas con clave definida
    key_n   = _key_total(key)
    pct_str = f'{round(correct / key_n * 100)}%' if key_n > 0 else ''

    # ── Eliminar fila de TOTALES existente antes de agregar el estudiante ─────
    all_rows = worksheet.get_all_values()
    for i, row in enumerate(all_rows):
        if row and len(row) > 2 and row[2] == '--- TOTALES ---':
            worksheet.delete_rows(i + 1)
            break

    # ── Agregar fila del estudiante ───────────────────────────────────────────
    now      = datetime.now()
    row_data = [now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'),
                student_name, exam_id]
    row_data += [a if a != '?' else '—' for a in answers]
    row_data += [detected,
                 correct if key_n > 0 else '',
                 pct_str]
    # Agregar curso al final si la columna existe
    if curso_col >= 0:
        # rellenar gaps si curso_col > len(row_data)
        while len(row_data) < curso_col:
            row_data.append('')
        row_data.append(curso or '')

    worksheet.append_row(row_data, value_input_option='RAW')
    row_num = len(worksheet.col_values(1))

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

# M SIPAGRE — Una sola jornada de 125 preguntas
# Cada materia tiene 25 preguntas → multiplicador 4.0 para max 100 por materia
# Puntaje general máx ≈ 461 (escalado 0-500 como SIPAGRE)
M_SIPAGRE_SUBJECTS = [
    ('Matematica',       (0,  24),   4.0),
    ('Lectura Critica',  (25, 49),   4.0),
    ('Sociales',         (50, 74),   4.0),
    ('Naturales',        (75, 99),   4.0),
    ('Ingles',           (100,124),  4.0),
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
    Extrae dict {key: {'name','id','answers','curso'}} de una hoja.
    Salta fila 1 (encabezado), fila 2 (clave) y fila TOTALES.
    Acepta estudiantes con solo nombre o solo ID (al menos uno).
    """
    all_rows = worksheet.get_all_values()
    if not all_rows:
        return {}
    header = all_rows[0]
    curso_col = -1
    for i, h in enumerate(header):
        if (h or '').strip().lower() == 'curso':
            curso_col = i
            break
    students = {}
    for row_idx, row in enumerate(all_rows[2:]):
        if not row or len(row) < 5:
            continue
        name = row[2].strip() if len(row) > 2 else ''
        sid  = row[3].strip() if len(row) > 3 else ''
        if name in (KEY_ROW_NAME, '--- TOTALES ---'):
            continue
        # Aceptar si tiene al menos nombre o ID
        if not name and not sid:
            continue
        answers = row[COL_OFFSET:]
        clean = []
        for a in answers:
            a = a.strip()
            if a in ('', '?', '\u2014', '-'):
                clean.append('')
            elif len(a) == 1 and a.upper() in 'ABCDEFGH':
                clean.append(a.upper())
            else:
                break
        curso = ''
        if curso_col >= 0 and curso_col < len(row):
            curso = (row[curso_col] or '').strip()
        # Key unica: ID si existe, sino nombre+row
        key = sid if sid else f'_{row_idx}_{name}'
        students[key] = {
            'name':    name,
            'id':      sid,
            'answers': clean,
            'curso':   curso,
        }
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
    for key in sorted(all_ids):
        s1 = students_1s.get(key)
        s2 = students_2s.get(key)
        s_any = s1 or s2
        name  = s_any.get('name','')
        sid   = s_any.get('id','') or (key if not key.startswith('_') else '')
        # Curso: priorizar el de 1S, luego 2S
        curso = ''
        if s1 and s1.get('curso'): curso = s1['curso']
        elif s2 and s2.get('curso'): curso = s2['curso']
        ans_1s = s1['answers'] if s1 else []
        ans_2s = s2['answers'] if s2 else []

        scores = {}
        for subj_name, ranges, _points_legacy in SIPAGRE_SUBJECTS:
            correct = 0
            total_q = 0
            for session, start, end in ranges:
                if session == '1S':
                    correct += _count_correct_range(ans_1s, key_1s, start, end)
                else:
                    correct += _count_correct_range(ans_2s, key_2s, start, end)
                total_q += (end - start + 1)
            # Puntaje 0-100 = correctas / total * 100
            pct_score = (correct / total_q * 100) if total_q > 0 else 0.0
            scores[subj_name] = {
                'correct': correct,
                'total':   total_q,
                'points':  round(pct_score, 2),
            }

        mat  = int(round(scores['Matematica']['points']))
        lect = int(round(scores['Lectura Critica']['points']))
        soc  = int(round(scores['Sociales']['points']))
        nat  = int(round(scores['Naturales']['points']))
        ing  = int(round(scores['Ingles']['points']))
        # Pesos: Mat, Lect, Soc, Nat = 3 cada una; Ing = 1.  Suma = 13
        general = int(round(5 * ((mat*3 + lect*3 + soc*3 + nat*3 + ing*1) / 13)))

        results.append({
            'id':       sid,
            'name':     name,
            'curso':    curso,
            'mat':      mat,
            'lect':     lect,
            'soc':      soc,
            'nat':      nat,
            'ing':      ing,
            'general':  general,
        })

    # Crear/sobrescribir hoja de resultados
    needed_rows = len(results) + 5  # header + datos + promedio + margen
    try:
        ws_res = spreadsheet.worksheet(results_sheet)
        ws_res.clear()
        if ws_res.row_count < needed_rows:
            ws_res.resize(rows=needed_rows)
    except Exception:
        ws_res = spreadsheet.add_worksheet(
            title=results_sheet, rows=needed_rows, cols=20)

    # Encabezado (con Curso)
    header = [
        'ID Estudiante', 'Nombre', 'Curso',
        'Matematica', 'Lectura Critica', 'Sociales',
        'Naturales', 'Ingles', 'Puntaje General'
    ]
    ws_res.update('A1:I1', [header], value_input_option='RAW')

    # Datos
    rows = []
    for r in results:
        rows.append([
            r['id'], r['name'], r.get('curso',''),
            r['mat'], r['lect'], r['soc'],
            r['nat'], r['ing'], r['general'],
        ])

    if rows:
        end_col = _col_letter(len(header) - 1)
        end_row = len(rows) + 1
        ws_res.update(f'A2:{end_col}{end_row}', rows, value_input_option='RAW')

        # Fila de promedios
        n = len(results)
        avg_mat  = int(round(sum(r['mat']     for r in results) / n))
        avg_lect = int(round(sum(r['lect']    for r in results) / n))
        avg_soc  = int(round(sum(r['soc']     for r in results) / n))
        avg_nat  = int(round(sum(r['nat']     for r in results) / n))
        avg_ing  = int(round(sum(r['ing']     for r in results) / n))
        avg_gen  = int(round(sum(r['general'] for r in results) / n))
        avg_row = end_row + 1
        ws_res.update(f'A{avg_row}:I{avg_row}',
                      [['', 'PROMEDIO', '', avg_mat, avg_lect, avg_soc, avg_nat, avg_ing, avg_gen]],
                      value_input_option='RAW')
        try:
            ws_res.format(f'A{avg_row}:I{avg_row}', {
                'backgroundColor': {'red': 0.93, 'green': 0.93, 'blue': 0.93},
                'textFormat': {'bold': True},
                'horizontalAlignment': 'CENTER'
            })
        except Exception:
            pass

    # Formato encabezado
    try:
        ws_res.format('A1:I1', {
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


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADOS M SIPAGRE — una sola jornada (125 preguntas)
# ─────────────────────────────────────────────────────────────────────────────
def generate_msipagre_results(sheet_m: str = 'M SIPAGRE',
                              results_sheet: str = 'Resultados M SIPAGRE') -> dict:
    """
    Calcula puntajes por materia y puntaje general para M SIPAGRE
    (una sola jornada de 125 preguntas) y los escribe en una hoja nueva/existente.
    """
    spreadsheet = _open_spreadsheet()

    try:
        ws_m = spreadsheet.worksheet(sheet_m)
    except Exception:
        return {'success': False, 'error': f'Hoja "{sheet_m}" no encontrada.'}

    key_m = _get_answer_key(ws_m)
    if not key_m:
        return {'success': False,
                'error': f'La hoja "{sheet_m}" no tiene clave de respuestas.'}

    students = _extract_students(ws_m)
    if not students:
        return {'success': False,
                'error': f'No se encontraron estudiantes en "{sheet_m}". Verifica que la hoja tenga filas de estudiantes con nombre o ID.'}

    results = []
    for key in sorted(students.keys()):
        s     = students[key]
        name  = s['name']
        sid   = s.get('id', '')
        curso = s.get('curso', '')
        ans   = s['answers']

        scores = {}
        for subj_name, (start, end), _points_legacy in M_SIPAGRE_SUBJECTS:
            correct = _count_correct_range(ans, key_m, start, end)
            total_q = end - start + 1
            pct_score = (correct / total_q * 100) if total_q > 0 else 0.0
            scores[subj_name] = round(pct_score, 2)

        mat  = int(round(scores['Matematica']))
        lect = int(round(scores['Lectura Critica']))
        soc  = int(round(scores['Sociales']))
        nat  = int(round(scores['Naturales']))
        ing  = int(round(scores['Ingles']))
        # Pesos: Mat, Lect, Soc, Nat = 3 cada una; Ing = 1.  Suma = 13
        general = int(round(5 * ((mat*3 + lect*3 + soc*3 + nat*3 + ing*1) / 13)))

        results.append({
            'id':       sid,
            'name':     name,
            'curso':    curso,
            'mat':      mat,
            'lect':     lect,
            'soc':      soc,
            'nat':      nat,
            'ing':      ing,
            'general':  general,
            'type':     'M',
        })

    # Crear/sobrescribir hoja de resultados
    needed_rows = len(results) + 5
    try:
        ws_res = spreadsheet.worksheet(results_sheet)
        ws_res.clear()
        if ws_res.row_count < needed_rows:
            ws_res.resize(rows=needed_rows)
    except Exception:
        ws_res = spreadsheet.add_worksheet(
            title=results_sheet, rows=needed_rows, cols=20)

    header = [
        'ID Estudiante', 'Nombre', 'Curso',
        'Matematica', 'Lectura Critica', 'Sociales',
        'Naturales', 'Ingles', 'Puntaje General'
    ]
    ws_res.update('A1:I1', [header], value_input_option='RAW')

    rows = []
    for r in results:
        rows.append([
            r['id'], r['name'], r.get('curso',''),
            r['mat'], r['lect'], r['soc'],
            r['nat'], r['ing'], r['general'],
        ])

    if rows:
        end_col = _col_letter(len(header) - 1)
        end_row = len(rows) + 1
        ws_res.update(f'A2:{end_col}{end_row}', rows, value_input_option='RAW')

        n = len(results)
        avg = {k: int(round(sum(r[k] for r in results) / n))
               for k in ['mat','lect','soc','nat','ing','general']}
        avg_row = end_row + 1
        ws_res.update(
            f'A{avg_row}:I{avg_row}',
            [['', 'PROMEDIO', '', avg['mat'], avg['lect'], avg['soc'],
              avg['nat'], avg['ing'], avg['general']]],
            value_input_option='RAW')
        try:
            ws_res.format(f'A{avg_row}:I{avg_row}', {
                'backgroundColor': {'red': 0.93, 'green': 0.93, 'blue': 0.93},
                'textFormat': {'bold': True},
                'horizontalAlignment': 'CENTER'
            })
        except Exception:
            pass

    try:
        ws_res.format('A1:I1', {
            'backgroundColor': {'red': 0.30, 'green': 0.39, 'blue': 0.85},
            'textFormat': {'bold': True,
                           'foregroundColor': {'red':1,'green':1,'blue':1}},
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
