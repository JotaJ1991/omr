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
DISTRIBUCION_SHEET = 'Distribucion'
KEYS_GRADE_SHEET   = 'ClavesGrado'
ANULADAS_SHEET     = 'PreguntasAnuladas'

# Tipos de simulacro
SIM_COMPLETO = 'completo'   # 2 sesiones (1S + 2S)
SIM_MEDIA    = 'media'      # 1 sesión (M)

# Materias canónicas
MATERIAS = ['Matematica', 'Lectura Critica', 'Sociales', 'Naturales', 'Ingles']

# Distribución estándar Media (5 bloques de 25 preguntas)
_MEDIA_DEFAULT = [
    {'materia':'Matematica',      'sesion':'M', 'inicio':1,   'fin':25},
    {'materia':'Lectura Critica', 'sesion':'M', 'inicio':26,  'fin':50},
    {'materia':'Sociales',        'sesion':'M', 'inicio':51,  'fin':75},
    {'materia':'Naturales',       'sesion':'M', 'inicio':76,  'fin':100},
    {'materia':'Ingles',          'sesion':'M', 'inicio':101, 'fin':125},
]

DEFAULT_DISTRIBUCIONES = {
    ('completo', '11'): [
        {'materia':'Matematica',      'sesion':'1S', 'inicio':1,  'fin':25},
        {'materia':'Matematica',      'sesion':'2S', 'inicio':26, 'fin':50},
        {'materia':'Lectura Critica', 'sesion':'1S', 'inicio':26, 'fin':66},
        {'materia':'Sociales',        'sesion':'1S', 'inicio':67, 'fin':91},
        {'materia':'Sociales',        'sesion':'2S', 'inicio':1,  'fin':25},
        {'materia':'Naturales',       'sesion':'1S', 'inicio':92, 'fin':120},
        {'materia':'Naturales',       'sesion':'2S', 'inicio':51, 'fin':79},
        {'materia':'Ingles',          'sesion':'2S', 'inicio':80, 'fin':134},
    ],
    # Media: 5 bloques de 25 preguntas para todos los grados 6° a 10°
    ('media', '6'):  _MEDIA_DEFAULT,
    ('media', '7'):  _MEDIA_DEFAULT,
    ('media', '8'):  _MEDIA_DEFAULT,
    ('media', '9'):  _MEDIA_DEFAULT,
    ('media', '10'): _MEDIA_DEFAULT,
}


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
    """Crea/verifica la hoja 'Simulacros'. Columnas: Nombre, Fecha, Tipo, Grados.
       Si la hoja se acaba de crear, registra simulacros legacy detectados."""
    ss = _open_spreadsheet()
    fresh = False
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
        fresh = True

    # Migracion de simulacros legacy (hojas '1S SIPAGRE', '2S SIPAGRE', 'M SIPAGRE')
    if fresh:
        try:
            existing_titles = {w.title for w in ss.worksheets()}
            rows_to_add = []
            if '1S SIPAGRE' in existing_titles and '2S SIPAGRE' in existing_titles:
                rows_to_add.append(['SIPAGRE Legacy', '', SIM_COMPLETO, '11'])
            if 'M SIPAGRE' in existing_titles:
                rows_to_add.append(['SIPAGRE M Legacy', '', SIM_MEDIA, '6,7,8,9,10'])
            for r in rows_to_add:
                ws.append_row(r, value_input_option='RAW')
        except Exception:
            pass
    return ws


def simulacro_sheet_names_for_legacy(nombre):
    """Mapeo especial para simulacros legacy con hojas con nombres distintos."""
    if nombre == 'SIPAGRE Legacy':
        return {'sheets': ['1S SIPAGRE', '2S SIPAGRE'], 'results': 'Resultados SIPAGRE'}
    if nombre == 'SIPAGRE M Legacy':
        return {'sheets': ['M SIPAGRE'], 'results': 'Resultados M SIPAGRE'}
    return None


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
            # Si es un simulacro legacy, usar mapeo especial
            sn = simulacro_sheet_names_for_legacy(nombre) or simulacro_sheet_names(nombre, tipo)
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


def analyze_simulacro_questions(simulacro_nombre: str) -> dict:
    """
    Para cada grado del simulacro genera su propia lista de preguntas
    clasificadas según la distribución configurada para ese grado.

    Retorna:
      {
        success: True,
        simulacro, tipo,
        grades_data: {grado: [{subject, q, key, total, correct, pct,
                                by_grade, by_curso}, ...]},
        questions: [...]   # vista canónica (primer grado disponible)
      }
    """
    sims = list_simulacros()
    sim = next((s for s in sims if s['nombre'] == simulacro_nombre), None)
    if not sim:
        return {'success': False, 'error': f'Simulacro "{simulacro_nombre}" no encontrado.'}

    spreadsheet = _open_spreadsheet()
    courses = list_courses()
    curso_to_grade = {c['name']: c.get('grado','') for c in courses}

    def grade_of(curso):
        c = (curso or '').strip()
        if not c: return ''
        g = curso_to_grade.get(c, '')
        if g: return g
        return _grade_from_course_name(c)

    distribuciones = list_distribuciones()

    # Cargar hojas y datos
    if sim['tipo'] == SIM_COMPLETO:
        try:
            ws_1s = spreadsheet.worksheet(sim['sheets'][0])
            ws_2s = spreadsheet.worksheet(sim['sheets'][1])
        except Exception:
            return {'success': False, 'error': 'No se encontraron las hojas 1S/2S del simulacro.'}
        keys_global = {'1S': _get_answer_key(ws_1s), '2S': _get_answer_key(ws_2s)}
        students_by_session = {
            '1S': _extract_students(ws_1s),
            '2S': _extract_students(ws_2s),
        }
    else:
        try:
            ws_m = spreadsheet.worksheet(sim['sheets'][0])
        except Exception:
            return {'success': False, 'error': 'No se encontró la hoja M del simulacro.'}
        keys_global = {'M': _get_answer_key(ws_m)}
        students_by_session = {'M': _extract_students(ws_m)}

    # Cache de claves por (session, grado)
    keys_cache = {}
    def key_for(session, grado):
        ck = (session, grado)
        if ck in keys_cache: return keys_cache[ck]
        specific = []
        try: specific = get_key_for_grade(simulacro_nombre, session, grado)
        except Exception: pass
        keys_cache[ck] = specific or keys_global.get(session, [])
        return keys_cache[ck]

    # Cache de anuladas por (session, grado) — sets de int (1-based)
    anu_cache = {}
    def anu_for(session, grado):
        ck = (session, grado)
        if ck in anu_cache: return anu_cache[ck]
        try:
            anu_cache[ck] = set(get_anuladas_for_grade(simulacro_nombre, session, grado))
        except Exception:
            anu_cache[ck] = set()
        return anu_cache[ck]

    grades_data = {}
    target_grados = list(sim.get('grados') or [])

    # Además incluir grados detectados a partir de los estudiantes presentes,
    # por si el simulacro fue creado sin contemplar algún grado (ej. 10°
    # añadido después). De este modo el análisis siempre incluye a los
    # grados que efectivamente contestaron.
    try:
        detected_grados = set()
        for _ses, _students in students_by_session.items():
            for _stu in _students.values():
                g = grade_of((_stu.get('curso','') or '').strip())
                if g:
                    detected_grados.add(g)
        merged = set(target_grados) | detected_grados
        target_grados = sorted(
            merged,
            key=lambda g: int(g) if g.isdigit() else 99
        )
    except Exception:
        pass

    for grado in target_grados:
        # Distribución del grado (configurada o default)
        dist = distribuciones.get((sim['tipo'], grado))
        if not dist:
            dist = DEFAULT_DISTRIBUCIONES.get((sim['tipo'], grado))
        if not dist:
            continue

        questions = []
        # Por cada bloque (materia, sesion, inicio, fin) generar preguntas
        # Numeración relativa por materia
        materia_counters = {}
        for entry in dist:
            mat = entry['materia']
            ses = (entry['sesion'] or '').upper()
            ini = int(entry['inicio'])
            fin = int(entry['fin'])
            students_dict = students_by_session.get(ses, {})
            ses_anu = anu_for(ses, grado)
            for idx in range(ini - 1, fin):
                materia_counters.setdefault(mat, 0)
                materia_counters[mat] += 1
                q_subj_idx = materia_counters[mat]
                q_num_orig = idx + 1
                is_anulada = q_num_orig in ses_anu
                # Clave que aplica a este grado
                k_arr = key_for(ses, grado)
                correct_ans = (k_arr[idx] or '').strip() if idx < len(k_arr) else ''
                # Pregunta anulada: cuenta a TODOS como correctos
                if is_anulada:
                    total = 0; correct = 0
                    by_curso = {}
                    for sid, stu in students_dict.items():
                        if grade_of((stu.get('curso','') or '').strip()) != grado:
                            continue
                        # Todos los estudiantes del grado cuentan como correctos
                        total   += 1
                        correct += 1
                        cc = (stu.get('curso','') or '').strip()
                        if cc:
                            by_curso.setdefault(cc, [0, 0])
                            by_curso[cc][0] += 1
                            by_curso[cc][1] += 1
                    if total == 0: continue
                    questions.append({
                        'subject':  mat,
                        'q':        q_subj_idx,
                        'sesion':   ses,
                        'idx_orig': q_num_orig,
                        'key':      correct_ans or '*',
                        'total':    total,
                        'correct':  correct,
                        'pct':      100,
                        'anulada':  True,
                        'by_grade': {grado: {'n': total, 'c': correct, 'pct': 100}},
                        'by_curso': {cc: {'n': t, 'c': c, 'pct': 100}
                                      for cc, (t,c) in by_curso.items() if t},
                    })
                    continue
                if not correct_ans or correct_ans in ('?', '—'): continue

                # IMPORTANTE: el denominador es el TOTAL de estudiantes del grado,
                # incluyendo a los que no respondieron (cuentan como incorrecto).
                # Esto mantiene un denominador estable entre preguntas y refleja
                # fielmente el % de acierto sobre la población.
                total = 0; correct = 0
                by_curso = {}
                for sid, stu in students_dict.items():
                    # Solo estudiantes del grado actual
                    if grade_of((stu.get('curso','') or '').strip()) != grado:
                        continue
                    ans_list = stu.get('answers', [])
                    ans = (ans_list[idx] if idx < len(ans_list) else '').strip()
                    total += 1
                    ok = bool(ans) and ans == correct_ans
                    if ok: correct += 1
                    cc = (stu.get('curso','') or '').strip()
                    if cc:
                        by_curso.setdefault(cc, [0, 0])
                        by_curso[cc][0] += 1
                        if ok: by_curso[cc][1] += 1
                if total == 0: continue
                questions.append({
                    'subject':  mat,
                    'q':        q_subj_idx,
                    'sesion':   ses,
                    'idx_orig': q_num_orig,
                    'key':      correct_ans,
                    'total':    total,
                    'correct':  correct,
                    'pct':      round(correct / total * 100),
                    'anulada':  False,
                    'by_grade': {grado: {'n': total, 'c': correct,
                                          'pct': round(correct/total*100)}},
                    'by_curso': {cc: {'n': t, 'c': c, 'pct': round(c/t*100)}
                                  for cc, (t,c) in by_curso.items() if t},
                })
        if questions:
            grades_data[grado] = questions

    # Vista por defecto (cuando no hay filtro): usar la del primer grado disponible
    default_questions = []
    if grades_data:
        first_grade = sorted(grades_data.keys(), key=lambda g: int(g) if g.isdigit() else 99)[0]
        default_questions = grades_data[first_grade]

    return {
        'success':     True,
        'simulacro':   simulacro_nombre,
        'tipo':        sim['tipo'],
        'questions':   default_questions,
        'grades_data': grades_data,
    }


# ── Distribución de preguntas por (tipo, grado) ──────────────────────
def _ensure_distribucion_sheet():
    """Hoja 'Distribucion' con columnas: Tipo, Grado, Materia, Sesion, Inicio, Fin."""
    ss = _open_spreadsheet()
    fresh = False
    try:
        ws = ss.worksheet(DISTRIBUCION_SHEET)
        header = ws.row_values(1) or []
        if len(header) < 6 or header[0].strip().lower() != 'tipo':
            ws.update('A1:F1', [['Tipo','Grado','Materia','Sesion','Inicio','Fin']],
                      value_input_option='RAW')
    except Exception:
        ws = ss.add_worksheet(title=DISTRIBUCION_SHEET, rows=200, cols=8)
        ws.update('A1:F1', [['Tipo','Grado','Materia','Sesion','Inicio','Fin']],
                  value_input_option='RAW')
        try:
            ws.format('A1:F1', {
                'backgroundColor': {'red':0.39, 'green':0.40, 'blue':0.95},
                'textFormat': {'bold':True,
                               'foregroundColor':{'red':1,'green':1,'blue':1}},
                'horizontalAlignment':'CENTER',
            })
        except Exception:
            pass
        fresh = True

    # Sembrar defaults faltantes (si la hoja está vacía o no tiene cierta combinación)
    try:
        existing_rows = ws.get_all_values()[1:]
        existing_keys = set()
        for r in existing_rows:
            if len(r) >= 2:
                existing_keys.add(((r[0] or '').strip().lower(), str(r[1] or '').strip()))
        rows_to_add = []
        for (tipo, grado), entries in DEFAULT_DISTRIBUCIONES.items():
            if (tipo, grado) not in existing_keys:
                for e in entries:
                    rows_to_add.append([tipo, grado, e['materia'], e['sesion'], e['inicio'], e['fin']])
        if rows_to_add:
            ws.append_rows(rows_to_add, value_input_option='RAW')
    except Exception:
        pass
    return ws


def list_distribuciones() -> dict:
    """Retorna {(tipo, grado): [{materia, sesion, inicio, fin}, ...]}."""
    try:
        ws = _ensure_distribucion_sheet()
        data = ws.get_all_values()[1:]
        result = {}
        for r in data:
            if len(r) < 6: continue
            tipo  = (r[0] or '').strip().lower()
            grado = (r[1] or '').strip()
            mat   = (r[2] or '').strip()
            ses   = (r[3] or '').strip().upper()
            try:
                ini = int(r[4]); fin = int(r[5])
            except Exception:
                continue
            if not tipo or not grado or not mat or ini < 1 or fin < ini:
                continue
            key = (tipo, grado)
            result.setdefault(key, [])
            result[key].append({'materia': mat, 'sesion': ses,
                                 'inicio': ini, 'fin': fin})
        return result
    except Exception:
        return {}


def list_distribuciones_json() -> list:
    """Versión serializable: lista de {tipo, grado, materias: [...]}"""
    d = list_distribuciones()
    out = []
    for (tipo, grado), entries in d.items():
        out.append({'tipo': tipo, 'grado': grado, 'materias': entries})
    out.sort(key=lambda x: (x['tipo'], int(x['grado']) if x['grado'].isdigit() else 99))
    return out


def save_distribucion(tipo: str, grado: str, materias: list) -> dict:
    """Reemplaza la distribución de (tipo, grado)."""
    tipo  = (tipo or '').strip().lower()
    grado = str(grado or '').strip()
    if tipo not in (SIM_COMPLETO, SIM_MEDIA):
        return {'success': False, 'error': 'Tipo inválido.'}
    if not grado:
        return {'success': False, 'error': 'Grado requerido.'}
    if not isinstance(materias, list) or not materias:
        return {'success': False, 'error': 'Lista de materias vacía.'}
    cleaned = []
    for m in materias:
        mat = (m.get('materia') or '').strip()
        ses = (m.get('sesion') or '').strip().upper()
        try:
            ini = int(m.get('inicio')); fin = int(m.get('fin'))
        except Exception:
            return {'success': False, 'error': 'Inicio/Fin inválidos.'}
        if not mat or ini < 1 or fin < ini:
            return {'success': False, 'error': 'Datos incompletos.'}
        cleaned.append([tipo, grado, mat, ses, ini, fin])

    try:
        ws = _ensure_distribucion_sheet()
        all_rows = ws.get_all_values()
        i = len(all_rows)
        while i >= 2:
            row = all_rows[i-1] if i-1 < len(all_rows) else []
            if (len(row) >= 2 and (row[0] or '').strip().lower() == tipo
                and str(row[1] or '').strip() == grado):
                ws.delete_rows(i)
            i -= 1
        ws.append_rows(cleaned, value_input_option='RAW')
        return {'success': True, 'distribuciones': list_distribuciones_json()}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_distribucion_for(tipo: str, grado: str) -> list:
    """Distribución para (tipo, grado), con fallback al default."""
    tipo  = (tipo or '').strip().lower()
    grado = str(grado or '').strip()
    d = list_distribuciones()
    if (tipo, grado) in d:
        return d[(tipo, grado)]
    if (tipo, grado) in DEFAULT_DISTRIBUCIONES:
        return DEFAULT_DISTRIBUCIONES[(tipo, grado)]
    return []


# ── Claves por (simulacro, sesion, grado) ────────────────────────────────
def _ensure_keys_grade_sheet(max_q=200):
    """
    Hoja 'ClavesGrado' con columnas:
      A: Simulacro  B: Sesion  C: Grado  D...: P1..Pmax_q
    Cada fila almacena la clave para una combinación.
    """
    ss = _open_spreadsheet()
    try:
        ws = ss.worksheet(KEYS_GRADE_SHEET)
        header = ws.row_values(1) or []
        if len(header) < 3 or header[0].strip().lower() != 'simulacro':
            new_header = ['Simulacro', 'Sesion', 'Grado'] + [f'P{i}' for i in range(1, max_q + 1)]
            end_col = _col_letter(len(new_header) - 1)
            ws.update(f'A1:{end_col}1', [new_header], value_input_option='RAW')
    except Exception:
        cols_total = 3 + max_q
        ws = ss.add_worksheet(title=KEYS_GRADE_SHEET, rows=200, cols=cols_total + 5)
        new_header = ['Simulacro', 'Sesion', 'Grado'] + [f'P{i}' for i in range(1, max_q + 1)]
        end_col = _col_letter(len(new_header) - 1)
        ws.update(f'A1:{end_col}1', [new_header], value_input_option='RAW')
        try:
            ws.format(f'A1:{end_col}1', {
                'backgroundColor': {'red':0.39, 'green':0.40, 'blue':0.95},
                'textFormat': {'bold':True,
                               'foregroundColor':{'red':1,'green':1,'blue':1}},
                'horizontalAlignment':'CENTER',
            })
        except Exception:
            pass
    return ws


def get_key_for_grade(simulacro: str, sesion: str, grado: str) -> list:
    """Retorna la clave guardada para (simulacro, sesion, grado), o [] si no existe."""
    simulacro = (simulacro or '').strip()
    sesion    = (sesion or '').strip().upper()
    grado     = str(grado or '').strip()
    if not simulacro or not sesion or not grado:
        return []
    try:
        ws = _ensure_keys_grade_sheet()
        rows = ws.get_all_values()
        for r in rows[1:]:
            if (len(r) >= 3
                and (r[0] or '').strip() == simulacro
                and (r[1] or '').strip().upper() == sesion
                and str(r[2] or '').strip() == grado):
                # Devolver desde la columna 4 hasta la última no vacía
                answers = r[3:]
                while answers and not (answers[-1] or '').strip():
                    answers.pop()
                return [(a or '').strip() for a in answers]
        return []
    except Exception:
        return []


def save_key_for_grade(simulacro: str, sesion: str, grado: str, answers: list) -> dict:
    """Guarda/actualiza la clave para (simulacro, sesion, grado)."""
    simulacro = (simulacro or '').strip()
    sesion    = (sesion or '').strip().upper()
    grado     = str(grado or '').strip()
    if not simulacro or not sesion or not grado:
        return {'success': False, 'error': 'simulacro, sesion y grado son requeridos.'}
    if not isinstance(answers, list):
        return {'success': False, 'error': 'answers debe ser una lista.'}
    n = len(answers)
    if n > 200:
        return {'success': False, 'error': 'Demasiadas respuestas (max 200).'}
    try:
        ws = _ensure_keys_grade_sheet(max_q=max(125, n))
        rows = ws.get_all_values()
        target_row = None
        for i, r in enumerate(rows):
            if i == 0: continue
            if (len(r) >= 3
                and (r[0] or '').strip() == simulacro
                and (r[1] or '').strip().upper() == sesion
                and str(r[2] or '').strip() == grado):
                target_row = i + 1
                break
        # Construir la fila completa
        row_data = [simulacro, sesion, grado] + [(a or '').upper() if a else '' for a in answers]
        end_col = _col_letter(len(row_data) - 1)
        if target_row:
            ws.update(f'A{target_row}:{end_col}{target_row}', [row_data], value_input_option='RAW')
        else:
            ws.append_row(row_data, value_input_option='RAW')
        return {'success': True, 'count': sum(1 for a in answers if a)}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def list_keys_grade() -> list:
    """Lista todas las combinaciones (simulacro, sesion, grado) que tienen clave."""
    try:
        ws = _ensure_keys_grade_sheet()
        rows = ws.get_all_values()[1:]
        out = []
        for r in rows:
            if len(r) < 3: continue
            sim   = (r[0] or '').strip()
            ses   = (r[1] or '').strip().upper()
            grado = str(r[2] or '').strip()
            if not sim or not ses or not grado: continue
            answers = r[3:]
            while answers and not (answers[-1] or '').strip():
                answers.pop()
            count = sum(1 for a in answers if (a or '').strip())
            out.append({
                'simulacro': sim, 'sesion': ses, 'grado': grado,
                'count': count, 'total': len(answers),
            })
        out.sort(key=lambda x: (x['simulacro'], x['sesion'],
                                int(x['grado']) if x['grado'].isdigit() else 99))
        return out
    except Exception:
        return []


# ── Preguntas anuladas por (simulacro, sesion, grado) ──────────────────
# Las preguntas anuladas se cuentan como "correctas para todos" sin importar
# qué marcó el estudiante (o si no marcó nada).
def _ensure_anuladas_sheet():
    """
    Hoja 'PreguntasAnuladas' con columnas:
      A: Simulacro  B: Sesion  C: Grado  D: Preguntas (CSV "22,45")
    Cada fila almacena las preguntas anuladas para una combinación.
    """
    ss = _open_spreadsheet()
    try:
        ws = ss.worksheet(ANULADAS_SHEET)
        header = ws.row_values(1) or []
        if len(header) < 4 or header[0].strip().lower() != 'simulacro':
            ws.update('A1:D1',
                      [['Simulacro', 'Sesion', 'Grado', 'Preguntas']],
                      value_input_option='RAW')
    except Exception:
        ws = ss.add_worksheet(title=ANULADAS_SHEET, rows=200, cols=8)
        ws.update('A1:D1',
                  [['Simulacro', 'Sesion', 'Grado', 'Preguntas']],
                  value_input_option='RAW')
        try:
            ws.format('A1:D1', {
                'backgroundColor': {'red':0.39, 'green':0.40, 'blue':0.95},
                'textFormat': {'bold':True,
                               'foregroundColor':{'red':1,'green':1,'blue':1}},
                'horizontalAlignment':'CENTER',
            })
        except Exception:
            pass
    return ws


def _parse_anuladas_csv(s: str) -> list:
    """'22, 45 ,  67' → [22, 45, 67] (1-based, ordenadas, sin duplicados)"""
    if not s: return []
    out = set()
    for tok in str(s).replace(';', ',').split(','):
        tok = tok.strip()
        if not tok: continue
        try:
            n = int(tok)
            if 1 <= n <= 200: out.add(n)
        except Exception:
            continue
    return sorted(out)


def get_anuladas_for_grade(simulacro: str, sesion: str, grado: str) -> list:
    """Retorna lista de números (1-based) anulados para (simulacro, sesion, grado)."""
    simulacro = (simulacro or '').strip()
    sesion    = (sesion or '').strip().upper()
    grado     = str(grado or '').strip()
    if not simulacro or not sesion or not grado:
        return []
    try:
        ws = _ensure_anuladas_sheet()
        rows = ws.get_all_values()
        for r in rows[1:]:
            if (len(r) >= 4
                and (r[0] or '').strip() == simulacro
                and (r[1] or '').strip().upper() == sesion
                and str(r[2] or '').strip() == grado):
                return _parse_anuladas_csv(r[3])
        return []
    except Exception:
        return []


def save_anuladas_for_grade(simulacro: str, sesion: str, grado: str,
                            preguntas) -> dict:
    """Guarda/actualiza las preguntas anuladas para (simulacro, sesion, grado)."""
    simulacro = (simulacro or '').strip()
    sesion    = (sesion or '').strip().upper()
    grado     = str(grado or '').strip()
    if not simulacro or not sesion or not grado:
        return {'success': False,
                'error': 'simulacro, sesion y grado son requeridos.'}
    if isinstance(preguntas, str):
        nums = _parse_anuladas_csv(preguntas)
    elif isinstance(preguntas, (list, tuple)):
        nums = _parse_anuladas_csv(','.join(str(p) for p in preguntas))
    else:
        return {'success': False,
                'error': 'preguntas debe ser lista o cadena.'}
    csv = ','.join(str(n) for n in nums)
    try:
        ws = _ensure_anuladas_sheet()
        rows = ws.get_all_values()
        target_row = None
        for i, r in enumerate(rows):
            if i == 0: continue
            if (len(r) >= 3
                and (r[0] or '').strip() == simulacro
                and (r[1] or '').strip().upper() == sesion
                and str(r[2] or '').strip() == grado):
                target_row = i + 1
                break
        # Si la lista queda vacía y existe la fila, la limpiamos
        if not nums and target_row:
            ws.update(f'A{target_row}:D{target_row}',
                      [['', '', '', '']], value_input_option='RAW')
            return {'success': True, 'count': 0, 'preguntas': []}
        row_data = [simulacro, sesion, grado, csv]
        if target_row:
            ws.update(f'A{target_row}:D{target_row}',
                      [row_data], value_input_option='RAW')
        else:
            if nums:
                ws.append_row(row_data, value_input_option='RAW')
        return {'success': True, 'count': len(nums), 'preguntas': nums}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def list_anuladas() -> list:
    """Lista todas las combinaciones (simulacro, sesion, grado) con anuladas."""
    try:
        ws = _ensure_anuladas_sheet()
        rows = ws.get_all_values()[1:]
        out = []
        for r in rows:
            if len(r) < 4: continue
            sim   = (r[0] or '').strip()
            ses   = (r[1] or '').strip().upper()
            grado = str(r[2] or '').strip()
            nums  = _parse_anuladas_csv(r[3])
            if not sim or not ses or not grado or not nums: continue
            out.append({
                'simulacro': sim, 'sesion': ses, 'grado': grado,
                'preguntas': nums, 'count': len(nums),
            })
        out.sort(key=lambda x: (x['simulacro'], x['sesion'],
                                int(x['grado']) if x['grado'].isdigit() else 99))
        return out
    except Exception:
        return []


def uppercase_all_student_names() -> dict:
    """Recorre todas las hojas de respuestas y convierte la columna Nombre (col C)
       a MAYÚSCULAS, excepto en filas KEY o TOTALES y hojas administrativas."""
    spreadsheet = _open_spreadsheet()
    skip_titles = {COURSES_SHEET, SIMULACROS_SHEET}
    total_changes = 0
    sheets_touched = 0
    errors = []
    for ws in spreadsheet.worksheets():
        if ws.title in skip_titles or ws.title.startswith('Resultados '):
            continue
        try:
            col = ws.col_values(3)  # columna C: Nombre
            updates = []
            for i, name in enumerate(col):
                if i == 0:  # encabezado
                    continue
                clean = (name or '').strip()
                if not clean or clean in (KEY_ROW_NAME, '--- TOTALES ---'):
                    continue
                upper = clean.upper()
                if upper != name:
                    updates.append({'range': f'C{i+1}', 'values': [[upper]]})
            if updates:
                ws.batch_update(updates, value_input_option='RAW')
                total_changes += len(updates)
                sheets_touched += 1
        except Exception as e:
            errors.append(f'{ws.title}: {e}')
    return {
        'success': True,
        'changes': total_changes,
        'sheets':  sheets_touched,
        'errors':  errors,
    }


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
    El nombre del estudiante se almacena siempre en MAYÚSCULAS.
    """
    student_name = (student_name or '').upper()
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


def _score_student_with_distribution(answers_by_session, keys_by_session,
                                     distribucion, anuladas_by_session=None):
    """
    Calcula los puntajes 0-100 por materia usando una distribución (tipo, grado).

    answers_by_session:  {'1S': [...], '2S': [...], 'M': [...]}
    keys_by_session:     {'1S': [...], '2S': [...], 'M': [...]}
    distribucion:        lista de {materia, sesion, inicio (1-based), fin (1-based)}
    anuladas_by_session: {'1S': set/list de números 1-based, '2S': ..., 'M': ...}
       Las preguntas anuladas cuentan como correctas para todos sin importar
       qué marcó el estudiante (o si no marcó nada). Si la clave para esa
       pregunta no existe, se usa un valor "fantasma" para que igualmente
       cuente: total += 1 y correct += 1.

    Retorna: {materia: float 0-100}.
    """
    anuladas_by_session = anuladas_by_session or {}
    # Normalizar a sets de int (1-based)
    anu = {}
    for ses, lst in (anuladas_by_session or {}).items():
        try:
            anu[(ses or '').upper()] = set(int(n) for n in (lst or []))
        except Exception:
            anu[(ses or '').upper()] = set()

    raw = {m: {'correct': 0, 'total': 0} for m in MATERIAS}
    for entry in (distribucion or []):
        mat = entry['materia']
        ses = (entry['sesion'] or '').upper()
        try:
            ini = int(entry['inicio']); fin = int(entry['fin'])
        except Exception:
            continue
        ans = answers_by_session.get(ses, [])
        key = keys_by_session.get(ses, [])
        ses_anu = anu.get(ses, set())
        for i in range(ini - 1, fin):
            q_num = i + 1   # 1-based
            is_anulada = q_num in ses_anu
            if i < 0 or (i >= len(key) and not is_anulada):
                continue
            k = (key[i] or '').strip() if i < len(key) else ''
            a = (ans[i] if i < len(ans) else '').strip()
            # Si la pregunta está anulada → cuenta como correcta para todos
            if is_anulada:
                if mat not in raw:
                    raw[mat] = {'correct': 0, 'total': 0}
                raw[mat]['total']   += 1
                raw[mat]['correct'] += 1
                continue
            if k in ('', '?', '—'):
                continue
            if mat not in raw:
                raw[mat] = {'correct': 0, 'total': 0}
            raw[mat]['total'] += 1
            if a == k:
                raw[mat]['correct'] += 1
    out = {}
    for mat in MATERIAS:
        t = raw.get(mat, {}).get('total', 0)
        c = raw.get(mat, {}).get('correct', 0)
        out[mat] = round(c / t * 100, 2) if t else 0.0
    return out


def _find_simulacro_for_sheet(sheet_name):
    """Devuelve el nombre del simulacro que contiene esa hoja, o '' si no se encuentra."""
    try:
        sims = list_simulacros()
        for s in sims:
            if sheet_name in (s.get('sheets') or []):
                return s['nombre']
    except Exception:
        pass
    return ''


def _find_simulacro_for_sheets(sheet_names):
    """Igual que _find_simulacro_for_sheet pero con varias hojas (devuelve la primera coincidencia)."""
    for sn in sheet_names:
        n = _find_simulacro_for_sheet(sn)
        if n: return n
    return ''


def _grade_of_curso(curso, courses_list=None):
    if courses_list is None:
        courses_list = list_courses()
    c = (curso or '').strip()
    for cc in courses_list:
        if cc['name'] == c and cc.get('grado'):
            return cc['grado']
    return _grade_from_course_name(c)


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

    # Leer claves globales (legacy). Si están vacías, validar que haya
    # claves por grado configuradas para el simulacro.
    key_1s = _get_answer_key(ws_1s)
    key_2s = _get_answer_key(ws_2s)

    def _has_grade_keys(session):
        sim_name = _find_simulacro_for_sheets([sheet_1s, sheet_2s])
        if not sim_name: return False
        try:
            for kg in list_keys_grade():
                if (kg['simulacro'] == sim_name
                        and kg['sesion'] == session
                        and kg.get('count', 0) > 0):
                    return True
        except Exception:
            pass
        return False

    if not key_1s and not _has_grade_keys('1S'):
        return {'success': False,
                'error': (f'No hay clave de respuestas para "{sheet_1s}". '
                          'Configura la clave por grado en la pestaña Clave o la global de la hoja.')}
    if not key_2s and not _has_grade_keys('2S'):
        return {'success': False,
                'error': (f'No hay clave de respuestas para "{sheet_2s}". '
                          'Configura la clave por grado en la pestaña Clave o la global de la hoja.')}

    # Extraer estudiantes
    students_1s = _extract_students(ws_1s)
    students_2s = _extract_students(ws_2s)

    # Unir por ID — incluir estudiantes que estén en al menos una sesión
    all_ids = set(students_1s.keys()) | set(students_2s.keys())
    if not all_ids:
        return {'success': False, 'error': 'No se encontraron estudiantes.'}

    courses_list = list_courses()
    distribuciones = list_distribuciones()
    sim_name = _find_simulacro_for_sheets([sheet_1s, sheet_2s])
    keys_by_grade = {}  # {grado: {'1S': k1s, '2S': k2s}}
    anuladas_by_grade = {}  # {grado: {'1S': [...], '2S': [...]}}

    # Calcular resultados
    results = []
    for key in sorted(all_ids):
        s1 = students_1s.get(key)
        s2 = students_2s.get(key)
        s_any = s1 or s2
        name  = s_any.get('name','')
        sid   = s_any.get('id','') or (key if not key.startswith('_') else '')
        curso = ''
        if s1 and s1.get('curso'): curso = s1['curso']
        elif s2 and s2.get('curso'): curso = s2['curso']
        ans_1s = s1['answers'] if s1 else []
        ans_2s = s2['answers'] if s2 else []

        grado = _grade_of_curso(curso, courses_list)
        dist = distribuciones.get(('completo', grado))
        if not dist:
            dist = DEFAULT_DISTRIBUCIONES.get(('completo', grado))
        if not dist:
            dist = DEFAULT_DISTRIBUCIONES.get(('completo', '11'))

        # Claves por grado (si están registradas)
        if grado not in keys_by_grade:
            specific_1s = get_key_for_grade(sim_name, '1S', grado) if sim_name else []
            specific_2s = get_key_for_grade(sim_name, '2S', grado) if sim_name else []
            keys_by_grade[grado] = {
                '1S': specific_1s or key_1s,
                '2S': specific_2s or key_2s,
            }
        kfor = keys_by_grade[grado]

        # Preguntas anuladas para este grado
        if grado not in anuladas_by_grade:
            try:
                anuladas_by_grade[grado] = {
                    '1S': get_anuladas_for_grade(sim_name, '1S', grado) if sim_name else [],
                    '2S': get_anuladas_for_grade(sim_name, '2S', grado) if sim_name else [],
                }
            except Exception:
                anuladas_by_grade[grado] = {'1S': [], '2S': []}
        anu_for = anuladas_by_grade[grado]

        scores = _score_student_with_distribution(
            {'1S': ans_1s, '2S': ans_2s},
            {'1S': kfor['1S'], '2S': kfor['2S']}, dist,
            anuladas_by_session=anu_for)

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
    # Si no hay clave global, verificar si hay alguna clave por grado registrada
    if not key_m:
        sim_name = _find_simulacro_for_sheet(sheet_m)
        has_any = False
        if sim_name:
            try:
                for kg in list_keys_grade():
                    if (kg['simulacro'] == sim_name and kg['sesion'] == 'M'
                            and kg.get('count', 0) > 0):
                        has_any = True; break
            except Exception:
                pass
        if not has_any:
            return {'success': False,
                    'error': (f'No hay clave de respuestas para "{sheet_m}". '
                              'Configura la clave por grado en la pestaña Clave o '
                              'la clave global de la hoja.')}

    students = _extract_students(ws_m)
    if not students:
        return {'success': False,
                'error': f'No se encontraron estudiantes en "{sheet_m}". Verifica que la hoja tenga filas de estudiantes con nombre o ID.'}

    courses_list = list_courses()
    distribuciones = list_distribuciones()

    # Cache de claves y anuladas por grado: {grado: ...}
    keys_by_grade = {}
    anuladas_by_grade = {}
    sim_name = _find_simulacro_for_sheet(sheet_m)

    results = []
    for key in sorted(students.keys()):
        s     = students[key]
        name  = s['name']
        sid   = s.get('id', '')
        curso = s.get('curso', '')
        ans   = s['answers']

        # Distribución según el grado del estudiante
        grado = _grade_of_curso(curso, courses_list)
        dist = distribuciones.get(('media', grado))
        if not dist:
            dist = DEFAULT_DISTRIBUCIONES.get(('media', grado))
        if not dist:
            dist = DEFAULT_DISTRIBUCIONES.get(('media', '10'))

        # Clave: si hay una específica para (simulacro, M, grado), úsala;
        # sino usa la global del sheet (key_m)
        if grado not in keys_by_grade:
            specific = get_key_for_grade(sim_name, 'M', grado) if sim_name else []
            keys_by_grade[grado] = specific or key_m
        key_for_student = keys_by_grade[grado]

        # Preguntas anuladas para este grado (sesión M)
        if grado not in anuladas_by_grade:
            try:
                anuladas_by_grade[grado] = (
                    get_anuladas_for_grade(sim_name, 'M', grado) if sim_name else [])
            except Exception:
                anuladas_by_grade[grado] = []
        anu_m = anuladas_by_grade[grado]

        scores = _score_student_with_distribution(
            {'M': ans}, {'M': key_for_student}, dist,
            anuladas_by_session={'M': anu_m})

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
