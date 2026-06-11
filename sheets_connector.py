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
import threading
from datetime import datetime

# ── Locks de concurrencia (gunicorn corre 4 threads en 1 worker) ───────────
# _CACHE_LOCK protege los caches globales de lectura.
# _WRITE_LOCK serializa las operaciones leer-luego-escribir (upserts por ID)
# para que dos guardados simultáneos del mismo estudiante no dupliquen filas.
_CACHE_LOCK = threading.RLock()
_WRITE_LOCK = threading.RLock()

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
ROSTER_SHEET       = 'RosterPersonalizado'
ESTUDIANTES_SHEET  = 'Estudiantes'   # roster GENERAL (no atado a simulacro)

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

# Grado 10° — Media con 20 preguntas por asignatura (total 100).
# Mat 1-20 | Lect 21-40 | Soc 41-60 | Nat 61-80 | Ing 81-100
_MEDIA_GRADO_10 = [
    {'materia':'Matematica',      'sesion':'M', 'inicio':1,   'fin':20},
    {'materia':'Lectura Critica', 'sesion':'M', 'inicio':21,  'fin':40},
    {'materia':'Sociales',        'sesion':'M', 'inicio':41,  'fin':60},
    {'materia':'Naturales',       'sesion':'M', 'inicio':61,  'fin':80},
    {'materia':'Ingles',          'sesion':'M', 'inicio':81,  'fin':100},
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
    # Media: 5 bloques de 25 preguntas para los grados 6° a 9°
    ('media', '6'):  _MEDIA_DEFAULT,
    ('media', '7'):  _MEDIA_DEFAULT,
    ('media', '8'):  _MEDIA_DEFAULT,
    ('media', '9'):  _MEDIA_DEFAULT,
    # Grado 10°: 20 por asignatura (total 100)
    ('media', '10'): _MEDIA_GRADO_10,
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


_SPREADSHEET_CACHE = {'ss': None, 'ts': 0}

# Cache de lectura por TTL (Time To Live) para reducir llamadas a Sheets
# y mejorar la capacidad de concurrencia del portal estudiantil.
_READ_CACHE = {}   # key → (timestamp, value)
_READ_CACHE_TTL = 60   # segundos. Conservador: cambios se reflejan en <1 min.


def _cached_read(key, fn, ttl=None):
    """Memoriza el resultado de fn() durante ttl segundos (thread-safe)."""
    import time
    if ttl is None: ttl = _READ_CACHE_TTL
    now = time.time()
    with _CACHE_LOCK:
        cached = _READ_CACHE.get(key)
        if cached and (now - cached[0]) < ttl:
            return cached[1]
    # fn() fuera del lock: una lectura lenta a Sheets no debe bloquear el
    # resto del cache (a lo sumo dos threads leen lo mismo una vez).
    val = fn()
    with _CACHE_LOCK:
        _READ_CACHE[key] = (now, val)
    return val


def invalidate_read_cache(prefix=''):
    """Limpia el cache. Si prefix se pasa, solo borra claves con ese prefijo."""
    with _CACHE_LOCK:
        if not prefix:
            _READ_CACHE.clear()
        else:
            for k in [k for k in _READ_CACHE if k.startswith(prefix)]:
                del _READ_CACHE[k]

def _open_spreadsheet(force_fresh: bool = False):
    """
    Abre el Sheets. Cachea el objeto durante 30 s para reducir llamadas a
    la Sheets API y evitar 429 (quota exceeded) cuando una sola operacion
    invoca varios helpers que abren la hoja.
    """
    import time
    now = time.time()
    with _CACHE_LOCK:
        cached = _SPREADSHEET_CACHE.get('ss')
        cached_ts = _SPREADSHEET_CACHE.get('ts', 0)
        if not force_fresh and cached is not None and (now - cached_ts) < 30:
            return cached
    client = _get_sheets_client()
    try:
        ss = client.open_by_key(SPREADSHEET_ID)
        with _CACHE_LOCK:
            _SPREADSHEET_CACHE['ss'] = ss
            _SPREADSHEET_CACHE['ts'] = now
        return ss
    except Exception as e:
        msg = str(e)
        # Detectar quota / rate limit
        if any(k in msg for k in ('429', 'Quota', 'quota', 'Rate', 'rate')):
            raise ValueError(
                "Google Sheets reporta limite de lecturas excedido (429). "
                "Espera unos 30-60 segundos y vuelve a intentar.")
        raise ValueError(
            f"No se pudo abrir el Sheets '{SPREADSHEET_ID}'. "
            f"Detalle: {msg[:200]}")


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

    # ── Validación: el nombre NO puede terminar en sufijos que coincidan con
    # los que se añaden a las hojas por sesión, porque generaría hojas con
    # nombres dobles tipo "Mayo 2026 Mañana M M" que confunden al sistema.
    bad_suffixes = (' M', ' 1S', ' 2S', '- M', '-M', '- 1S', '- 2S')
    n_upper = nombre.upper()
    for suf in bad_suffixes:
        if n_upper.endswith(suf.upper()):
            return {'success': False,
                    'error': (f'El nombre no debe terminar en "{suf.strip()}" '
                              f'porque genera conflictos con las hojas de sesión. '
                              f'Usa un nombre como "Mayo 2026 IETECI Mañana" '
                              f'(el sistema añade la sesión automáticamente).')}
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

    distribuciones = list_distribuciones(simulacro_nombre)

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

    # Pre-cargar TODAS las claves por grado y anuladas del simulacro en UNA
    # lectura para evitar múltiples llamadas a Sheets API que disparan 429.
    keys_idx = {}
    anu_idx  = {}
    try:
        ws_kg = spreadsheet.worksheet(KEYS_GRADE_SHEET)
        for r in ws_kg.get_all_values()[1:]:
            if len(r) >= 3 and (r[0] or '').strip() == simulacro_nombre:
                ses = (r[1] or '').strip().upper()
                gr  = str(r[2] or '').strip()
                ans = list(r[3:])
                while ans and not (ans[-1] or '').strip(): ans.pop()
                keys_idx[(ses, gr)] = [(a or '').strip() for a in ans]
    except Exception as e:
        print(f'[analisis] ADVERTENCIA: no se pudieron leer las claves por '
              f'grado de "{simulacro_nombre}": {e}', flush=True)
        keys_idx = {}
    try:
        for a in list_anuladas():
            if a['simulacro'] == simulacro_nombre:
                anu_idx[(a['sesion'], a['grado'])] = set(a['preguntas'])
    except Exception as e:
        print(f'[analisis] ADVERTENCIA: no se pudieron leer las anuladas de '
              f'"{simulacro_nombre}": {e}', flush=True)
        anu_idx = {}

    # Cache de claves por (session, grado)
    keys_cache = {}
    def key_for(session, grado):
        ck = (session, grado)
        if ck in keys_cache: return keys_cache[ck]
        specific = keys_idx.get(ck, [])
        keys_cache[ck] = specific or keys_global.get(session, [])
        return keys_cache[ck]

    # Cache de anuladas por (session, grado) — sets de int (1-based)
    anu_cache = {}
    def anu_for(session, grado):
        ck = (session, grado)
        if ck in anu_cache: return anu_cache[ck]
        anu_cache[ck] = anu_idx.get(ck, set())
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

    # Diagnóstico: anuladas aplicadas por (sesion, grado)
    anuladas_applied = {}
    for (ses_a, grado_a), nums in anu_cache.items():
        if not nums: continue
        anuladas_applied.setdefault(grado_a, {})[ses_a] = sorted(nums)

    return {
        'success':         True,
        'simulacro':       simulacro_nombre,
        'tipo':            sim['tipo'],
        'questions':       default_questions,
        'grades_data':     grades_data,
        'anuladas_applied': anuladas_applied,
    }


# ── Distribución de preguntas por (tipo, grado) ──────────────────────
def _ensure_distribucion_sheet():
    """
    Hoja 'Distribucion': Tipo, Grado, Materia, Sesion, Inicio, Fin, Simulacro.
    La columna Simulacro (G) es opcional: vacía = distribución POR DEFECTO
    (aplica a todos los simulacros); con nombre = específica de ese simulacro
    y tiene prioridad sobre la default.
    """
    ss = _open_spreadsheet()
    fresh = False
    try:
        ws = ss.worksheet(DISTRIBUCION_SHEET)
        header = ws.row_values(1) or []
        if (len(header) < 7 or header[0].strip().lower() != 'tipo'
                or (header[6] or '').strip().lower() != 'simulacro'):
            ws.update('A1:G1',
                      [['Tipo','Grado','Materia','Sesion','Inicio','Fin','Simulacro']],
                      value_input_option='RAW')
    except Exception:
        ws = ss.add_worksheet(title=DISTRIBUCION_SHEET, rows=200, cols=8)
        ws.update('A1:G1',
                  [['Tipo','Grado','Materia','Sesion','Inicio','Fin','Simulacro']],
                  value_input_option='RAW')
        try:
            ws.format('A1:G1', {
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


def _read_distribucion_rows() -> list:
    """Lee la hoja Distribucion. Devuelve [(simulacro, tipo, grado, entry)]."""
    try:
        ws = _ensure_distribucion_sheet()
        data = ws.get_all_values()[1:]
        out = []
        for r in data:
            if len(r) < 6: continue
            tipo  = (r[0] or '').strip().lower()
            grado = (r[1] or '').strip()
            mat   = (r[2] or '').strip()
            ses   = (r[3] or '').strip().upper()
            sim   = (r[6] or '').strip() if len(r) >= 7 else ''
            try:
                ini = int(r[4]); fin = int(r[5])
            except Exception:
                continue
            if not tipo or not grado or not mat or ini < 1 or fin < ini:
                continue
            out.append((sim, tipo, grado,
                        {'materia': mat, 'sesion': ses,
                         'inicio': ini, 'fin': fin}))
        return out
    except Exception:
        return []


def list_distribuciones(simulacro: str = '') -> dict:
    """
    Retorna el mapa EFECTIVO {(tipo, grado): [{materia, sesion, inicio, fin}]}
    para un simulacro: sus distribuciones específicas tienen prioridad y,
    donde no haya, se usan las por defecto (columna Simulacro vacía).
    Sin argumento devuelve solo las por defecto (compatibilidad histórica).
    """
    simulacro = (simulacro or '').strip()
    base = {}      # por defecto (sim == '')
    specific = {}  # del simulacro pedido
    for sim, tipo, grado, entry in _read_distribucion_rows():
        key = (tipo, grado)
        if not sim:
            base.setdefault(key, []).append(entry)
        elif simulacro and sim == simulacro:
            specific.setdefault(key, []).append(entry)
    result = dict(base)
    result.update(specific)   # lo específico del simulacro pisa lo default
    return result


def list_distribuciones_json() -> list:
    """Lista serializable: [{simulacro, tipo, grado, materias:[...]}]."""
    grouped = {}
    for sim, tipo, grado, entry in _read_distribucion_rows():
        grouped.setdefault((sim, tipo, grado), []).append(entry)
    out = []
    for (sim, tipo, grado), entries in grouped.items():
        out.append({'simulacro': sim, 'tipo': tipo, 'grado': grado,
                    'materias': entries})
    out.sort(key=lambda x: (x['simulacro'], x['tipo'],
                            int(x['grado']) if x['grado'].isdigit() else 99))
    return out


def save_distribucion(tipo: str, grado: str, materias: list,
                      simulacro: str = '') -> dict:
    """
    Reemplaza la distribución de (simulacro, tipo, grado).
    simulacro vacío = la distribución POR DEFECTO de (tipo, grado).
    """
    tipo  = (tipo or '').strip().lower()
    grado = str(grado or '').strip()
    simulacro = (simulacro or '').strip()
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
        cleaned.append([tipo, grado, mat, ses, ini, fin, simulacro])

    try:
      with _WRITE_LOCK:
        ws = _ensure_distribucion_sheet()
        all_rows = ws.get_all_values()
        i = len(all_rows)
        while i >= 2:
            row = all_rows[i-1] if i-1 < len(all_rows) else []
            row_sim = (row[6] or '').strip() if len(row) >= 7 else ''
            if (len(row) >= 2 and (row[0] or '').strip().lower() == tipo
                and str(row[1] or '').strip() == grado
                and row_sim == simulacro):
                ws.delete_rows(i)
            i -= 1
        ws.append_rows(cleaned, value_input_option='RAW')
        return {'success': True, 'distribuciones': list_distribuciones_json()}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def delete_distribucion(tipo: str, grado: str, simulacro: str) -> dict:
    """Elimina la distribución específica de un simulacro (vuelve al default).
    Solo aplica a filas con simulacro NO vacío."""
    tipo  = (tipo or '').strip().lower()
    grado = str(grado or '').strip()
    simulacro = (simulacro or '').strip()
    if not simulacro:
        return {'success': False,
                'error': 'Solo se pueden eliminar distribuciones de un simulacro.'}
    try:
      with _WRITE_LOCK:
        ws = _ensure_distribucion_sheet()
        all_rows = ws.get_all_values()
        i = len(all_rows)
        removed = 0
        while i >= 2:
            row = all_rows[i-1] if i-1 < len(all_rows) else []
            row_sim = (row[6] or '').strip() if len(row) >= 7 else ''
            if (len(row) >= 2 and (row[0] or '').strip().lower() == tipo
                and str(row[1] or '').strip() == grado
                and row_sim == simulacro):
                ws.delete_rows(i)
                removed += 1
            i -= 1
        return {'success': True, 'removed': removed,
                'distribuciones': list_distribuciones_json()}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_distribucion_for(tipo: str, grado: str, simulacro: str = '') -> list:
    """Distribución efectiva para (simulacro, tipo, grado), con fallbacks."""
    tipo  = (tipo or '').strip().lower()
    grado = str(grado or '').strip()
    d = list_distribuciones(simulacro)
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

    Nota: invalida el cache de lectura de hojas Resultados/respuestas
    para que el portal y los reportes vean los datos frescos.
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

    # ── Construir fila del estudiante ─────────────────────────────────────────
    now      = datetime.now()
    row_data = [now.strftime('%Y-%m-%d'), now.strftime('%H:%M:%S'),
                student_name, exam_id]
    row_data += [a if a != '?' else '—' for a in answers]
    row_data += [detected,
                 correct if key_n > 0 else '',
                 pct_str]
    # Agregar curso al final si la columna existe
    if curso_col >= 0:
        while len(row_data) < curso_col:
            row_data.append('')
        row_data.append(curso or '')

    # Sección leer-luego-escribir bajo lock: sin él, dos guardados
    # simultáneos del mismo ID verían ambos "no existe" y duplicarían la fila.
    with _WRITE_LOCK:
        # ── Eliminar fila de TOTALES existente antes de agregar el estudiante ─
        all_rows = worksheet.get_all_values()
        for i, row in enumerate(all_rows):
            if row and len(row) > 2 and row[2] == '--- TOTALES ---':
                worksheet.delete_rows(i + 1)
                break

        # ── DEDUPE: si ya hay fila con el mismo ID, ACTUALIZAR en vez de agregar
        #    El ID está en col D (índice 3). Omite fila 1 (header) y 2 (clave).
        existing_row = None
        if exam_id:
            for i, row in enumerate(all_rows):
                if i < 2: continue
                if len(row) > 3 and (row[3] or '').strip() == str(exam_id).strip():
                    existing_row = i + 1   # 1-based
                    break

        was_updated = False
        if existing_row:
            # UPDATE: usar la fila existente para preservar orden histórico
            end_col = _col_letter(len(row_data) - 1)
            worksheet.update(f'A{existing_row}:{end_col}{existing_row}',
                             [row_data], value_input_option='RAW')
            row_num = existing_row
            was_updated = True
        else:
            worksheet.append_row(row_data, value_input_option='RAW')
            row_num = len(worksheet.col_values(1))

        _update_totals_row(worksheet, n_questions)

    return {
        'row':     row_num,
        'correct': correct if key_n > 0 else None,
        'pct':     pct_str,
        'updated': was_updated,
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


def _compute_nat_subscores_grade10(ans, key, anuladas_set=None):
    """
    Para grado 10° Media: calcula sub-puntajes
      - Química:  P61-P80  (20 preguntas)
      - Física:   P81-P100 (20 preguntas)
    Devuelve {'quim': float, 'fis': float} en escala 0-100.
    Aplica la misma lógica que el scoring oficial: las preguntas anuladas
    cuentan como correctas para todos.
    """
    anu = anuladas_set or set()
    try: anu = set(int(n) for n in anu)
    except Exception: anu = set()

    def block_score(ini, fin):
        total = 0; correct = 0
        for i in range(ini - 1, fin):
            q_num = i + 1
            is_anu = q_num in anu
            if is_anu:
                total += 1; correct += 1
                continue
            k = (key[i] if i < len(key) else '').strip()
            if not k or k in ('?', '—', '-'):
                continue
            a = (ans[i] if i < len(ans) else '').strip()
            total += 1
            if a == k: correct += 1
        return round(correct / total * 100, 2) if total else 0.0

    return {'quim': block_score(61, 80), 'fis': block_score(81, 100)}


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


# ─────────────────────────────────────────────────────────────────────────────
# RESULTADOS — núcleo común para SIPAGRE (1S+2S) y M SIPAGRE (M)
# ─────────────────────────────────────────────────────────────────────────────
def _generate_results_core(session_sheets, tipo, results_sheet,
                           fallback_grado, header_color):
    """
    Núcleo común de generación de resultados (antes duplicado en
    generate_sipagre_results y generate_msipagre_results).

    session_sheets: lista de (sesion, nombre_hoja) en orden,
                    p.ej. [('1S','...'), ('2S','...')] o [('M','...')].
    tipo:           'completo' | 'media'. Decide la distribución y, en media,
                    los subscores Química/Física para grado 10°.
    fallback_grado: grado cuya distribución default se usa como último recurso.
    header_color:   color de fondo del encabezado en la hoja de resultados.
    """
    spreadsheet = _open_spreadsheet()

    # Abrir hojas fuente (error claro por cada una, en orden)
    worksheets = {}
    for ses, name in session_sheets:
        try:
            worksheets[ses] = spreadsheet.worksheet(name)
        except Exception:
            return {'success': False, 'error': f'Hoja "{name}" no encontrada.'}

    sheet_names = [name for _, name in session_sheets]
    sim_name = _find_simulacro_for_sheets(sheet_names)

    # Claves globales (fila 2 de cada hoja). Si faltan, debe existir al menos
    # una clave por grado para esa sesión.
    keys_global = {ses: _get_answer_key(ws) for ses, ws in worksheets.items()}

    def _has_grade_keys(session):
        if not sim_name:
            return False
        try:
            for kg in list_keys_grade():
                if (kg['simulacro'] == sim_name
                        and kg['sesion'] == session
                        and kg.get('count', 0) > 0):
                    return True
        except Exception:
            pass
        return False

    for ses, name in session_sheets:
        if not keys_global[ses] and not _has_grade_keys(ses):
            return {'success': False,
                    'error': (f'No hay clave de respuestas para "{name}". '
                              'Configura la clave por grado en la pestaña '
                              'Clave o la clave global de la hoja.')}

    # Extraer estudiantes por sesión y unir por ID
    students_by_ses = {ses: _extract_students(ws)
                       for ses, ws in worksheets.items()}
    all_ids = set()
    for d in students_by_ses.values():
        all_ids |= set(d.keys())
    if not all_ids:
        if len(session_sheets) == 1:
            return {'success': False,
                    'error': (f'No se encontraron estudiantes en '
                              f'"{sheet_names[0]}". Verifica que la hoja tenga '
                              'filas de estudiantes con nombre o ID.')}
        return {'success': False, 'error': 'No se encontraron estudiantes.'}

    courses_list = list_courses()
    distribuciones = list_distribuciones(sim_name)

    # Pre-cargar TODAS las claves por grado y anuladas en una sola lectura
    # de cada hoja (evita N llamadas a Sheets API que disparan 429 quota).
    keys_idx = {}   # (ses, grado) -> answers list (1-based completas)
    anu_idx  = {}   # (ses, grado) -> [pregunta1, ...]
    if sim_name:
        try:
            ws_kg = _open_spreadsheet().worksheet(KEYS_GRADE_SHEET)
            for r in ws_kg.get_all_values()[1:]:
                if len(r) >= 3 and (r[0] or '').strip() == sim_name:
                    ses = (r[1] or '').strip().upper()
                    gr  = str(r[2] or '').strip()
                    ans = list(r[3:])
                    while ans and not (ans[-1] or '').strip(): ans.pop()
                    keys_idx[(ses, gr)] = [(a or '').strip() for a in ans]
        except Exception as e:
            # No silenciar: sin claves por grado se calificaría con la clave
            # global, dando puntajes distintos sin que nadie lo note.
            print(f'[resultados] ADVERTENCIA: no se pudieron leer las claves '
                  f'por grado de "{sim_name}": {e}', flush=True)
            keys_idx = {}
        try:
            for a in list_anuladas():
                if a['simulacro'] == sim_name:
                    anu_idx[(a['sesion'], a['grado'])] = a['preguntas']
        except Exception as e:
            print(f'[resultados] ADVERTENCIA: no se pudieron leer las '
                  f'anuladas de "{sim_name}": {e}', flush=True)
            anu_idx = {}

    sessions = [ses for ses, _ in session_sheets]
    with_quimfis = (tipo == 'media')
    keys_by_grade = {}      # {grado: {ses: clave}}
    anuladas_by_grade = {}  # {grado: {ses: [preguntas]}}

    results = []
    for key in sorted(all_ids):
        per_ses = {ses: students_by_ses[ses].get(key) for ses in sessions}
        s_any = next((per_ses[ses] for ses in sessions if per_ses[ses]), None)
        name  = s_any.get('name', '')
        sid   = s_any.get('id', '')
        curso = next((per_ses[ses]['curso'] for ses in sessions
                      if per_ses[ses] and per_ses[ses].get('curso')), '')
        answers_by_ses = {ses: (per_ses[ses]['answers'] if per_ses[ses] else [])
                          for ses in sessions}

        grado = _grade_of_curso(curso, courses_list)
        dist = (distribuciones.get((tipo, grado))
                or DEFAULT_DISTRIBUCIONES.get((tipo, grado))
                or DEFAULT_DISTRIBUCIONES.get((tipo, fallback_grado)))

        # Claves y anuladas por grado (lookup en memoria, sin tocar Sheets)
        if grado not in keys_by_grade:
            keys_by_grade[grado] = {
                ses: (keys_idx.get((ses, grado), []) or keys_global[ses])
                for ses in sessions}
        kfor = keys_by_grade[grado]
        if grado not in anuladas_by_grade:
            anuladas_by_grade[grado] = {
                ses: anu_idx.get((ses, grado), []) for ses in sessions}
        anu_for = anuladas_by_grade[grado]

        scores = _score_student_with_distribution(
            answers_by_ses, kfor, dist, anuladas_by_session=anu_for)

        mat  = int(round(scores['Matematica']))
        lect = int(round(scores['Lectura Critica']))
        soc  = int(round(scores['Sociales']))
        nat  = int(round(scores['Naturales']))
        ing  = int(round(scores['Ingles']))
        # Sistema ICFES Saber 11: pesos 3-3-3-3-1 (Mat, Lect, Soc, Nat = 3
        # cada una; Ing = 1). Suma de pesos = 13. Multiplicamos por 5
        # para escalar a 0-500. Max general = 500.
        general = int(round(5 * ((mat*3 + lect*3 + soc*3 + nat*3 + ing*1) / 13)))

        entry = {
            'id':       sid,
            'name':     name,
            'curso':    curso,
            'mat':      mat,
            'lect':     lect,
            'soc':      soc,
            'nat':      nat,
            'ing':      ing,
            'general':  general,
        }
        if with_quimfis:
            # Solo grado 10°: descomponer Naturales en Química (P61-P80) y
            # Física (P81-P100). Para otros grados queda vacío.
            quim, fis = '', ''
            if str(grado) == '10':
                sub = _compute_nat_subscores_grade10(
                    answers_by_ses[sessions[0]], kfor[sessions[0]],
                    set(anu_for[sessions[0]] or []))
                quim = int(round(sub['quim']))
                fis  = int(round(sub['fis']))
            entry['quim']  = quim
            entry['fis']   = fis
            entry['grado'] = grado
            # 'general' debe quedar después de ing en el orden histórico de
            # la fila, pero en el dict el orden no afecta la escritura.
            entry['type']  = 'M'
        results.append(entry)

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

    header = [
        'ID Estudiante', 'Nombre', 'Curso',
        'Matematica', 'Lectura Critica', 'Sociales',
        'Naturales', 'Ingles', 'Puntaje General',
    ]
    if with_quimfis:
        header += ['Quimica (10°)', 'Fisica (10°)']
    end_col = _col_letter(len(header) - 1)
    ws_res.update(f'A1:{end_col}1', [header], value_input_option='RAW')

    rows = []
    for r in results:
        row = [r['id'], r['name'], r.get('curso', ''),
               r['mat'], r['lect'], r['soc'],
               r['nat'], r['ing'], r['general']]
        if with_quimfis:
            row += [r.get('quim', ''), r.get('fis', '')]
        rows.append(row)

    if rows:
        end_row = len(rows) + 1
        ws_res.update(f'A2:{end_col}{end_row}', rows, value_input_option='RAW')

        # Fila de promedios
        n = len(results)
        avg = {k: int(round(sum(r[k] for r in results) / n))
               for k in ['mat', 'lect', 'soc', 'nat', 'ing', 'general']}
        avg_vals = ['', 'PROMEDIO', '', avg['mat'], avg['lect'], avg['soc'],
                    avg['nat'], avg['ing'], avg['general']]
        if with_quimfis:
            # Promedio Quím/Fís solo sobre los de 10°
            only10 = [r for r in results
                      if isinstance(r.get('quim'), int)
                      and isinstance(r.get('fis'), int)]
            avg_vals += [
                (int(round(sum(r['quim'] for r in only10) / len(only10)))
                 if only10 else ''),
                (int(round(sum(r['fis'] for r in only10) / len(only10)))
                 if only10 else ''),
            ]
        avg_row = end_row + 1
        ws_res.update(f'A{avg_row}:{end_col}{avg_row}', [avg_vals],
                      value_input_option='RAW')
        try:
            ws_res.format(f'A{avg_row}:{end_col}{avg_row}', {
                'backgroundColor': {'red': 0.93, 'green': 0.93, 'blue': 0.93},
                'textFormat': {'bold': True},
                'horizontalAlignment': 'CENTER'
            })
        except Exception:
            pass

    # Formato encabezado (histórico: siempre A1:I1, también en media)
    try:
        ws_res.format('A1:I1', {
            'backgroundColor': header_color,
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


def generate_sipagre_results(sheet_1s: str, sheet_2s: str,
                             results_sheet: str = 'Resultados SIPAGRE') -> dict:
    """Combina las sesiones 1S y 2S por ID de estudiante (tipo 'completo')."""
    return _generate_results_core(
        [('1S', sheet_1s), ('2S', sheet_2s)], 'completo', results_sheet,
        fallback_grado='11',
        header_color={'red': 0.13, 'green': 0.27, 'blue': 0.53})


def generate_msipagre_results(sheet_m: str = 'M SIPAGRE',
                              results_sheet: str = 'Resultados M SIPAGRE') -> dict:
    """Una sola jornada (sesión M, tipo 'media') con Quím/Fís para 10°."""
    return _generate_results_core(
        [('M', sheet_m)], 'media', results_sheet,
        fallback_grado='10',
        header_color={'red': 0.30, 'green': 0.39, 'blue': 0.85})




# ═══════════════════════════════════════════════════════════════════════════
# ROSTER PERSONALIZADO (para hojas con QR — mapea ID -> nombre/curso)
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_roster_sheet():
    """Crea la hoja Roster si no existe. Cols: Simulacro, Curso, ID, Nombre."""
    ss = _open_spreadsheet()
    try:
        ws = ss.worksheet(ROSTER_SHEET)
        header = ws.row_values(1) or []
        if len(header) < 4 or header[0].strip().lower() != 'simulacro':
            ws.update('A1:D1', [['Simulacro', 'Curso', 'ID', 'Nombre']],
                      value_input_option='RAW')
    except Exception:
        ws = ss.add_worksheet(title=ROSTER_SHEET, rows=2000, cols=4)
        ws.update('A1:D1', [['Simulacro', 'Curso', 'ID', 'Nombre']],
                  value_input_option='RAW')
        try:
            ws.format('A1:D1', {
                'backgroundColor': {'red': 0.12, 'green': 0.23, 'blue': 0.54},
                'textFormat': {'bold': True,
                               'foregroundColor': {'red':1,'green':1,'blue':1}},
            })
        except Exception:
            pass
    return ws


def save_roster_for_simulacro(simulacro_safe: str, students: list) -> dict:
    """
    Reemplaza el roster del simulacro indicado (por nombre 'safe', formato
    igual al del QR). students: lista de dicts con 'curso', 'id', 'nombre'.
    """
    simulacro_safe = (simulacro_safe or '').strip()
    if not simulacro_safe:
        return {'success': False, 'error': 'simulacro requerido'}
    if not isinstance(students, list):
        return {'success': False, 'error': 'students debe ser lista'}
    try:
      # Borrar-luego-insertar serializado (ver _WRITE_LOCK).
      with _WRITE_LOCK:
        ws = _ensure_roster_sheet()
        all_rows = ws.get_all_values()
        # Borrar filas existentes de este simulacro (de abajo arriba)
        for i in range(len(all_rows) - 1, 0, -1):
            r = all_rows[i]
            if r and len(r) > 0 and (r[0] or '').strip() == simulacro_safe:
                try:
                    ws.delete_rows(i + 1)  # 1-based
                except Exception:
                    pass
        # Insertar nuevos
        rows_to_add = []
        for s in students:
            curso = str(s.get('curso', '')).strip()
            ident = str(s.get('id', '')).strip()
            nombre = str(s.get('nombre', '')).strip().upper()
            if not (ident and (curso or nombre)):
                continue
            rows_to_add.append([simulacro_safe, curso, ident, nombre])
        if rows_to_add:
            ws.append_rows(rows_to_add, value_input_option='RAW')
        invalidate_read_cache('roster:')
        return {'success': True, 'count': len(rows_to_add)}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def get_student_from_roster(simulacro_safe: str, student_id: str) -> dict:
    """
    Busca un estudiante por ID dentro del roster de un simulacro.
    Devuelve {'curso', 'id', 'nombre'} o None.
    """
    simulacro_safe = (simulacro_safe or '').strip().upper()
    student_id = (student_id or '').strip()
    if not simulacro_safe or not student_id:
        return None
    try:
        # Cacheado 60s: cada QR escaneado consulta el roster; sin caché eso
        # es una lectura completa de la hoja por escaneo (gasta cuota API).
        rows = _cached_read('roster:all',
                            lambda: _ensure_roster_sheet().get_all_values())
        for r in rows[1:]:
            if (len(r) >= 4
                and (r[0] or '').strip().upper() == simulacro_safe
                and (r[2] or '').strip() == student_id):
                return {
                    'curso':  (r[1] or '').strip(),
                    'id':     (r[2] or '').strip(),
                    'nombre': (r[3] or '').strip(),
                }
    except Exception:
        pass
    return None


def list_roster(simulacro_safe: str = '') -> list:
    """Lista el roster, opcionalmente filtrado por simulacro_safe."""
    try:
        ws = _ensure_roster_sheet()
        out = []
        for r in ws.get_all_values()[1:]:
            if len(r) < 4: continue
            sim = (r[0] or '').strip()
            if simulacro_safe and sim != simulacro_safe: continue
            out.append({
                'simulacro': sim,
                'curso':     (r[1] or '').strip(),
                'id':        (r[2] or '').strip(),
                'nombre':    (r[3] or '').strip(),
            })
        return out
    except Exception:
        return []


# ═══════════════════════════════════════════════════════════════════════════
# PORTAL ESTUDIANTIL — Consulta de resultados por ID
# ═══════════════════════════════════════════════════════════════════════════

def get_student_results_all_simulacros(student_id: str) -> list:
    """
    Busca al estudiante (por ID) en TODAS las hojas de resultados de TODOS
    los simulacros existentes.

    Devuelve una lista (puede tener varios elementos si está en varios
    simulacros):
      [{
        'simulacro':  'Mayo 2026 IETAGRO',
        'tipo':       'completo' | 'media',
        'fecha':      '2026-05-10',
        'sheet':      'Resultados Mayo 2026 IETAGRO',
        'nombre':     'MARIA JOSE GUTIERREZ',
        'curso':      '1101',
        'puntajes':   {'mat':, 'lect':, 'soc':, 'nat':, 'ing':, 'gen':,
                       'quim':?, 'fis':?},
        'avg_curso':  {...},     # promedio del curso (comparación)
        'avg_grado':  {...},     # promedio del grado (comparación)
      }, ...]
    """
    student_id = str(student_id or '').strip()
    if not student_id:
        return []
    try:
        sims = list_simulacros()
    except Exception:
        sims = []

    spreadsheet = _open_spreadsheet()

    # Indexar simulacros por nombre para mapear hojas (incluso huérfanas)
    sim_by_name = {(s.get('nombre') or '').strip(): s for s in (sims or [])}

    # Cache de la lista de hojas Resultados (cambia raramente, TTL 5min)
    def _list_result_sheets():
        try:
            return [ws.title for ws in spreadsheet.worksheets()
                    if (ws.title or '').lower().startswith('resultados ')]
        except Exception:
            sheets = []
            for s in (sims or []):
                rs = s.get('results') or []
                if isinstance(rs, str): rs = [rs]
                sheets.extend(rs)
            return sheets
    candidate_sheets = _cached_read('result_sheets:list', _list_result_sheets,
                                     ttl=300)

    out = []

    for res_sheet in candidate_sheets:
        # Inferir simulacro desde el nombre de la hoja
        title_no_prefix = res_sheet[len('Resultados '):].strip()
        sim = sim_by_name.get(title_no_prefix)
        if sim is None:
            # quitar sufijos de sesión (M / 1S / 2S)
            for suf in (' M', ' 1S', ' 2S', ' - M', ' - 1S', ' - 2S'):
                if title_no_prefix.endswith(suf):
                    cand = title_no_prefix[:-len(suf)].strip()
                    if cand in sim_by_name:
                        sim = sim_by_name[cand]
                        break
        if sim is None:
            # Simulacro no registrado (huérfano) — usamos placeholder
            sim = {'nombre': title_no_prefix, 'tipo': '', 'fecha': ''}

        if True:
            # Cache de las filas de cada hoja Resultados (TTL 60s)
            def _read_sheet_rows(sheet_name=res_sheet):
                try:
                    return spreadsheet.worksheet(sheet_name).get_all_values()
                except Exception:
                    return []
            rows = _cached_read(f'result_sheet:rows:{res_sheet}',
                                _read_sheet_rows, ttl=60)
            if not rows or len(rows) < 2:
                continue

            # Buscar fila del estudiante por ID (columna A, índice 0)
            student_row = None
            for row in rows[1:]:
                if not row: continue
                rid = (row[0] or '').strip() if len(row) > 0 else ''
                if rid == student_id:
                    student_row = row
                    break
            if student_row is None:
                continue

            # Parsear puntajes (cols: A=ID, B=Nombre, C=Curso, D=Mat,
            # E=Lect, F=Soc, G=Nat, H=Ing, I=General, J=Quim, K=Fis)
            def _num(v):
                try: return int(round(float((v or '').strip())))
                except: return None
            nombre = (student_row[1] or '').strip() if len(student_row) > 1 else ''
            curso  = (student_row[2] or '').strip() if len(student_row) > 2 else ''
            puntajes = {
                'mat':  _num(student_row[3]) if len(student_row) > 3 else None,
                'lect': _num(student_row[4]) if len(student_row) > 4 else None,
                'soc':  _num(student_row[5]) if len(student_row) > 5 else None,
                'nat':  _num(student_row[6]) if len(student_row) > 6 else None,
                'ing':  _num(student_row[7]) if len(student_row) > 7 else None,
                'gen':  _num(student_row[8]) if len(student_row) > 8 else None,
                'quim': _num(student_row[9])  if len(student_row) > 9  else None,
                'fis':  _num(student_row[10]) if len(student_row) > 10 else None,
            }

            # Calcular promedios del curso y del grado (anónimos)
            avg_curso = {k: None for k in puntajes}
            avg_grado = {k: None for k in puntajes}
            cnt_curso = 0
            cnt_grado = 0
            grado_stu = _grade_from_course_name(curso)

            def _accum(target, source_row):
                for i, key in enumerate(
                        ['mat','lect','soc','nat','ing','gen','quim','fis']):
                    col_idx = 3 + i  # mat=col D
                    if col_idx < len(source_row):
                        v = _num(source_row[col_idx])
                        if v is not None:
                            if target.get(key) is None:
                                target[key] = [0, 0]   # [suma, count]
                            target[key][0] += v
                            target[key][1] += 1

            # Acumuladores temporales (luego divide). NO descartar filas sin
            # ID — son estudiantes válidos en hojas anteriores al QR.
            acc_curso = {}
            acc_grado = {}
            for r in rows[1:]:
                if not r or len(r) < 3: continue
                if (r[1] or '').strip() == 'PROMEDIO': continue
                # Aceptar la fila si tiene al menos nombre o ID
                if not ((r[0] or '').strip() or (r[1] or '').strip()):
                    continue
                r_curso = (r[2] or '').strip()
                r_grado = _grade_from_course_name(r_curso)
                if r_curso == curso:
                    _accum(acc_curso, r)
                    cnt_curso += 1
                if grado_stu and r_grado == grado_stu:
                    _accum(acc_grado, r)
                    cnt_grado += 1
            for k, v in acc_curso.items():
                if v and v[1] > 0:
                    avg_curso[k] = int(round(v[0] / v[1]))
            for k, v in acc_grado.items():
                if v and v[1] > 0:
                    avg_grado[k] = int(round(v[0] / v[1]))

            # Posición en el curso (1-based, mayor puntaje general = pos 1).
            # Identificamos al estudiante por su fila exacta (objeto) ya que
            # podría no haber ID en otras filas del curso.
            pos_curso = None
            tot_curso = 0
            try:
                curso_scores = []  # (row_obj, gen)
                for r in rows[1:]:
                    if not r or len(r) < 9: continue
                    if (r[1] or '').strip() == 'PROMEDIO': continue
                    if (r[2] or '').strip() != curso: continue
                    g = _num(r[8])
                    if g is not None:
                        curso_scores.append((r, g))
                tot_curso = len(curso_scores)
                curso_scores.sort(key=lambda x: x[1], reverse=True)
                for i, (r, _g) in enumerate(curso_scores):
                    if r is student_row:
                        pos_curso = i + 1
                        break
            except Exception:
                pass

            # ── PERCENTILES POR GRADO ──
            # Para cada asignatura y para el general, calcula:
            #   percentil = (# estudiantes del grado con puntaje <= el suyo) / n * 100
            # Permite que el PDF muestre las barras de percentil.
            percentiles = {}
            try:
                # Recolectar todos los puntajes del grado
                grade_rows = []
                for r in rows[1:]:
                    if not r or len(r) < 3: continue
                    if (r[1] or '').strip() == 'PROMEDIO': continue
                    if not ((r[0] or '').strip() or (r[1] or '').strip()): continue
                    if grado_stu and _grade_from_course_name(
                            (r[2] or '').strip()) == grado_stu:
                        grade_rows.append(r)
                    elif not grado_stu and (r[2] or '').strip() == curso:
                        grade_rows.append(r)
                n = len(grade_rows)
                fields_idx = {
                    'mat':  3, 'lect': 4, 'soc': 5, 'nat': 6, 'ing': 7,
                    'general': 8,
                }
                if n > 0:
                    for f, ci in fields_idx.items():
                        my_v = _num(student_row[ci]) if ci < len(student_row) else None
                        if my_v is None:
                            percentiles[f] = 0
                            continue
                        cnt = 0
                        for r in grade_rows:
                            if ci >= len(r): continue
                            v = _num(r[ci])
                            if v is not None and v <= my_v:
                                cnt += 1
                        percentiles[f] = int(round(cnt / n * 100))
            except Exception:
                percentiles = {}

            out.append({
                'simulacro':   sim.get('nombre', ''),
                'tipo':        sim.get('tipo', ''),
                'fecha':       sim.get('fecha', ''),
                'sheet':       res_sheet,
                'nombre':      nombre,
                'curso':       curso,
                'grado':       grado_stu,
                'puntajes':    puntajes,
                'avg_curso':   avg_curso,
                'avg_grado':   avg_grado,
                'pos_curso':   pos_curso,
                'tot_curso':   tot_curso,
                'n_curso':     cnt_curso,
                'n_grado':     cnt_grado,
                'percentiles': percentiles,
            })

    # Orden: más reciente primero
    out.sort(key=lambda r: r.get('fecha', ''), reverse=True)
    return out


# ═══════════════════════════════════════════════════════════════════════════
# MIGRACIÓN DE IDs: vincular documentos a estudiantes existentes en una hoja
# de resultados. Útil para simulacros calificados antes del sistema QR.
# ═══════════════════════════════════════════════════════════════════════════

def _normalize_name(name: str) -> str:
    """Minúsculas, sin acentos, espacios normalizados."""
    import unicodedata
    n = (name or '').strip().lower()
    n = unicodedata.normalize('NFD', n)
    n = ''.join(c for c in n if unicodedata.category(c) != 'Mn')
    n = ' '.join(n.split())
    return n


def _name_words(name: str) -> set:
    """Conjunto de palabras del nombre normalizado (independiente del orden)."""
    return set(_normalize_name(name).split())


def match_xlsx_to_results_sheet(xlsx_students: list, sheet_name: str) -> dict:
    """
    Cruza una lista de estudiantes (de XLSX con cols Curso/Documento/Nombre)
    contra las filas de una hoja de resultados existente.

    xlsx_students:  [{'curso','id','nombre'}, ...]
    sheet_name:     'Resultados Mayo 2026 IETECI M' (existente)

    Devuelve dict con 4 listas:
      - matched: matches de alta confianza (mismas palabras del nombre)
      - ambiguous: posibles matches que requieren confirmación
      - not_found: estudiantes del XLSX que no encontramos en la hoja
      - already_has_id: estudiantes cuya fila YA tenía un ID
                        (no se sobrescribe sin confirmación explícita)
    Cada item tiene la estructura:
      {'xlsx_curso','xlsx_id','xlsx_nombre',
       'sheet_row','sheet_curso','sheet_nombre','sheet_id_existing'}
    """
    if not xlsx_students:
        return {'matched': [], 'ambiguous': [], 'not_found': [],
                'already_has_id': []}
    try:
        ws = _open_spreadsheet().worksheet(sheet_name)
        rows = ws.get_all_values()
    except Exception as e:
        return {'success': False, 'error': f'No se pudo abrir "{sheet_name}": {e}'}

    if len(rows) < 2:
        return {'matched': [], 'ambiguous': [], 'not_found': xlsx_students,
                'already_has_id': []}

    # Indexar filas del sheet (saltando header). Guarda fila 1-based.
    sheet_rows = []   # [(row_num, id, nombre, curso, word_set)]
    for i, r in enumerate(rows[1:], start=2):  # i = 1-based row number
        if not r or len(r) < 3: continue
        if (r[1] or '').strip() == 'PROMEDIO': continue
        nombre = (r[1] or '').strip()
        if not nombre: continue
        ident = (r[0] or '').strip()
        curso = (r[2] or '').strip()
        sheet_rows.append({
            'row_num': i,
            'id': ident,
            'nombre': nombre,
            'curso': curso,
            'words': _name_words(nombre),
        })

    matched = []
    ambiguous = []
    not_found = []
    already_has_id = []

    for stu in xlsx_students:
        xn  = stu.get('nombre', '')
        xid = stu.get('id', '')
        xc  = stu.get('curso', '')
        xw  = _name_words(xn)
        if not xw or not xid:
            continue
        # Buscar matches exactos (mismas palabras)
        candidates = []
        for sr in sheet_rows:
            if sr['words'] == xw:
                candidates.append(('exact', sr))
        # Si no hay exactos, buscar parciales (>= 70% overlap)
        if not candidates:
            for sr in sheet_rows:
                if not sr['words']: continue
                overlap = len(xw & sr['words'])
                union   = len(xw | sr['words'])
                if union == 0: continue
                jac = overlap / union
                if jac >= 0.70:
                    candidates.append(('partial', sr, jac))

        if not candidates:
            not_found.append({
                'xlsx_curso': xc, 'xlsx_id': xid, 'xlsx_nombre': xn,
                'sheet_row': None, 'sheet_curso': '', 'sheet_nombre': '',
                'sheet_id_existing': '',
            })
            continue

        # Filtrar candidatos por curso si hay varios
        if len(candidates) > 1 and xc:
            same_curso = [c for c in candidates if c[1]['curso'] == xc]
            if same_curso:
                candidates = same_curso

        # Si solo hay 1 candidato y es exact → matched
        # Si hay 1 candidato pero es partial → ambiguous
        # Si hay varios → ambiguous (reportar todos)
        if len(candidates) == 1 and candidates[0][0] == 'exact':
            sr = candidates[0][1]
            item = {
                'xlsx_curso': xc, 'xlsx_id': xid, 'xlsx_nombre': xn,
                'sheet_row': sr['row_num'],
                'sheet_curso': sr['curso'],
                'sheet_nombre': sr['nombre'],
                'sheet_id_existing': sr['id'],
            }
            if sr['id'] and sr['id'] != xid:
                already_has_id.append(item)
            else:
                matched.append(item)
        else:
            # Devolver todos los candidatos (top 3) para que el usuario elija
            cands_data = []
            for c in candidates[:3]:
                sr = c[1]
                cands_data.append({
                    'row': sr['row_num'],
                    'nombre': sr['nombre'],
                    'curso': sr['curso'],
                    'id_existing': sr['id'],
                    'type': c[0],
                })
            ambiguous.append({
                'xlsx_curso': xc, 'xlsx_id': xid, 'xlsx_nombre': xn,
                'candidates': cands_data,
            })

    return {
        'success':         True,
        'matched':         matched,
        'ambiguous':       ambiguous,
        'not_found':       not_found,
        'already_has_id':  already_has_id,
        'sheet':           sheet_name,
        'sheet_total':     len(sheet_rows),
        'xlsx_total':      len(xlsx_students),
    }


def apply_id_matches_to_sheet(sheet_name: str, matches: list) -> dict:
    """
    Aplica una lista de matches confirmados, escribiendo el ID en la
    columna A de la fila correspondiente.

    matches: lista de {'sheet_row': int, 'xlsx_id': str}
    """
    if not matches:
        return {'success': True, 'updated': 0}
    try:
        ws = _open_spreadsheet().worksheet(sheet_name)
    except Exception as e:
        return {'success': False, 'error': f'No se pudo abrir "{sheet_name}": {e}'}
    # Hacer updates batch para ahorrar llamadas a la API
    updates = []
    for m in matches:
        rn = m.get('sheet_row')
        nid = (m.get('xlsx_id') or '').strip()
        if not rn or not nid: continue
        updates.append({
            'range':  f'A{rn}',
            'values': [[nid]],
        })
    if not updates:
        return {'success': True, 'updated': 0}
    try:
        ws.batch_update(updates, value_input_option='RAW')
        return {'success': True, 'updated': len(updates)}
    except Exception as e:
        return {'success': False, 'error': str(e)}


# ═══════════════════════════════════════════════════════════════════════════
# ROSTER GENERAL DE ESTUDIANTES — auto-completar nombre al teclear ID
# Independiente de simulacros. Una sola fila por estudiante (dedup por ID).
# ═══════════════════════════════════════════════════════════════════════════

def _ensure_estudiantes_sheet():
    """Crea la hoja Estudiantes si no existe. Cols: ID, Nombre, Curso."""
    ss = _open_spreadsheet()
    try:
        ws = ss.worksheet(ESTUDIANTES_SHEET)
        header = ws.row_values(1) or []
        if len(header) < 3 or header[0].strip().lower() != 'id':
            ws.update('A1:C1', [['ID', 'Nombre', 'Curso']],
                      value_input_option='RAW')
    except Exception:
        ws = ss.add_worksheet(title=ESTUDIANTES_SHEET, rows=3000, cols=3)
        ws.update('A1:C1', [['ID', 'Nombre', 'Curso']], value_input_option='RAW')
        try:
            ws.format('A1:C1', {
                'backgroundColor': {'red': 0.12, 'green': 0.23, 'blue': 0.54},
                'textFormat': {'bold': True,
                               'foregroundColor': {'red':1,'green':1,'blue':1}},
            })
        except Exception:
            pass
    return ws


def save_estudiantes(students: list, replace: bool = False) -> dict:
    """
    Guarda/actualiza el roster general.
      - students: lista de {'id','nombre','curso'}
      - replace: si True, borra TODAS las filas existentes y reescribe.
                 si False, hace upsert (existentes se actualizan por ID,
                 nuevos se añaden).
    Devuelve {'success':True, 'count': N, 'inserted':, 'updated':}.
    """
    if not isinstance(students, list):
        return {'success': False, 'error': 'students debe ser lista'}
    try:
      # Upsert leer-luego-escribir: serializado para no duplicar IDs si dos
      # threads suben el roster a la vez.
      with _WRITE_LOCK:
        ws = _ensure_estudiantes_sheet()
        rows_existing = ws.get_all_values()
        # Indexar existentes por ID
        existing_idx = {}   # id -> row_num (1-based)
        for i, r in enumerate(rows_existing[1:], start=2):
            if r and len(r) > 0 and r[0].strip():
                existing_idx[r[0].strip()] = i

        # Limpiar incoming
        clean = []
        seen_ids = set()
        for s in students:
            sid = str(s.get('id') or '').strip()
            nombre = str(s.get('nombre') or '').strip().upper()
            curso = str(s.get('curso') or '').strip()
            if not sid or sid in seen_ids:
                continue
            seen_ids.add(sid)
            clean.append({'id': sid, 'nombre': nombre, 'curso': curso})

        if replace:
            # Borrar todo (excepto header)
            n_rows = len(rows_existing)
            if n_rows > 1:
                ws.batch_clear([f'A2:C{n_rows}'])
            if clean:
                payload = [[s['id'], s['nombre'], s['curso']] for s in clean]
                ws.update(f'A2:C{1 + len(payload)}', payload, value_input_option='RAW')
            return {'success': True, 'count': len(clean),
                    'replaced': True}
        else:
            # Upsert: updates a existentes + appends a nuevos
            updates = []
            new_rows = []
            inserted = 0
            updated = 0
            for s in clean:
                if s['id'] in existing_idx:
                    rn = existing_idx[s['id']]
                    updates.append({
                        'range':  f'A{rn}:C{rn}',
                        'values': [[s['id'], s['nombre'], s['curso']]],
                    })
                    updated += 1
                else:
                    new_rows.append([s['id'], s['nombre'], s['curso']])
                    inserted += 1
            if updates:
                ws.batch_update(updates, value_input_option='RAW')
            if new_rows:
                ws.append_rows(new_rows, value_input_option='RAW')
            return {'success': True, 'count': inserted + updated,
                    'inserted': inserted, 'updated': updated}
    except Exception as e:
        return {'success': False, 'error': str(e)}


def _get_estudiantes_indexed():
    """Carga el roster general y lo indexa por ID. Cacheado 60s."""
    def _load():
        try:
            ws = _ensure_estudiantes_sheet()
            idx = {}
            for r in ws.get_all_values()[1:]:
                if r and (r[0] or '').strip():
                    sid = (r[0] or '').strip()
                    idx[sid] = {
                        'id':     sid,
                        'nombre': (r[1] or '').strip() if len(r) > 1 else '',
                        'curso':  (r[2] or '').strip() if len(r) > 2 else '',
                    }
            return idx
        except Exception:
            return {}
    return _cached_read('estudiantes:idx', _load, ttl=60)


def get_student_global(student_id: str) -> dict:
    """Busca un estudiante por ID en el roster general. Devuelve dict o None.

    Optimizado con cache (1 lectura cada 60s para todo el roster, después
    lookup en memoria O(1) — soporta cientos de consultas/min sin tocar
    Sheets API).
    """
    sid = str(student_id or '').strip()
    if not sid:
        return None
    try:
        idx = _get_estudiantes_indexed()
        return idx.get(sid)
    except Exception:
        pass
    # Fallback: lectura directa (solo si el cache falla)
    try:
        ws = _ensure_estudiantes_sheet()
        for r in ws.get_all_values()[1:]:
            if len(r) >= 1 and (r[0] or '').strip() == sid:
                return {
                    'id':     sid,
                    'nombre': (r[1] or '').strip() if len(r) > 1 else '',
                    'curso':  (r[2] or '').strip() if len(r) > 2 else '',
                }
    except Exception:
        pass
    return None


def count_estudiantes() -> int:
    """Cuenta los estudiantes registrados en el roster general."""
    try:
        ws = _ensure_estudiantes_sheet()
        rows = ws.get_all_values()
        return sum(1 for r in rows[1:] if r and r[0].strip())
    except Exception:
        return 0
