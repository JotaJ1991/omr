"""
OMR App  —  Calificador JMR  (multi-perfil)
"""
import os
import json
import traceback
import base64
import threading
import uuid
import secrets
from flask import Flask, request, jsonify, render_template, session, Response
from werkzeug.utils import secure_filename
from omr_processor import process_exam_image, _parse_qr_payload
from exam_profiles import PROFILE_LIST, get_profile, DEFAULT_PROFILE_ID
from sheets_connector import (
    save_to_sheets, save_answer_key, get_answer_key, get_sheet_data,
    list_sheets, create_sheet, generate_sipagre_results,
    generate_msipagre_results,
    list_courses, add_course, delete_course,
    list_simulacros, add_simulacro, delete_simulacro,
    uppercase_all_student_names,
    analyze_simulacro_questions,
    list_distribuciones_json, save_distribucion,
    get_key_for_grade, save_key_for_grade, list_keys_grade,
    get_anuladas_for_grade, save_anuladas_for_grade, list_anuladas,
    get_student_from_roster, list_roster,
    get_student_results_all_simulacros,
    match_xlsx_to_results_sheet, apply_id_matches_to_sheet,
    save_estudiantes, get_student_global, count_estudiantes,
    _open_spreadsheet,
)
# PDF generation moved to browser-side (jsPDF) — no server imports needed

app = Flask(__name__)
# SECRET_KEY: SIEMPRE desde el entorno. Si falta, se genera una aleatoria por
# arranque (las sesiones del portal se invalidan en cada reinicio, pero nunca
# se usa una clave fija publicada en el repositorio, que permitiría falsificar
# sesiones). Configura SECRET_KEY en Render -> Environment para sesiones
# estables entre despliegues.
app.secret_key = os.environ.get('SECRET_KEY') or secrets.token_hex(32)
if not os.environ.get('SECRET_KEY'):
    print('[ADVERTENCIA] SECRET_KEY no configurada; usando clave aleatoria '
          'temporal (las sesiones se invalidan al reiniciar).', flush=True)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER']      = 'uploads'
# Crear la carpeta de subidas a nivel de módulo (bajo gunicorn no se ejecuta
# el bloque __main__, así que debe garantizarse aquí).
os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)
# Auto-recargar templates al modificarlos (sin esto Flask cachea el HTML)
app.config['TEMPLATES_AUTO_RELOAD'] = True
app.jinja_env.auto_reload = True

# ── Salvaguarda de memoria ──────────────────────────────────────────────────
# Limita cuántos escaneos PESADOS (OpenCV) se procesan a la vez dentro del
# worker. Aunque entren 5 docentes simultáneamente, solo N se procesan al
# mismo tiempo y el resto ESPERA EN COLA unos segundos (no se pierde nada).
# Esto evita el OutOfMemory en la instancia de 512 MB. Las peticiones livianas
# (cargar páginas, guardar en Sheets, portal) NO pasan por aquí, así que no se
# bloquean. Ajustable con la variable de entorno MAX_CONCURRENT_SCANS.
_MAX_CONCURRENT_SCANS = int(os.environ.get('MAX_CONCURRENT_SCANS', '2'))
_SCAN_SEMAPHORE = threading.BoundedSemaphore(_MAX_CONCURRENT_SCANS)
# Tiempo máximo que un escaneo espera en cola antes de rendirse (segundos).
# Evita que una petición quede colgada indefinidamente si hay saturación.
_SCAN_QUEUE_TIMEOUT = int(os.environ.get('SCAN_QUEUE_TIMEOUT', '90'))

# Forzar no-cache en respuestas HTML para evitar caché del navegador
@app.after_request
def _no_cache_html(resp):
    """Evita que el navegador cachee las páginas HTML — así los cambios
    en index.html / portal_estudiante.html se reflejan al instante."""
    ctype = resp.headers.get('Content-Type', '')
    if ctype.startswith('text/html'):
        resp.headers['Cache-Control'] = 'no-store, no-cache, must-revalidate, max-age=0'
        resp.headers['Pragma']        = 'no-cache'
        resp.headers['Expires']       = '0'
    return resp

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
DEFAULT_SHEET      = 'Respuestas'

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def active_sheet():
    return session.get('active_sheet', DEFAULT_SHEET)

def active_profile_id():
    return session.get('profile_id', DEFAULT_PROFILE_ID)


# ── Página principal ──────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html',
                           profiles=PROFILE_LIST,
                           default_profile=DEFAULT_PROFILE_ID)


# ── Hojas ─────────────────────────────────────────────────────────────────────

@app.route('/active_sheet', methods=['GET'])
def get_active_sheet():
    try:    sheets = list_sheets()
    except: sheets = []
    return jsonify({'active': active_sheet(), 'sheets': sheets})

@app.route('/active_sheet', methods=['POST'])
def set_active_sheet():
    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Nombre vacío'}), 400
    session['active_sheet'] = name
    session.modified = True
    return jsonify({'success': True, 'active': name})

@app.route('/sheets', methods=['GET'])
def sheets_list():
    try:    return jsonify({'success': True, 'sheets': list_sheets()})
    except Exception as e: return jsonify({'success': False, 'error': str(e)})

@app.route('/sheets', methods=['POST'])
def sheets_create():
    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Nombre vacío'}), 400
    result = create_sheet(name)
    if result['success']:
        session['active_sheet'] = name
        session.modified = True
        result['active'] = name
    return jsonify(result)


# ── Perfil de examen ──────────────────────────────────────────────────────────

@app.route('/profile', methods=['GET'])
def get_profile_route():
    pid     = active_profile_id()
    profile = get_profile(pid)
    return jsonify({
        'active_id':   pid,
        'active_name': profile['name'],
        'total_q':     profile['total_q'],
        'profiles':    PROFILE_LIST,
        # Opciones por pregunta — útil para la UI de clave
        'options_per_q': _build_options_per_q(profile),
    })

@app.route('/profile', methods=['POST'])
def set_profile_route():
    data = request.get_json(silent=True) or {}
    pid  = data.get('profile_id', '').strip()
    p    = get_profile(pid)
    if p['id'] != pid:   # fallback ocurrió → ID inválido
        return jsonify({'success': False, 'error': f'Perfil "{pid}" no existe.'}), 400
    session['profile_id'] = pid
    session.modified = True
    profile = get_profile(pid)
    return jsonify({
        'success':       True,
        'active_id':     pid,
        'active_name':   profile['name'],
        'total_q':       profile['total_q'],
        'options_per_q': _build_options_per_q(profile),
    })

def _build_options_per_q(profile) -> list:
    """Lista de longitud total_q con las opciones válidas de cada pregunta."""
    result = []
    for col in profile['columns']:
        n = col['q_end'] - col['q_start'] + 1
        result.extend([col['options']] * n)
    return result


# ── Procesar imagen ───────────────────────────────────────────────────────────

@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No se recibió imagen'}), 400

    file = request.files['image']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'success': False,
                        'error': 'Formato no soportado. Usa JPG, PNG o WEBP'}), 400

    pid         = request.form.get('profile_id', active_profile_id())
    profile     = get_profile(pid)
    total_q     = profile['total_q']

    # ── Determinar el total efectivo según el grado del estudiante ──
    # Si el cliente envía el curso, calculamos cuántas preguntas se evalúan
    # realmente para ese grado (suma del fin de la distribución). Las
    # respuestas más allá de esa posición se descartan y NO aparecen en el
    # debug visual.
    curso_req = (request.form.get('curso') or '').strip()
    effective_total_q = None
    print(f'[/process] perfil={pid} curso_recibido={curso_req!r}', flush=True)
    if curso_req:
        try:
            from sheets_connector import (
                _grade_from_course_name, list_distribuciones,
                DEFAULT_DISTRIBUCIONES,
            )
            grado_stu = _grade_from_course_name(curso_req)
            if grado_stu:
                # Buscar la distribucion del grado en ambos tipos.
                # No siempre el perfil OMR coincide con el tipo del simulacro:
                # ej. IETECI Mañana es tipo='media' pero 6°/7°/8° usan perfil
                # 1S SIPAGRE (formato fisico de "completo"). Por eso probamos
                # media PRIMERO (mas comun) y completo como fallback.
                dists = list_distribuciones()
                dist = None
                tipo_used = None
                for tipo in ('media', 'completo'):
                    d = dists.get((tipo, grado_stu)) or \
                        DEFAULT_DISTRIBUCIONES.get((tipo, grado_stu))
                    if d:
                        dist = d
                        tipo_used = tipo
                        break
                if dist:
                    effective_total_q = max((int(e.get('fin', 0)) for e in dist),
                                            default=0) or None
                    print(f'[/process] grado={grado_stu} tipo_usado={tipo_used} '
                          f'effective_total_q={effective_total_q}', flush=True)
                else:
                    print(f'[/process] grado={grado_stu} '
                          f'SIN DISTRIBUCIÓN configurada (ni media ni completo)',
                          flush=True)
        except Exception:
            traceback.print_exc()
            effective_total_q = None

    current_raw = request.form.get('current_answers', '')
    try:
        current = json.loads(current_raw) if current_raw else []
    except Exception:
        current = []

    try:
        # Prefijo único: dos docentes subiendo "image.jpg" a la vez no deben
        # pisarse el archivo (con 4 threads el segundo sobreescribiría al
        # primero y un docente recibiría las respuestas del otro).
        filename = f"{uuid.uuid4().hex}_{secure_filename(file.filename)}"
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Cola de procesamiento: espera turno si ya hay N escaneos pesados en
        # curso. Protege la RAM de 512 MB sin perder la petición.
        if not _SCAN_SEMAPHORE.acquire(timeout=_SCAN_QUEUE_TIMEOUT):
            if os.path.exists(filepath):
                os.remove(filepath)
            return jsonify({'success': False,
                            'error': 'El servidor está procesando muchas hojas '
                                     'en este momento. Espera unos segundos y '
                                     'vuelve a escanear.'}), 503
        try:
            result = process_exam_image(filepath, profile_id=pid, debug=True,
                                        effective_total_q=effective_total_q)
        finally:
            _SCAN_SEMAPHORE.release()

        # Leer imagen anotada y eliminar archivos temporales
        debug_path = filepath.rsplit('.', 1)[0] + '_debug.jpg'
        img_b64 = None
        if os.path.exists(debug_path):
            with open(debug_path, 'rb') as f:
                img_b64 = base64.b64encode(f.read()).decode()
            os.remove(debug_path)
        if os.path.exists(filepath):
            os.remove(filepath)

        if not result['success']:
            return jsonify({'success': False, 'error': result['error']}), 422

        new_answers = result['answers']
        # Si se truncó por effective_total_q, usar ese como total real
        total_q = result.get('total_q', total_q)

        if current and len(current) == total_q:
            merged = current[:]
            filled = []
            for i in range(total_q):
                if merged[i] == '?' and new_answers[i] != '?':
                    merged[i] = new_answers[i]
                    filled.append(i + 1)
        else:
            merged = new_answers
            filled = [i + 1 for i, a in enumerate(merged) if a != '?']

        detected   = len([a for a in merged if a != '?'])
        missing    = total_q - detected
        new_filled = len(filled) if current else 0

        # Enriquecer el QR con datos del roster (si el estudiante está registrado)
        qr_info = result.get('qr')
        # Respaldo: si el servidor no detectó el QR (p.ej. la cámara guiada
        # recortó el encabezado y dejó el QR fuera de la imagen), usar la pista
        # decodificada en el cliente desde el frame completo.
        if not qr_info:
            qr_hint = (request.form.get('qr_hint', '') or '').strip()
            if qr_hint:
                hinted = _parse_qr_payload(qr_hint)
                if hinted:
                    qr_info = hinted
                    print(f'[/process] QR no detectado en servidor; usando '
                          f'pista del cliente: {qr_hint}', flush=True)
        if qr_info and qr_info.get('student_id'):
            try:
                roster_match = get_student_from_roster(
                    qr_info.get('simulacro', ''),
                    qr_info.get('student_id', ''))
                # Fallback: si no está en el roster del simulacro, buscar en el
                # roster general (Estudiantes) por ID, para autocompletar el nombre.
                if not roster_match:
                    g = get_student_global(qr_info.get('student_id', ''))
                    if g:
                        roster_match = {
                            'nombre': g.get('nombre', ''),
                            'curso':  g.get('curso', ''),
                        }
                if roster_match:
                    qr_info = dict(qr_info)
                    qr_info['nombre'] = roster_match.get('nombre', '')
                    # Si el roster tiene curso y el QR no, usar el del roster
                    if not qr_info.get('curso') and roster_match.get('curso'):
                        qr_info['curso'] = roster_match['curso']
                    qr_info['roster_matched'] = True
                else:
                    qr_info = dict(qr_info)
                    qr_info['roster_matched'] = False
            except Exception:
                pass

        return jsonify({
            'success':     True,
            'answers':     merged,
            'confidence':  result['confidence'],
            'detected':    detected,
            'missing':     missing,
            'new_filled':  new_filled,
            'filled_qs':   filled[:20],
            'total_q':     total_q,
            'profile_id':  pid,
            'debug_image': img_b64,
            'qr':          qr_info,   # info del estudiante si la hoja tiene QR
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error interno: {str(e)}'}), 500


# ── Guardar en Sheets ─────────────────────────────────────────────────────────

@app.route('/save', methods=['POST'])
def save():
    data = request.get_json(silent=True) or {}

    student_name = data.get('student_name', '').strip()
    exam_id      = data.get('exam_id', '').strip()
    curso        = (data.get('curso') or '').strip()
    answers      = data.get('answers', [])
    sheet_name   = data.get('sheet_name', active_sheet())

    if not student_name and not exam_id:
        return jsonify({'success': False,
                        'error': 'Ingresa al menos nombre o ID del estudiante.'}), 400
    n = len(answers)
    if n < 1 or n > 200:
        return jsonify({'success': False,
                        'error': 'Número de respuestas inválido.'}), 400

    try:
        sheets_result = save_to_sheets(
            student_name=student_name,
            exam_id=exam_id,
            answers=answers,
            sheet_name=sheet_name,
            curso=curso
        )
        detected = len([a for a in answers if a not in ('?', '')])
        correct  = sheets_result.get('correct')
        pct      = sheets_result.get('pct', '')

        was_update = bool(sheets_result.get('updated'))
        verb = 'Actualizado' if was_update else 'Guardado'
        if correct is not None:
            msg = f'{verb} en "{sheet_name}" — {correct}/{n} correctas ({pct})'
        else:
            msg = f'{verb} en "{sheet_name}" — {detected}/{n} detectadas.'

        return jsonify({
            'success':    True,
            'row':        sheets_result.get('row'),
            'sheets_url': sheets_result.get('url'),
            'sheet_name': sheet_name,
            'detected':   detected,
            'correct':    correct,
            'pct':        pct,
            'updated':    was_update,
            'message':    msg
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error al guardar: {str(e)}'}), 500


@app.route('/courses', methods=['GET'])
def courses_list():
    try:
        return jsonify({'success': True, 'courses': list_courses()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e), 'courses': []}), 500


@app.route('/courses', methods=['POST'])
def courses_add():
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    grado = (data.get('grado') or '').strip()
    try:
        result = add_course(name, grado)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/courses', methods=['DELETE'])
def courses_delete():
    data = request.get_json(silent=True) or {}
    name = (data.get('name') or '').strip()
    try:
        result = delete_course(name)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ─── Simulacros ────────────────────────────────────────────────────────
@app.route('/simulacros', methods=['GET'])
def simulacros_list():
    try:
        return jsonify({'success': True, 'simulacros': list_simulacros()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e), 'simulacros': []}), 500


@app.route('/simulacros', methods=['POST'])
def simulacros_add():
    data = request.get_json(silent=True) or {}
    nombre = (data.get('nombre') or '').strip()
    fecha  = (data.get('fecha') or '').strip()
    tipo   = (data.get('tipo') or '').strip()
    grados = data.get('grados') or []
    try:
        result = add_simulacro(nombre, fecha, tipo, grados)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/keys_grade', methods=['GET'])
def keys_grade_list():
    sim   = (request.args.get('simulacro') or '').strip()
    ses   = (request.args.get('sesion') or '').strip()
    grado = (request.args.get('grado') or '').strip()
    try:
        if sim and ses and grado:
            ans = get_key_for_grade(sim, ses, grado)
            return jsonify({'success': True, 'answers': ans})
        return jsonify({'success': True, 'keys': list_keys_grade()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/keys_grade', methods=['POST'])
def keys_grade_save():
    data = request.get_json(silent=True) or {}
    sim   = (data.get('simulacro') or '').strip()
    ses   = (data.get('sesion') or '').strip()
    grado = str(data.get('grado') or '').strip()
    answers = data.get('answers') or []
    try:
        return jsonify(save_key_for_grade(sim, ses, grado, answers))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/anuladas', methods=['GET'])
def anuladas_list():
    sim   = (request.args.get('simulacro') or '').strip()
    ses   = (request.args.get('sesion') or '').strip()
    grado = (request.args.get('grado') or '').strip()
    try:
        if sim and ses and grado:
            nums = get_anuladas_for_grade(sim, ses, grado)
            return jsonify({'success': True, 'preguntas': nums})
        return jsonify({'success': True, 'anuladas': list_anuladas()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/anuladas', methods=['POST'])
def anuladas_save():
    data = request.get_json(silent=True) or {}
    sim   = (data.get('simulacro') or '').strip()
    ses   = (data.get('sesion') or '').strip()
    grado = str(data.get('grado') or '').strip()
    preguntas = data.get('preguntas')
    if preguntas is None:
        preguntas = data.get('csv') or ''
    try:
        return jsonify(save_anuladas_for_grade(sim, ses, grado, preguntas))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/distribuciones', methods=['GET'])
def distribuciones_list():
    try:
        return jsonify({'success': True, 'distribuciones': list_distribuciones_json()})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/distribuciones', methods=['POST'])
def distribuciones_save():
    data = request.get_json(silent=True) or {}
    tipo  = (data.get('tipo') or '').strip()
    grado = str(data.get('grado') or '').strip()
    materias = data.get('materias') or []
    try:
        return jsonify(save_distribucion(tipo, grado, materias))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/analyze_questions', methods=['POST'])
def analyze_questions_endpoint():
    data = request.get_json(silent=True) or {}
    nombre = (data.get('simulacro') or '').strip()
    try:
        return jsonify(analyze_simulacro_questions(nombre))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/uppercase_names', methods=['POST'])
def uppercase_names_endpoint():
    """Convierte todos los nombres de estudiantes a MAYÚSCULAS en todas las hojas."""
    try:
        return jsonify(uppercase_all_student_names())
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ── PERSONALIZACIÓN DE HOJAS IETAGRO (QR + datos pre-impresos) ─────────────
@app.route('/personalize/preview', methods=['POST'])
def personalize_preview():
    """
    Recibe un archivo XLSX y devuelve la lista de estudiantes detectados
    (sin generar PDFs). Útil para que la UI muestre preview antes.
    """
    try:
        from pdf_personalizer import parse_roster_xlsx
        f = request.files.get('roster')
        if not f:
            return jsonify({'success': False,
                            'error': 'No se recibió archivo (campo "roster").'}), 400
        xlsx_bytes = f.read()
        return jsonify(parse_roster_xlsx(xlsx_bytes))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/personalize/generate', methods=['POST'])
def personalize_generate():
    """
    Recibe el XLSX + nombre del simulacro, genera todos los PDFs personalizados
    (1S+2S por estudiante) y devuelve un ZIP organizado por curso.
    """
    try:
        from pdf_personalizer import parse_roster_xlsx, generate_pdfs_zip
        f = request.files.get('roster')
        sim_id = (request.form.get('simulacro') or '').strip()
        if not f:
            return jsonify({'success': False,
                            'error': 'No se recibió archivo (campo "roster").'}), 400
        if not sim_id:
            return jsonify({'success': False,
                            'error': 'Falta el nombre del simulacro.'}), 400
        xlsx_bytes = f.read()
        parsed = parse_roster_xlsx(xlsx_bytes)
        if not parsed.get('success'):
            return jsonify(parsed), 400
        # Generar
        result = generate_pdfs_zip(parsed['students'], sim_id)
        if not result.get('success'):
            return jsonify(result), 500
        # Devolver UN SOLO PDF con todas las páginas mergeadas
        pdf_bytes = result.get('pdf_bytes') or result['zip_bytes']
        sim_safe = ''.join(c if c.isalnum() else '_' for c in sim_id).strip('_')
        return Response(
            pdf_bytes,
            mimetype='application/pdf',
            headers={
                'Content-Disposition':
                    f'attachment; filename="hojas_{sim_safe}.pdf"',
                'X-PDF-Count':     str(result.get('count', 0)),
                'X-PDF-Failures':  str(result.get('failures', 0)),
                'X-Roster-Saved':  str(result.get('roster_saved', 0)),
            }
        )
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/simulacros', methods=['DELETE'])
def simulacros_delete():
    data = request.get_json(silent=True) or {}
    nombre = (data.get('nombre') or '').strip()
    try:
        result = delete_simulacro(nombre)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/get_key', methods=['GET'])
def get_key():
    """Retorna la clave guardada en la hoja activa."""
    sheet_name = request.args.get('sheet', active_sheet())
    try:
        result = get_answer_key(sheet_name)
        return jsonify(result)
    except Exception as e:
        return jsonify({'success': False, 'error': str(e), 'answers': []})


@app.route('/save_key', methods=['POST'])
def save_key():
    data       = request.get_json(silent=True) or {}
    answers    = data.get('answers', [])
    sheet_name = data.get('sheet_name', active_sheet())

    if not answers:
        return jsonify({'success': False, 'error': 'No se recibieron respuestas.'}), 400
    try:
        result = save_answer_key(answers, sheet_name)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ── Resultados ────────────────────────────────────────────────────────────────

@app.route('/results')
def results():
    sheet_name = request.args.get('sheet', active_sheet())
    try:
        data = get_sheet_data(sheet_name)
        return jsonify({'success': True, 'data': data, 'sheet': sheet_name})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/sipagre_results', methods=['POST'])
def sipagre_results():
    """Genera la hoja de Resultados SIPAGRE combinando 1S y 2S."""
    data = request.get_json(silent=True) or {}
    sheet_1s       = data.get('sheet_1s', '1S SIPAGRE')
    sheet_2s       = data.get('sheet_2s', '2S SIPAGRE')
    results_sheet  = data.get('results_sheet', 'Resultados SIPAGRE')
    try:
        result = generate_sipagre_results(sheet_1s, sheet_2s, results_sheet)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error: {str(e)}'}), 500


@app.route('/msipagre_results', methods=['POST'])
def msipagre_results():
    """Genera la hoja de Resultados M SIPAGRE (única jornada de 125 preguntas)."""
    data = request.get_json(silent=True) or {}
    sheet_m        = data.get('sheet_m', 'M SIPAGRE')
    results_sheet  = data.get('results_sheet', 'Resultados M SIPAGRE')
    try:
        result = generate_msipagre_results(sheet_m, results_sheet)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error: {str(e)}'}), 500


## Las rutas de PDF se eliminaron — ahora se generan 100% en el navegador con jsPDF


@app.route('/debug_image', methods=['POST'])
def debug_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No se recibió imagen'}), 400
    file     = request.files['image']
    pid      = request.form.get('profile_id', active_profile_id())
    filename = f"debug_{uuid.uuid4().hex}_{secure_filename(file.filename)}"
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
    file.save(filepath)

    # Misma cola de procesamiento que /process (protege la RAM).
    if not _SCAN_SEMAPHORE.acquire(timeout=_SCAN_QUEUE_TIMEOUT):
        if os.path.exists(filepath):
            os.remove(filepath)
        return jsonify({'success': False,
                        'error': 'Servidor ocupado, intenta de nuevo en unos '
                                 'segundos.'}), 503
    try:
        result = process_exam_image(filepath, profile_id=pid, debug=True)
    finally:
        _SCAN_SEMAPHORE.release()
    debug_path = filepath.rsplit('.', 1)[0] + '_debug.jpg'
    img_b64    = None

    if os.path.exists(debug_path):
        with open(debug_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
        os.remove(debug_path)
    if os.path.exists(filepath):
        os.remove(filepath)

    return jsonify({
        'success':     result['success'],
        'answers':     result.get('answers', []),
        'debug_image': img_b64
    })


# ═══════════════════════════════════════════════════════════════════════════
# ROSTER GENERAL DE ESTUDIANTES — autocompletar nombre al teclear ID
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/estudiantes/lookup', methods=['GET'])
def estudiantes_lookup():
    """Búsqueda rápida por ID. Devuelve {success, id, nombre, curso} o 404."""
    sid = (request.args.get('id') or '').strip()
    if not sid:
        return jsonify({'success': False, 'error': 'id requerido'}), 400
    try:
        s = get_student_global(sid)
        if not s:
            return jsonify({'success': False, 'found': False}), 200
        return jsonify({'success': True, 'found': True, **s})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/estudiantes/count', methods=['GET'])
def estudiantes_count():
    """Cuántos estudiantes hay en el roster general."""
    try:
        return jsonify({'success': True, 'count': count_estudiantes()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/estudiantes/upload', methods=['POST'])
def estudiantes_upload():
    """Sube XLSX (Curso, Documento, Nombres). Modo upsert por defecto;
    si form['replace']='1' borra todo y reemplaza."""
    try:
        from pdf_personalizer import parse_roster_xlsx
        f = request.files.get('roster')
        if not f:
            return jsonify({'success': False,
                            'error': 'No se recibió archivo.'}), 400
        replace = (request.form.get('replace') or '0') == '1'
        parsed = parse_roster_xlsx(f.read())
        if not parsed.get('success'):
            return jsonify(parsed), 400
        result = save_estudiantes(parsed['students'], replace=replace)
        if not result.get('success'):
            return jsonify(result), 500
        result['xlsx_total'] = parsed.get('count', 0)
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# MIGRACIÓN DE IDs — vincular documentos a estudiantes de hojas existentes
# ═══════════════════════════════════════════════════════════════════════════

@app.route('/migrate_ids/result_sheets', methods=['GET'])
def migrate_ids_result_sheets():
    """Lista TODAS las hojas que empiezan con 'Resultados '."""
    try:
        ss = _open_spreadsheet()
        out = []
        for ws in ss.worksheets():
            t = ws.title or ''
            if t.lower().startswith('resultados '):
                out.append(t)
        out.sort()
        return jsonify({'success': True, 'sheets': out})
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/migrate_ids/preview', methods=['POST'])
def migrate_ids_preview():
    """Recibe XLSX + sheet_name, devuelve matches a confirmar."""
    try:
        from pdf_personalizer import parse_roster_xlsx
        f = request.files.get('roster')
        sheet_name = (request.form.get('sheet_name') or '').strip()
        if not f or not sheet_name:
            return jsonify({'success': False,
                            'error': 'Falta archivo o hoja destino.'}), 400
        parsed = parse_roster_xlsx(f.read())
        if not parsed.get('success'):
            return jsonify(parsed), 400
        result = match_xlsx_to_results_sheet(parsed['students'], sheet_name)
        if not result.get('success', True):
            return jsonify(result), 400
        return jsonify(result)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


@app.route('/migrate_ids/apply', methods=['POST'])
def migrate_ids_apply():
    """Aplica los matches confirmados a la hoja."""
    try:
        data = request.get_json(silent=True) or {}
        sheet_name = (data.get('sheet_name') or '').strip()
        matches    = data.get('matches') or []
        if not sheet_name or not matches:
            return jsonify({'success': False,
                            'error': 'Faltan datos para aplicar.'}), 400
        return jsonify(apply_id_matches_to_sheet(sheet_name, matches))
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': str(e)}), 500


# ═══════════════════════════════════════════════════════════════════════════
# PORTAL ESTUDIANTIL  (público — login con ID + contraseña=ID)
# ═══════════════════════════════════════════════════════════════════════════
import time
from collections import defaultdict

# Rate-limit en memoria contra brute-force: máx 10 intentos/min por IP
_login_attempts = defaultdict(list)
_RATE_WINDOW_S = 60
_RATE_MAX = 10


def _rate_limit_ok(ip: str) -> bool:
    now = time.time()
    # Limpia intentos viejos
    _login_attempts[ip] = [t for t in _login_attempts[ip]
                           if now - t < _RATE_WINDOW_S]
    if len(_login_attempts[ip]) >= _RATE_MAX:
        return False
    _login_attempts[ip].append(now)
    return True


@app.route('/resultados', methods=['GET'])
def resultados_portal():
    """Página del portal estudiantil (login)."""
    return render_template('portal_estudiante.html')


@app.route('/resultados/login', methods=['POST'])
def resultados_login():
    """Valida credenciales (usuario=ID, contraseña=ID) y devuelve resultados."""
    ip = request.remote_addr or 'unknown'
    if not _rate_limit_ok(ip):
        return jsonify({'success': False,
                        'error': 'Demasiados intentos. Espera un minuto.'}), 429

    data = request.get_json(silent=True) or {}
    user = (data.get('usuario') or '').strip()
    pwd  = (data.get('password') or '').strip()

    if not user or not pwd:
        return jsonify({'success': False,
                        'error': 'Debes ingresar usuario y contraseña.'}), 400
    # Esquema simple: la contraseña debe ser igual al usuario (ID)
    if user != pwd:
        return jsonify({'success': False,
                        'error': 'Credenciales incorrectas.'}), 401

    try:
        results = get_student_results_all_simulacros(user)
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False,
                        'error': f'Error consultando resultados: {e}'}), 500

    if not results:
        return jsonify({'success': False,
                        'error': 'No encontramos resultados con ese ID. '
                                 'Verifica el número o pregunta a tu docente.'}), 404

    # Datos del estudiante (tomamos el primer resultado para el header)
    student_info = {
        'id':     user,
        'nombre': results[0].get('nombre', ''),
        'curso':  results[0].get('curso', ''),
        'grado':  results[0].get('grado', ''),
    }
    # Guardar sesión para el dashboard
    session['student_id'] = user
    session['student_info'] = student_info

    return jsonify({
        'success':     True,
        'student':     student_info,
        'results':     results,
        'count':       len(results),
    })


@app.route('/resultados/logout', methods=['POST'])
def resultados_logout():
    session.pop('student_id', None)
    session.pop('student_info', None)
    return jsonify({'success': True})


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    # PORT y debug se leen del entorno para que el MISMO archivo funcione:
    #   - Local:  python app.py            -> puerto 5000, debug ON
    #   - Render: gunicorn app:app ...     -> no usa este bloque (debug OFF)
    # Para activar debug local sin riesgo en producción, deja FLASK_DEBUG sin
    # definir en Render (allí se sirve con gunicorn, que ignora este bloque).
    port  = int(os.environ.get('PORT', 5000))
    debug = os.environ.get('FLASK_DEBUG', '1') == '1'
    app.run(debug=debug, host='0.0.0.0', port=port)
