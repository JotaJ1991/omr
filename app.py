"""
OMR App  —  Calificador JMR  (multi-perfil)
"""
import os
import json
import traceback
import base64
from flask import Flask, request, jsonify, render_template, session, Response
from werkzeug.utils import secure_filename
from omr_processor import process_exam_image
from exam_profiles import PROFILE_LIST, get_profile, DEFAULT_PROFILE_ID
from sheets_connector import (
    save_to_sheets, save_answer_key, get_answer_key, get_sheet_data,
    list_sheets, create_sheet, generate_sipagre_results
)
# PDF generation moved to browser-side (jsPDF) — no server imports needed

app = Flask(__name__)
app.secret_key  = os.environ.get('SECRET_KEY', 'omr-secret-2025')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER']      = 'uploads'

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

    current_raw = request.form.get('current_answers', '')
    try:
        current = json.loads(current_raw) if current_raw else []
    except Exception:
        current = []

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = process_exam_image(filepath, profile_id=pid, debug=True)

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
    answers      = data.get('answers', [])
    sheet_name   = data.get('sheet_name', active_sheet())

    if not student_name:
        return jsonify({'success': False,
                        'error': 'El nombre del estudiante es obligatorio.'}), 400
    n = len(answers)
    if n < 1 or n > 200:
        return jsonify({'success': False,
                        'error': 'Número de respuestas inválido.'}), 400

    try:
        sheets_result = save_to_sheets(
            student_name=student_name,
            exam_id=exam_id,
            answers=answers,
            sheet_name=sheet_name
        )
        detected = len([a for a in answers if a not in ('?', '')])
        correct  = sheets_result.get('correct')
        pct      = sheets_result.get('pct', '')

        if correct is not None:
            msg = f'Guardado en "{sheet_name}" — {correct}/{n} correctas ({pct})'
        else:
            msg = f'Guardado en "{sheet_name}" — {detected}/{n} detectadas.'

        return jsonify({
            'success':    True,
            'row':        sheets_result.get('row'),
            'sheets_url': sheets_result.get('url'),
            'detected':   detected,
            'correct':    correct,
            'pct':        pct,
            'message':    msg
        })
    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error al guardar: {str(e)}'}), 500


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


## Las rutas de PDF se eliminaron — ahora se generan 100% en el navegador con jsPDF


@app.route('/debug_image', methods=['POST'])
def debug_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No se recibió imagen'}), 400
    file     = request.files['image']
    pid      = request.form.get('profile_id', active_profile_id())
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_' + filename)
    file.save(filepath)

    result     = process_exam_image(filepath, profile_id=pid, debug=True)
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


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
