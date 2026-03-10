"""
OMR App  —  Calificador JMR
Flujo: procesar (solo detección) → complementar fotos → guardar (Sheets)
"""
import os
import json
import traceback
import base64
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from omr_processor import process_exam_image
from sheets_connector import (
    save_to_sheets, save_answer_key, get_sheet_data,
    list_sheets, create_sheet
)

app = Flask(__name__)
app.secret_key = os.environ.get('SECRET_KEY', 'omr-secret-2025')
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}
DEFAULT_SHEET = 'Respuestas'

def allowed_file(f):
    return '.' in f and f.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

def active_sheet():
    return session.get('active_sheet', DEFAULT_SHEET)


# ── Hojas ─────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')

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


# ── Procesar imagen (SIN guardar en Sheets) ───────────────────────────────────

@app.route('/process', methods=['POST'])
def process():
    """
    Solo detecta respuestas. NO guarda en Sheets.
    Acepta 'current_answers' (JSON) con las respuestas ya acumuladas
    de fotos anteriores. Combina: los '?' se rellenan con la foto nueva.
    """
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No se recibió imagen'}), 400

    file = request.files['image']
    if not file.filename or not allowed_file(file.filename):
        return jsonify({'success': False,
                        'error': 'Formato no soportado. Usa JPG, PNG o WEBP'}), 400

    # Respuestas acumuladas de fotos anteriores (pueden llegar vacías)
    current_raw = request.form.get('current_answers', '')
    try:
        current = json.loads(current_raw) if current_raw else []
    except Exception:
        current = []

    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = process_exam_image(filepath)
        os.remove(filepath)

        if not result['success']:
            return jsonify({'success': False, 'error': result['error']}), 422

        new_answers = result['answers']   # lista de 125

        # Combinar: mantener respuesta existente si no es '?',
        # completar con la nueva foto donde había '?'
        if current and len(current) == 125:
            merged   = current[:]
            filled   = []   # índices que se completaron en esta foto
            for i in range(125):
                if merged[i] == '?' and new_answers[i] != '?':
                    merged[i] = new_answers[i]
                    filled.append(i + 1)   # número de pregunta (1-based)
        else:
            merged = new_answers
            filled = [i + 1 for i, a in enumerate(merged) if a != '?']

        detected   = len([a for a in merged if a != '?'])
        missing    = 125 - detected
        new_filled = len(filled) if current else 0   # solo cuenta las nuevas en fotos extra

        return jsonify({
            'success':    True,
            'answers':    merged,
            'confidence': result['confidence'],
            'detected':   detected,
            'missing':    missing,
            'new_filled': new_filled,   # cuántas se rescataron en esta foto extra
            'filled_qs':  filled[:20],  # primeras 20 para el aviso (no spamear)
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error interno: {str(e)}'}), 500


# ── Guardar en Sheets (autorizado por el usuario) ─────────────────────────────

@app.route('/save', methods=['POST'])
def save():
    """
    Recibe el acumulado final de respuestas y lo guarda en Sheets.
    Incluye conteo de correctas si la hoja tiene clave definida.
    """
    data = request.get_json(silent=True) or {}

    student_name = data.get('student_name', '').strip()
    exam_id      = data.get('exam_id', '').strip()
    answers      = data.get('answers', [])
    sheet_name   = data.get('sheet_name', active_sheet())

    if not student_name:
        return jsonify({'success': False,
                        'error': 'El nombre del estudiante es obligatorio.'}), 400
    n = len(answers)
    if n < 1 or n > 125:
        return jsonify({'success': False,
                        'error': 'Número de respuestas inválido (1-125).'}), 400

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
            msg = (f'Guardado en "{sheet_name}" — '
                   f'{correct}/{n} correctas ({pct})')
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


@app.route('/save_key', methods=['POST'])
def save_key():
    """
    Guarda o actualiza la clave de respuestas correctas en fila 2 de la hoja.
    Body JSON: { answers: [...], sheet_name: "..." }
    """
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


@app.route('/debug_image', methods=['POST'])
def debug_image():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No se recibió imagen'}), 400
    file     = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_' + filename)
    file.save(filepath)

    result     = process_exam_image(filepath, debug=True)
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
