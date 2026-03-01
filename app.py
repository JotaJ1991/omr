"""
OMR App  —  Calificador de exámenes de selección múltiple
"""
import os
import json
import traceback
import base64
from flask import Flask, request, jsonify, render_template, session
from werkzeug.utils import secure_filename
from omr_processor import process_exam_image
from sheets_connector import (
    save_to_sheets, get_sheet_data,
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
    """Hoja activa guardada en sesión. Persiste entre escaneos."""
    return session.get('active_sheet', DEFAULT_SHEET)


# ─────────────────────────────────────────────────────────────────────────────
# RUTAS PRINCIPALES
# ─────────────────────────────────────────────────────────────────────────────

@app.route('/')
def index():
    return render_template('index.html')


@app.route('/active_sheet', methods=['GET'])
def get_active_sheet():
    """Devuelve la hoja activa y la lista completa de hojas."""
    try:
        sheets = list_sheets()
    except Exception as e:
        sheets = []
    return jsonify({
        'active': active_sheet(),
        'sheets': sheets
    })


@app.route('/active_sheet', methods=['POST'])
def set_active_sheet():
    """Cambia la hoja activa (persiste en sesión)."""
    data = request.get_json(silent=True) or {}
    name = data.get('name', '').strip()
    if not name:
        return jsonify({'success': False, 'error': 'Nombre vacío'}), 400
    session['active_sheet'] = name
    session.modified = True
    return jsonify({'success': True, 'active': name})


@app.route('/sheets', methods=['GET'])
def sheets_list():
    """Lista todas las pestañas del Spreadsheet."""
    try:
        return jsonify({'success': True, 'sheets': list_sheets()})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/sheets', methods=['POST'])
def sheets_create():
    """Crea una nueva pestaña y la activa automáticamente."""
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


@app.route('/process', methods=['POST'])
def process():
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No se recibió imagen'}), 400

    file         = request.files['image']
    student_name = request.form.get('student_name', 'Sin nombre')
    exam_id      = request.form.get('exam_id', '')
    sheet_name   = request.form.get('sheet_name', active_sheet())

    if not file.filename or not allowed_file(file.filename):
        return jsonify({'success': False,
                        'error': 'Formato no soportado. Usa JPG, PNG o WEBP'}), 400
    try:
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        result = process_exam_image(filepath)
        if not result['success']:
            return jsonify({'success': False, 'error': result['error']}), 422

        answers    = result['answers']
        confidence = result['confidence']

        sheets_result = save_to_sheets(
            student_name=student_name,
            exam_id=exam_id,
            answers=answers,
            sheet_name=sheet_name
        )
        os.remove(filepath)

        detected = len([a for a in answers if a != '?'])
        return jsonify({
            'success':            True,
            'answers':            answers,
            'confidence':         confidence,
            'questions_detected': detected,
            'sheet_name':         sheet_name,
            'sheets_url':         sheets_result.get('url'),
            'message': f'Examen procesado en "{sheet_name}". '
                       f'{detected}/125 preguntas detectadas.'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error interno: {str(e)}'}), 500


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

    result       = process_exam_image(filepath, debug=True)
    debug_path   = filepath.rsplit('.', 1)[0] + '_debug.jpg'
    img_b64      = None

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
