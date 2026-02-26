"""
OMR App - Calificador de exámenes de selección múltiple
Autor: Generado con Claude
"""
import os
import json
import traceback
from flask import Flask, request, jsonify, render_template, send_from_directory
from werkzeug.utils import secure_filename
from omr_processor import process_exam_image
from sheets_connector import save_to_sheets, get_sheet_data

app = Flask(__name__)
app.config['MAX_CONTENT_LENGTH'] = 16 * 1024 * 1024  # 16MB max
app.config['UPLOAD_FOLDER'] = 'uploads'

ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg', 'webp'}

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS


@app.route('/')
def index():
    return render_template('index.html')


@app.route('/process', methods=['POST'])
def process():
    """Recibe imagen, extrae respuestas y las guarda en Google Sheets."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No se recibió imagen'}), 400

    file = request.files['image']
    student_name = request.form.get('student_name', 'Sin nombre')
    exam_id = request.form.get('exam_id', '')

    if file.filename == '':
        return jsonify({'success': False, 'error': 'Archivo vacío'}), 400

    if not allowed_file(file.filename):
        return jsonify({'success': False, 'error': 'Formato no soportado. Usa JPG, PNG o WEBP'}), 400

    try:
        # Guardar imagen temporalmente
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        # Procesar imagen con OMR
        result = process_exam_image(filepath)

        if not result['success']:
            return jsonify({'success': False, 'error': result['error']}), 422

        answers = result['answers']        # Lista de 125 respuestas (A/B/C/D o '?')
        confidence = result['confidence']  # Nivel de confianza global 0-100

        # Guardar en Google Sheets
        sheets_result = save_to_sheets(
            student_name=student_name,
            exam_id=exam_id,
            answers=answers
        )

        # Limpiar archivo temporal
        os.remove(filepath)

        return jsonify({
            'success': True,
            'answers': answers,
            'confidence': confidence,
            'questions_detected': len([a for a in answers if a != '?']),
            'sheets_row': sheets_result.get('row', None),
            'message': f'Examen procesado. {len([a for a in answers if a != "?"])}/125 preguntas detectadas.'
        })

    except Exception as e:
        traceback.print_exc()
        return jsonify({'success': False, 'error': f'Error interno: {str(e)}'}), 500


@app.route('/results')
def results():
    """Muestra los resultados guardados en Sheets."""
    try:
        data = get_sheet_data()
        return jsonify({'success': True, 'data': data})
    except Exception as e:
        return jsonify({'success': False, 'error': str(e)})


@app.route('/debug_image', methods=['POST'])
def debug_image():
    """Endpoint de depuración: devuelve imagen anotada con las detecciones."""
    if 'image' not in request.files:
        return jsonify({'success': False, 'error': 'No se recibió imagen'}), 400

    file = request.files['image']
    filename = secure_filename(file.filename)
    filepath = os.path.join(app.config['UPLOAD_FOLDER'], 'debug_' + filename)
    file.save(filepath)

    from omr_processor import process_exam_image
    import base64
    result = process_exam_image(filepath, debug=True)

    debug_img_path = filepath.replace('.', '_debug.')
    if os.path.exists(debug_img_path):
        with open(debug_img_path, 'rb') as f:
            img_b64 = base64.b64encode(f.read()).decode()
        os.remove(debug_img_path)

    os.remove(filepath)
    return jsonify({
        'success': result['success'],
        'answers': result.get('answers', []),
        'debug_image': img_b64 if os.path.exists(debug_img_path.replace('.', '_debug.')) else None
    })


if __name__ == '__main__':
    os.makedirs('uploads', exist_ok=True)
    app.run(debug=True, host='0.0.0.0', port=5000)
