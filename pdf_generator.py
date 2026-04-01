"""
pdf_generator.py  —  Genera reportes de resultados SIPAGRE en PDF
=================================================================
Estructura similar al reporte ICFES Saber 11:
  - Datos del estudiante
  - Puntaje global con grafico de percentil
  - Puntaje por prueba con graficos de percentil
"""

import io
import unicodedata
from fpdf import FPDF


def _safe_text(text):
    """Reemplaza caracteres no soportados por Helvetica (latin-1)."""
    if not isinstance(text, str):
        text = str(text)
    # Intentar codificar a latin-1; si falla, normalizar y quitar diacriticos
    try:
        text.encode('latin-1')
        return text
    except UnicodeEncodeError:
        # Normalizar NFD y quitar marcas diacriticas
        nfkd = unicodedata.normalize('NFKD', text)
        cleaned = ''.join(c for c in nfkd if not unicodedata.combining(c))
        # Reemplazar caracteres especiales comunes
        cleaned = cleaned.replace('\u2014', '-')   # em dash
        cleaned = cleaned.replace('\u2013', '-')   # en dash
        cleaned = cleaned.replace('\u2018', "'")   # left single quote
        cleaned = cleaned.replace('\u2019', "'")   # right single quote
        cleaned = cleaned.replace('\u201c', '"')   # left double quote
        cleaned = cleaned.replace('\u201d', '"')   # right double quote
        # Fallback: reemplazar lo que quede fuera de latin-1
        try:
            cleaned.encode('latin-1')
        except UnicodeEncodeError:
            cleaned = cleaned.encode('latin-1', errors='replace').decode('latin-1')
        return cleaned


SUBJECTS = [
    ('Lectura Critica', 'lect'),
    ('Matematicas',     'mat'),
    ('Sociales y Ciudadanas', 'soc'),
    ('Ciencias Naturales',    'nat'),
    ('Ingles',          'ing'),
]

# Puntaje maximo teorico por materia (para escala de percentil)
MAX_SCORES = {'mat': 100, 'lect': 100, 'soc': 100, 'nat': 100, 'ing': 100}


def _calc_percentiles(all_results):
    """
    Calcula el percentil de cada estudiante para cada materia y general.
    Percentil = (# estudiantes con puntaje <= al del estudiante) / total * 100
    """
    fields = ['mat', 'lect', 'soc', 'nat', 'ing', 'general']
    n = len(all_results)
    if n == 0:
        return {}

    # Asegurar que los valores sean numericos (pueden venir como string del JSON)
    for r in all_results:
        for f in fields:
            r[f] = int(r.get(f, 0) or 0)

    # Ordenar valores por campo
    sorted_vals = {}
    for f in fields:
        sorted_vals[f] = sorted(r[f] for r in all_results)

    percentiles = {}
    for r in all_results:
        sid = r['id']
        percentiles[sid] = {}
        for f in fields:
            val = r[f]
            # Contar cuantos tienen puntaje <= al de este estudiante
            count = sum(1 for v in sorted_vals[f] if v <= val)
            percentiles[sid][f] = int(round(count / n * 100))

    return percentiles


def _draw_percentile_bar(pdf, x, y, w, h, percentile):
    """Dibuja una barra de percentil con marcador."""
    # Linea base
    pdf.set_draw_color(150, 150, 150)
    pdf.set_line_width(0.3)
    pdf.line(x, y + h/2, x + w, y + h/2)

    # Marcas de 0 y 100
    pdf.set_font('Helvetica', '', 7)
    pdf.set_text_color(120, 120, 120)
    pdf.text(x - 2, y + h/2 - 3, '0')
    pdf.text(x + w - 2, y + h/2 - 3, '100')

    # Ticks intermedios
    for pct in [25, 50, 75]:
        tx = x + (pct / 100) * w
        pdf.line(tx, y + h/2 - 2, tx, y + h/2 + 2)

    # Marcador del percentil (triangulo)
    px = x + (percentile / 100) * w
    py = y + h/2 + 1
    # Triangulo apuntando arriba
    pdf.set_fill_color(40, 40, 40)
    pdf.polygon(
        [(px - 3, py + 5), (px + 3, py + 5), (px, py)],
        style='F'
    )
    # Valor del percentil
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(40, 40, 40)
    pdf.text(px - 3, py + 11, str(percentile))


def generate_student_pdf(student, percentiles):
    """Genera un PDF para un estudiante individual. Retorna bytes."""
    pdf = FPDF(orientation='P', unit='mm', format='letter')
    pdf.set_auto_page_break(auto=False)
    pdf.add_page()

    pw = pdf.w - 20  # ancho util (margenes 10mm)

    # ── TITULO ────────────────────────────────────────────────────────────────
    pdf.set_fill_color(30, 30, 30)
    pdf.rect(10, 10, pw, 18, 'F')
    pdf.set_font('Helvetica', 'B', 14)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, 12)
    pdf.cell(pw, 7, 'REPORTE DE RESULTADOS ESTUDIANTE', align='C')
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_xy(10, 19)
    pdf.cell(pw, 7, 'SIMULACRO SIPAGRE', align='C')

    # ── DATOS ESTUDIANTE ──────────────────────────────────────────────────────
    pdf.set_text_color(40, 40, 40)
    y = 34
    pdf.set_font('Helvetica', '', 9)

    fields = [
        ('Apellidos y nombres:', _safe_text(student['name'])),
        ('Identificacion:', student['id']),
    ]
    for label, value in fields:
        pdf.set_xy(12, y)
        pdf.set_font('Helvetica', '', 9)
        pdf.cell(40, 6, label)
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(100, 6, str(value))
        y += 7

    # ── RESULTADOS GLOBALES ───────────────────────────────────────────────────
    y += 5
    pdf.set_fill_color(30, 30, 30)
    pdf.rect(10, y, pw, 8, 'F')
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, y + 1)
    pdf.cell(pw, 6, 'RESULTADOS GLOBALES', align='C')

    y += 12
    # Puntaje global - caja
    pdf.set_draw_color(100, 100, 100)
    pdf.set_line_width(0.5)
    pdf.rect(15, y, pw/2 - 10, 28, 'D')

    pdf.set_text_color(40, 40, 40)
    pdf.set_font('Helvetica', '', 9)
    pdf.set_xy(17, y + 2)
    pdf.cell(60, 5, 'PUNTAJE GENERAL')
    pdf.set_font('Helvetica', 'B', 28)
    pdf.set_xy(17, y + 8)
    pdf.cell(pw/2 - 14, 18, str(student['general']), align='C')

    # Percentil global
    pct_gen = percentiles.get('general', 0)
    px = pw/2 + 10
    pdf.set_font('Helvetica', 'B', 9)
    pdf.set_text_color(40, 40, 40)
    pdf.set_xy(px, y)
    pdf.cell(pw/2, 5, 'PERCENTIL GENERAL')
    pdf.set_font('Helvetica', '', 7.5)
    pdf.set_xy(px, y + 5)
    pdf.cell(pw/2, 5, 'Respecto a los evaluados, usted esta aqui.')
    _draw_percentile_bar(pdf, px + 5, y + 12, pw/2 - 20, 14, pct_gen)

    # ── RESULTADOS POR PRUEBA ─────────────────────────────────────────────────
    y += 38
    pdf.set_fill_color(30, 30, 30)
    pdf.rect(10, y, pw, 8, 'F')
    pdf.set_font('Helvetica', 'B', 10)
    pdf.set_text_color(255, 255, 255)
    pdf.set_xy(10, y + 1)
    pdf.cell(pw, 6, 'RESULTADOS POR PRUEBA', align='C')

    y += 12
    # Encabezados de la tabla
    col_w = [60, 40, pw - 100]
    pdf.set_fill_color(220, 220, 220)
    pdf.rect(10, y, pw, 8, 'F')
    pdf.set_font('Helvetica', 'B', 8)
    pdf.set_text_color(40, 40, 40)
    pdf.set_xy(10, y + 1)
    pdf.cell(col_w[0], 6, 'PRUEBA', align='C')
    pdf.cell(col_w[1], 6, 'PUNTAJE', align='C')
    pdf.cell(col_w[2], 6, 'PERCENTIL', align='C')
    y += 10

    for subj_name, key in SUBJECTS:
        score = student.get(key, 0)
        pct   = percentiles.get(key, 0)

        # Fila
        pdf.set_draw_color(200, 200, 200)
        pdf.line(10, y + 22, 10 + pw, y + 22)

        pdf.set_font('Helvetica', '', 9)
        pdf.set_text_color(40, 40, 40)
        pdf.set_xy(12, y + 6)
        pdf.cell(col_w[0] - 4, 6, subj_name)

        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_xy(10 + col_w[0], y + 3)
        pdf.cell(col_w[1], 14, str(score), align='C')

        # Percentil bar
        bar_x = 10 + col_w[0] + col_w[1] + 8
        bar_w = col_w[2] - 16
        _draw_percentile_bar(pdf, bar_x, y + 2, bar_w, 14, pct)

        y += 24

    # ── PIE ───────────────────────────────────────────────────────────────────
    pdf.set_font('Helvetica', 'I', 7)
    pdf.set_text_color(150, 150, 150)
    pdf.set_xy(10, pdf.h - 15)
    pdf.cell(pw, 5, 'Generado automaticamente - Simulacro SIPAGRE', align='C')

    return pdf.output()


def generate_all_pdfs(all_results):
    """
    Genera un PDF con todos los estudiantes (uno por pagina).
    Retorna bytes del PDF.
    """
    percentiles = _calc_percentiles(all_results)

    pdf = FPDF(orientation='P', unit='mm', format='letter')
    pdf.set_auto_page_break(auto=False)

    for student in all_results:
        sid = student['id']
        pcts = percentiles.get(sid, {})

        pdf.add_page()
        pw = pdf.w - 20

        # ── TITULO ────────────────────────────────────────────────────────
        pdf.set_fill_color(30, 30, 30)
        pdf.rect(10, 10, pw, 18, 'F')
        pdf.set_font('Helvetica', 'B', 14)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(10, 12)
        pdf.cell(pw, 7, 'REPORTE DE RESULTADOS ESTUDIANTE', align='C')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_xy(10, 19)
        pdf.cell(pw, 7, 'SIMULACRO SIPAGRE', align='C')

        # ── DATOS ─────────────────────────────────────────────────────────
        y = 34
        pdf.set_text_color(40, 40, 40)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_xy(12, y)
        pdf.cell(40, 6, 'Apellidos y nombres:')
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(100, 6, _safe_text(student['name']))
        y += 7
        pdf.set_font('Helvetica', '', 9)
        pdf.set_xy(12, y)
        pdf.cell(40, 6, 'Identificacion:')
        pdf.set_font('Helvetica', 'B', 9)
        pdf.cell(100, 6, str(student['id']))

        # ── GLOBALES ──────────────────────────────────────────────────────
        y += 12
        pdf.set_fill_color(30, 30, 30)
        pdf.rect(10, y, pw, 8, 'F')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(10, y + 1)
        pdf.cell(pw, 6, 'RESULTADOS GLOBALES', align='C')

        y += 12
        pdf.set_draw_color(100, 100, 100)
        pdf.set_line_width(0.5)
        pdf.rect(15, y, pw/2 - 10, 28, 'D')

        pdf.set_text_color(40, 40, 40)
        pdf.set_font('Helvetica', '', 9)
        pdf.set_xy(17, y + 2)
        pdf.cell(60, 5, 'PUNTAJE GENERAL')
        pdf.set_font('Helvetica', 'B', 28)
        pdf.set_xy(17, y + 8)
        pdf.cell(pw/2 - 14, 18, str(student['general']), align='C')

        pct_gen = pcts.get('general', 0)
        px = pw/2 + 10
        pdf.set_font('Helvetica', 'B', 9)
        pdf.set_text_color(40, 40, 40)
        pdf.set_xy(px, y)
        pdf.cell(pw/2, 5, 'PERCENTIL GENERAL')
        pdf.set_font('Helvetica', '', 7.5)
        pdf.set_xy(px, y + 5)
        pdf.cell(pw/2, 5, 'Respecto a los evaluados, usted esta aqui.')
        _draw_percentile_bar(pdf, px + 5, y + 12, pw/2 - 20, 14, pct_gen)

        # ── POR PRUEBA ────────────────────────────────────────────────────
        y += 38
        pdf.set_fill_color(30, 30, 30)
        pdf.rect(10, y, pw, 8, 'F')
        pdf.set_font('Helvetica', 'B', 10)
        pdf.set_text_color(255, 255, 255)
        pdf.set_xy(10, y + 1)
        pdf.cell(pw, 6, 'RESULTADOS POR PRUEBA', align='C')

        y += 12
        col_w = [60, 40, pw - 100]
        pdf.set_fill_color(220, 220, 220)
        pdf.rect(10, y, pw, 8, 'F')
        pdf.set_font('Helvetica', 'B', 8)
        pdf.set_text_color(40, 40, 40)
        pdf.set_xy(10, y + 1)
        pdf.cell(col_w[0], 6, 'PRUEBA', align='C')
        pdf.cell(col_w[1], 6, 'PUNTAJE', align='C')
        pdf.cell(col_w[2], 6, 'PERCENTIL', align='C')
        y += 10

        for subj_name, key in SUBJECTS:
            score = student.get(key, 0)
            pct   = pcts.get(key, 0)

            pdf.set_draw_color(200, 200, 200)
            pdf.line(10, y + 22, 10 + pw, y + 22)

            pdf.set_font('Helvetica', '', 9)
            pdf.set_text_color(40, 40, 40)
            pdf.set_xy(12, y + 6)
            pdf.cell(col_w[0] - 4, 6, subj_name)

            pdf.set_font('Helvetica', 'B', 14)
            pdf.set_xy(10 + col_w[0], y + 3)
            pdf.cell(col_w[1], 14, str(score), align='C')

            bar_x = 10 + col_w[0] + col_w[1] + 8
            bar_w = col_w[2] - 16
            _draw_percentile_bar(pdf, bar_x, y + 2, bar_w, 14, pct)

            y += 24

        pdf.set_font('Helvetica', 'I', 7)
        pdf.set_text_color(150, 150, 150)
        pdf.set_xy(10, pdf.h - 15)
        pdf.cell(pw, 5, 'Generado automaticamente - Simulacro SIPAGRE', align='C')

    return pdf.output()
