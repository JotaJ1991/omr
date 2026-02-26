"""
omr_processor.py
----------------
Motor de reconocimiento óptico de marcas (OMR) para hoja de respuestas de 125 preguntas.

CÓMO FUNCIONA:
1. Detecta los 4 marcadores fiduciales negros en las esquinas
2. Corrige la perspectiva (aplana la imagen aunque esté torcida)
3. Divide el área de respuestas en una grilla de 125 filas × 4 columnas
4. Para cada burbuja mide el % de píxeles oscuros
5. La burbuja con más relleno = respuesta seleccionada
"""

import cv2
import numpy as np
import os


# ─────────────────────────────────────────────────────────────────────────────
# CONSTANTES DE CONFIGURACIÓN — ajusta según tu impresión real
# ─────────────────────────────────────────────────────────────────────────────

# Tamaño interno de trabajo después de corregir perspectiva (píxeles)
WORK_W = 1240   # ≈ ancho carta a 120 dpi
WORK_H = 1754   # ≈ alto carta a 120 dpi

# Umbral de relleno: si la burbuja tiene más del X% de píxeles oscuros → marcada
FILL_THRESHOLD = 0.18  # 18%  (ajusta si hay falsos positivos/negativos)

# Umbral para considerar que hay DOBLE marca (respuesta anulada)
DOUBLE_THRESHOLD = 0.12

# ─────────────────────────────────────────────────────────────────────────────
# REGIÓN DE LA GRILLA DE RESPUESTAS (en fracción del ancho/alto de WORK)
# Estos valores corresponden al layout del LaTeX proporcionado.
# Si el resultado es incorrecto, ajusta estos márgenes.
# ─────────────────────────────────────────────────────────────────────────────

# El área útil de respuestas (excluyendo encabezado, márgenes, etc.)
GRID_TOP    = 0.285   # 28.5% desde arriba
GRID_BOTTOM = 0.875   # 97.5% desde arriba
GRID_LEFT   = 0.3   # 3% desde la izquierda
GRID_RIGHT  = 0.875   # 97.5% desde la izquierda

# Número de columnas de preguntas en el multicol
N_COLS_PAGE = 5        # 5 columnas de preguntas en el LaTeX
QUESTIONS_PER_COL = 25 # 125 / 5 = 25 preguntas por columna
N_OPTIONS = 4          # A, B, C, D


def process_exam_image(image_path: str, debug: bool = False) -> dict:
    """
    Procesa una imagen de hoja de respuestas y extrae las 125 respuestas.

    Args:
        image_path: Ruta a la imagen (jpg/png/webp)
        debug: Si True, guarda imagen anotada con sufijo _debug

    Returns:
        dict con:
          success (bool)
          answers (list[str]): 125 elementos, cada uno 'A','B','C','D' o '?'
          confidence (float): 0-100
          error (str): mensaje de error si success=False
    """
    # ── 1. Cargar imagen ──────────────────────────────────────────────────────
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': 'No se pudo leer la imagen. ¿Formato inválido?'}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # ── 2. Detectar y corregir perspectiva ────────────────────────────────────
    warped = _correct_perspective(gray)
    if warped is None:
        # Si no se encontraron marcadores, intentar con la imagen completa
        warped = cv2.resize(gray, (WORK_W, WORK_H))
        perspective_ok = False
    else:
        perspective_ok = True

    # ── 3. Binarizar ──────────────────────────────────────────────────────────
    # Umbralización adaptativa para manejar iluminación desigual
    binary = cv2.adaptiveThreshold(
        warped, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=10
    )

    # ── 4. Extraer respuestas de la grilla ───────────────────────────────────
    answers, confidences, bubble_coords = _extract_answers(binary)

    # ── 5. Calcular confianza global ──────────────────────────────────────────
    avg_confidence = float(np.mean(confidences)) if confidences else 0.0

    # ── 6. Debug: dibujar anotaciones ────────────────────────────────────────
    if debug:
        _save_debug_image(warped, answers, bubble_coords, image_path)

    return {
        'success': True,
        'answers': answers,
        'confidence': round(avg_confidence, 1),
        'perspective_corrected': perspective_ok
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES INTERNAS
# ─────────────────────────────────────────────────────────────────────────────

def _correct_perspective(gray: np.ndarray) -> np.ndarray | None:
    """
    Detecta los 4 marcadores fiduciales cuadrados negros en las esquinas
    y aplica transformación de perspectiva.
    """
    h, w = gray.shape

    # Binarización global para detectar los cuadrados negros
    _, thresh = cv2.threshold(gray, 60, 255, cv2.THRESH_BINARY_INV)

    # Encontrar contornos
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Filtrar contornos que parezcan cuadrados/rectángulos de tamaño correcto
    # Los marcadores del LaTeX son aprox 0.5cm × 0.5cm en carta
    # En una foto de celular, estimamos entre 30-200 píxeles de lado
    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 500 or area > 30000:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if 0.4 < aspect < 2.5:  # Forma más o menos cuadrada/rectangular
            fill = area / (bw * bh)
            if fill > 0.5:  # Bien relleno (cuadrado negro sólido)
                candidates.append({
                    'cx': x + bw // 2,
                    'cy': y + bh // 2,
                    'area': area,
                    'x': x, 'y': y, 'w': bw, 'h': bh
                })

    if len(candidates) < 4:
        return None

    # Ordenar candidatos por posición para identificar esquinas
    # Usar los 4 más extremos
    corners = _identify_corners(candidates, w, h)
    if corners is None:
        return None

    tl, tr, br, bl = corners

    # Puntos destino (imagen rectificada)
    dst = np.float32([
        [0, 0],
        [WORK_W - 1, 0],
        [WORK_W - 1, WORK_H - 1],
        [0, WORK_H - 1]
    ])

    src = np.float32([tl, tr, br, bl])
    M = cv2.getPerspectiveTransform(src, dst)
    warped = cv2.warpPerspective(gray, M, (WORK_W, WORK_H))
    return warped


def _identify_corners(candidates: list, img_w: int, img_h: int):
    """
    Dado un conjunto de candidatos a marcadores, identifica los 4 de las esquinas.
    Devuelve (top-left, top-right, bottom-right, bottom-left) como arrays [x, y].
    """
    cx_mid = img_w / 2
    cy_mid = img_h / 2

    # Clasificar por cuadrante
    tl_cands = [c for c in candidates if c['cx'] < cx_mid and c['cy'] < cy_mid]
    tr_cands = [c for c in candidates if c['cx'] > cx_mid and c['cy'] < cy_mid]
    br_cands = [c for c in candidates if c['cx'] > cx_mid and c['cy'] > cy_mid]
    bl_cands = [c for c in candidates if c['cx'] < cx_mid and c['cy'] > cy_mid]

    if not (tl_cands and tr_cands and br_cands and bl_cands):
        return None

    # Tomar el más cercano a cada esquina
    def closest_to(cands, px, py):
        return min(cands, key=lambda c: (c['cx'] - px)**2 + (c['cy'] - py)**2)

    tl = closest_to(tl_cands, 0, 0)
    tr = closest_to(tr_cands, img_w, 0)
    br = closest_to(br_cands, img_w, img_h)
    bl = closest_to(bl_cands, 0, img_h)

    return (
        [tl['cx'], tl['cy']],
        [tr['cx'], tr['cy']],
        [br['cx'], br['cy']],
        [bl['cx'], bl['cy']]
    )


def _extract_answers(binary: np.ndarray) -> tuple[list, list, list]:
    """
    Extrae las respuestas analizando el relleno de cada burbuja.

    El layout del LaTeX tiene:
    - 5 columnas de preguntas (multicol{5})
    - 25 preguntas por columna
    - 4 opciones (A,B,C,D) por pregunta
    - Cada columna tiene además la línea vertical de separación (columnseprule)

    Retorna:
        answers: ['A','B','C','D' o '?'] × 125
        confidences: [0-100] × 125
        bubble_coords: lista de (x,y,w,h) para debug
    """
    h, w = binary.shape

    grid_top    = int(GRID_TOP    * h)
    grid_bottom = int(GRID_BOTTOM * h)
    grid_left   = int(GRID_LEFT   * w)
    grid_right  = int(GRID_RIGHT  * w)

    grid_h = grid_bottom - grid_top
    grid_w = grid_right  - grid_left

    # Ancho de cada "super-columna" (columna de preguntas)
    col_w = grid_w / N_COLS_PAGE
    # Alto de cada fila de pregunta dentro de una columna
    row_h = grid_h / QUESTIONS_PER_COL

    # Dentro de cada celda de pregunta, las 4 burbujas se distribuyen
    # horizontalmente. En el LaTeX: \bubble{A}\hspace{0.1cm}\bubble{B}...
    # Aproximamos: las 4 burbujas ocupan ~70% del ancho de la super-columna
    bubble_area_ratio = 0.68  # Fracción del ancho de col dedicada a burbujas
    bubble_w_frac = bubble_area_ratio / N_OPTIONS  # fracción por burbuja

    answers = []
    confidences = []
    bubble_coords = []

    for col_idx in range(N_COLS_PAGE):
        col_x_start = grid_left + col_idx * col_w

        for row_idx in range(QUESTIONS_PER_COL):
            q_num = col_idx * QUESTIONS_PER_COL + row_idx + 1

            # Centro vertical de la fila
            row_y_start = grid_top + row_idx * row_h
            row_y_end   = row_y_start + row_h

            # Márgenes verticales dentro de la fila (burbuja no ocupa todo el alto)
            bubble_margin_v = int(row_h * 0.12)
            b_y1 = int(row_y_start + bubble_margin_v)
            b_y2 = int(row_y_end   - bubble_margin_v)
            if b_y1 >= b_y2:
                b_y2 = b_y1 + 1

            fills = []
            coords = []

            for opt_idx in range(N_OPTIONS):
                # Las burbujas están al FINAL de la celda (alineadas a la derecha del número)
                # El número de pregunta ocupa aprox 30% del ancho
                number_offset = col_w * 0.28
                b_x1 = int(col_x_start + number_offset + opt_idx * (col_w * bubble_w_frac))
                b_x2 = int(b_x1 + col_w * bubble_w_frac * 0.85)

                # Clamp a bordes
                b_x1 = max(0, min(b_x1, w - 1))
                b_x2 = max(0, min(b_x2, w - 1))

                if b_x1 >= b_x2:
                    fills.append(0)
                    coords.append((b_x1, b_y1, 1, b_y2 - b_y1))
                    continue

                # Región de interés (ROI) de la burbuja
                roi = binary[b_y1:b_y2, b_x1:b_x2]
                if roi.size == 0:
                    fills.append(0)
                    coords.append((b_x1, b_y1, b_x2 - b_x1, b_y2 - b_y1))
                    continue

                # Porcentaje de píxeles blancos (=marcas oscuras en invertida)
                fill_ratio = np.sum(roi > 127) / roi.size
                fills.append(fill_ratio)
                coords.append((b_x1, b_y1, b_x2 - b_x1, b_y2 - b_y1))

            # Determinar respuesta
            answer, confidence = _determine_answer(fills)
            answers.append(answer)
            confidences.append(confidence)
            bubble_coords.append(list(zip(['A','B','C','D'], coords, fills)))

    return answers, confidences, bubble_coords


def _determine_answer(fills: list) -> tuple[str, float]:
    """
    Dado el relleno de las 4 burbujas, determina cuál fue marcada.

    Lógica:
    - Si ninguna supera FILL_THRESHOLD → '?' (no marcada)
    - Si más de una supera DOUBLE_THRESHOLD → '?' (doble marca)
    - Caso normal → la de mayor relleno

    Devuelve (letra, confianza_0_a_100)
    """
    options = ['A', 'B', 'C', 'D']
    max_fill = max(fills)
    max_idx  = fills.index(max_fill)

    # ¿Alguna burbuja fue marcada?
    if max_fill < FILL_THRESHOLD:
        return '?', 0.0

    # ¿Hay doble marca? (dos o más burbujas con relleno significativo)
    marked = [f for f in fills if f >= DOUBLE_THRESHOLD]
    if len(marked) > 1:
        return '?', 10.0  # Anulada

    # Calcular confianza: diferencia entre la mayor y la segunda mayor
    sorted_fills = sorted(fills, reverse=True)
    second = sorted_fills[1] if len(sorted_fills) > 1 else 0
    contrast = (max_fill - second) / (max_fill + 1e-6)
    confidence = min(100, contrast * 150)  # Escalar a 0-100

    return options[max_idx], round(confidence, 1)


def _save_debug_image(warped: np.ndarray, answers: list, bubble_coords: list, original_path: str):
    """Guarda imagen a color con rectángulos alrededor de cada burbuja detectada."""
    debug_img = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    colors = {
        'A': (255, 100, 100),  # Azul
        'B': (100, 200, 100),  # Verde
        'C': (100, 100, 255),  # Rojo
        'D': (200, 200, 0),    # Cyan
        '?': (0, 0, 255),      # Rojo puro = no detectado
    }

    for q_idx, (answer, q_bubbles) in enumerate(zip(answers, bubble_coords)):
        for opt_letter, (bx, by, bw, bh), fill in q_bubbles:
            color = colors.get(answer, (128, 128, 128))
            thickness = 2 if opt_letter == answer and answer != '?' else 1
            cv2.rectangle(debug_img, (bx, by), (bx + bw, by + bh), color, thickness)

    # Guardar
    base, ext = os.path.splitext(original_path)
    debug_path = base + '_debug.jpg'
    cv2.imwrite(debug_path, debug_img)
