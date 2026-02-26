"""
omr_processor.py — CALIBRADO con imagen real (10/10 respuestas de referencia correctas)
----------------------------------------------------------------------------------------
Valores medidos directamente sobre la hoja impresa de José Mercado.

DISCRIMINADOR CLAVE:
  Las burbujas NO marcadas igual tienen relleno alto (~0.30) porque el texto
  impreso "A B C D" es oscuro. El discriminador real NO es el fill máximo sino
  el CONTRASTE entre la burbuja más oscura y la segunda: una burbuja marcada
  a mano tiene mucho más relleno que las otras (contrast > 0.055).
"""

import cv2
import numpy as np
import os

# ─────────────────────────────────────────────────────────────────────────────
# TAMAÑO INTERNO DE TRABAJO
# ─────────────────────────────────────────────────────────────────────────────
WORK_W = 1328
WORK_H = 1675

# ─────────────────────────────────────────────────────────────────────────────
# POSICIONES X ABSOLUTAS — calibradas con imagen real
# Medidas: P13=D(204), P34=C(425), P54=C(684), P69=B(642),
#          P88=A(856), P104=C(1221), P118=D(1276)
# ─────────────────────────────────────────────────────────────────────────────
BUBBLE_X = {
    0: [66,  112, 158, 204],   # Col 1: P1–P25
    1: [333, 379, 425, 471],   # Col 2: P26–P50
    2: [600, 642, 684, 726],   # Col 3: P51–P75
    3: [856, 902, 948, 994],   # Col 4: P76–P100
    4: [1111,1166,1221,1276],  # Col 5: P101–P125
}

# ─────────────────────────────────────────────────────────────────────────────
# POSICIONES Y — calibradas con anclas P13(row12)=1088 y P69(row18)=1351
# Y = 562 + row * 43.83
# ─────────────────────────────────────────────────────────────────────────────
Y_FIRST = 562      # Centro Y de la fila 1 (P1, P26, P51, P76, P101)
Y_STEP  = 43.83    # Píxeles entre filas consecutivas

# ─────────────────────────────────────────────────────────────────────────────
# PARÁMETROS DE DETECCIÓN
# ─────────────────────────────────────────────────────────────────────────────
BUBBLE_RADIUS  = 13     # Radio de muestreo en píxeles
FILL_THRESHOLD = 0.08   # Fill mínimo (umbral bajo — el discriminador real es MIN_CONTRAST)
MIN_CONTRAST   = 0.045  # Diferencia mínima entre burbuja 1ra y 2da
                        # El texto impreso produce contrast < 0.03
                        # Una marca real produce contrast > 0.06


def process_exam_image(image_path: str, debug: bool = False) -> dict:
    """
    Procesa imagen de hoja de respuestas y extrae las 125 respuestas.

    Returns dict con:
      success (bool)
      answers  (list[str]): 125 elementos — 'A','B','C','D' o '?'
      confidence (float):   0–100
      error (str):          mensaje si success=False
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': 'No se pudo leer la imagen.'}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # 1. Recortar hoja blanca del fondo oscuro (si existe)
    sheet_gray, _ = _extract_sheet(gray)

    # 2. Corregir perspectiva con marcadores fiduciales
    warped, perspective_ok = _correct_perspective(sheet_gray)

    # 3. Binarizar adaptativamente
    binary = cv2.adaptiveThreshold(
        warped, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=31, C=10
    )

    # 4. Posiciones Y de las 25 filas
    y_positions = [int(Y_FIRST + i * Y_STEP) for i in range(25)]

    # 5. Leer respuestas
    answers, confidences = _read_answers(binary, y_positions)

    # 6. Debug opcional
    if debug:
        _save_debug(warped, answers, y_positions, image_path)

    avg_conf = float(np.mean(confidences)) if confidences else 0.0

    return {
        'success': True,
        'answers': answers,
        'confidence': round(avg_conf, 1),
        'perspective_corrected': perspective_ok
    }


# ─────────────────────────────────────────────────────────────────────────────
# FUNCIONES INTERNAS
# ─────────────────────────────────────────────────────────────────────────────

def _extract_sheet(gray: np.ndarray):
    """Recorta la hoja blanca si hay fondo oscuro visible."""
    h, w = gray.shape
    _, white = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    contours, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return gray, (0, 0)

    largest = max(contours, key=cv2.contourArea)
    if cv2.contourArea(largest) > 0.70 * h * w:
        return gray, (0, 0)

    sx, sy, sw, sh = cv2.boundingRect(largest)
    return gray[sy:sy+sh, sx:sx+sw], (sx, sy)


def _correct_perspective(gray: np.ndarray):
    """
    Detecta los 4 marcadores fiduciales negros (cuadrados sólidos en esquinas)
    y aplana la perspectiva.
    """
    h, w = gray.shape
    _, thresh = cv2.threshold(gray, 80, 255, cv2.THRESH_BINARY_INV)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    candidates = []
    for cnt in contours:
        area = cv2.contourArea(cnt)
        if area < 200 or area > 50000:
            continue
        x, y, bw, bh = cv2.boundingRect(cnt)
        aspect = bw / bh if bh > 0 else 0
        if 0.35 < aspect < 2.8 and (area / (bw * bh)) > 0.45:
            candidates.append({'cx': x + bw//2, 'cy': y + bh//2, 'area': area})

    corners = _find_corners(candidates, w, h)
    if corners is None:
        return cv2.resize(gray, (WORK_W, WORK_H)), False

    tl, tr, br, bl = corners
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0,0],[WORK_W-1,0],[WORK_W-1,WORK_H-1],[0,WORK_H-1]])
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (WORK_W, WORK_H)), True


def _find_corners(candidates, img_w, img_h):
    """Identifica TL, TR, BR, BL."""
    if len(candidates) < 4:
        return None
    cx_mid, cy_mid = img_w / 2, img_h / 2

    def closest(cands, px, py):
        return min(cands, key=lambda c: (c['cx']-px)**2 + (c['cy']-py)**2)

    quads = {
        'tl': [c for c in candidates if c['cx'] < cx_mid and c['cy'] < cy_mid],
        'tr': [c for c in candidates if c['cx'] > cx_mid and c['cy'] < cy_mid],
        'br': [c for c in candidates if c['cx'] > cx_mid and c['cy'] > cy_mid],
        'bl': [c for c in candidates if c['cx'] < cx_mid and c['cy'] > cy_mid],
    }
    if not all(quads.values()):
        return None

    tl = closest(quads['tl'], 0, 0)
    tr = closest(quads['tr'], img_w, 0)
    br = closest(quads['br'], img_w, img_h)
    bl = closest(quads['bl'], 0, img_h)
    return [tl['cx'],tl['cy']], [tr['cx'],tr['cy']], [br['cx'],br['cy']], [bl['cx'],bl['cy']]


def _read_answers(binary: np.ndarray, y_positions: list):
    """Lee el relleno de cada burbuja y determina las respuestas."""
    answers, confidences = [], []

    for col_idx in range(5):
        xs = BUBBLE_X[col_idx]
        for row_idx in range(25):
            y = y_positions[row_idx]
            fills = []
            for x in xs:
                y1 = max(0, y - BUBBLE_RADIUS)
                y2 = min(WORK_H, y + BUBBLE_RADIUS)
                x1 = max(0, x - BUBBLE_RADIUS)
                x2 = min(WORK_W, x + BUBBLE_RADIUS)
                roi = binary[y1:y2, x1:x2]
                fills.append(np.sum(roi > 127) / roi.size if roi.size > 0 else 0.0)

            answer, conf = _pick_answer(fills)
            answers.append(answer)
            confidences.append(conf)

    return answers, confidences


def _pick_answer(fills: list):
    """
    Elige la respuesta marcada.

    El discriminador clave es el CONTRASTE (fill_max - fill_2nd), no el fill absoluto.
    El texto impreso produce fills uniformes (~0.30 en todas las opciones), 
    mientras que una marca real hace que UNA burbuja sea notablemente más oscura.
    """
    options  = ['A', 'B', 'C', 'D']
    max_fill = max(fills)
    max_idx  = fills.index(max_fill)

    if max_fill < FILL_THRESHOLD:
        return '?', 0.0

    sorted_f = sorted(fills, reverse=True)
    contrast = sorted_f[0] - sorted_f[1]

    if contrast < MIN_CONTRAST:
        return '?', 0.0   # Fills demasiado uniformes → texto impreso, no marca

    # Confianza: contraste como fracción del fill máximo, escalado a 0-100
    confidence = min(100.0, (contrast / sorted_f[0]) * 150)
    return options[max_idx], round(confidence, 1)


def _save_debug(warped, answers, y_positions, original_path):
    """Guarda imagen anotada con las detecciones."""
    debug  = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    colors = {
        'A': (255, 80,  80),
        'B': (80,  200, 80),
        'C': (80,  80,  255),
        'D': (200, 200, 0),
        '?': (50,  50,  50),
    }

    for col_idx in range(5):
        xs  = BUBBLE_X[col_idx]
        for row_idx in range(25):
            q   = col_idx * 25 + row_idx
            ans = answers[q]
            y   = y_positions[row_idx]
            clr = colors.get(ans, (128, 128, 128))

            for oi, x in enumerate(xs):
                if 'ABCD'[oi] == ans:
                    cv2.circle(debug, (x, y), BUBBLE_RADIUS, clr, 2)
                    cv2.putText(debug, 'ABCD'[oi], (x-5, y+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, clr, 1)
                else:
                    cv2.circle(debug, (x, y), BUBBLE_RADIUS, (160, 160, 160), 1)

    base, _ = os.path.splitext(original_path)
    cv2.imwrite(base + '_debug.jpg', debug)
