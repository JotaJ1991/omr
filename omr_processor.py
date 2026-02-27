"""
omr_processor.py  —  v4  (hoja con burbujas rellenas + timing tracks)
======================================================================
Diseñado para la hoja de respuestas nueva con:
  * Burbujas OMR estilo ICFES (circulos grises, sin letra)
  * Timing marks (rectangulos negros) al inicio de cada fila
  * 4 marcadores fiduciales en las esquinas

FLUJO:
  1. Detectar la hoja blanca y recortarla del fondo
  2. Detectar los 4 marcadores fiduciales → corregir perspectiva
  3. Detectar los timing marks → calibrar Y exacta de cada fila
  4. Medir relleno de cada burbuja (A/B/C/D) con ROI circular
  5. La burbuja con mayor relleno Y contraste suficiente = respuesta

VENTAJAS SOBRE LA HOJA ANTERIOR:
  - Burbujas vacias son blancas (sin texto impreso) → contraste maximo
  - Timing marks permiten calibracion automatica por foto
  - No depende de posiciones Y hardcodeadas
"""

import cv2
import numpy as np
import os
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# TAMANIO CANONICO DE TRABAJO
# ---------------------------------------------------------------------------
WORK_W = 1275
WORK_H = 1650

# ---------------------------------------------------------------------------
# POSICIONES X DE LAS BURBUJAS (fraccion de WORK_W)
# Ajustar tras imprimir y fotografiar la primera hoja real
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# POSICIONES X DE LAS BURBUJAS (fraccion de WORK_W)
# Calculado matemáticamente desde la grilla de LaTeX:
# ---------------------------------------------------------------------------
BUBBLE_FX = {
    0: [0.109, 0.143, 0.177, 0.212],   # P1-P25   (Col 0)
    1: [0.289, 0.323, 0.358, 0.392],   # P26-P50  (Col 1)
    2: [0.469, 0.504, 0.538, 0.572],   # P51-P75  (Col 2)
    3: [0.650, 0.684, 0.719, 0.753],   # P76-P100 (Col 3)
    4: [0.830, 0.865, 0.899, 0.933],   # P101-P125(Col 4)
}

# X del timing mark por columna (borde izq del rectangulo)
TIMING_FX = {0: 0.056, 1: 0.236, 2: 0.416, 3: 0.597, 4: 0.777}

# Zona vertical donde estan las respuestas
ANSWERS_TOP_F    = 0.28
ANSWERS_BOTTOM_F = 0.97

# ---------------------------------------------------------------------------
# PARAMETROS DE DETECCION
# Con burbujas blancas vacias el contraste es altisimo:
#   vacia  → fill ~0.05  (solo el borde del circulo)
#   marcada con lapiz → fill ~0.50 o mas
# ---------------------------------------------------------------------------
BUBBLE_RADIUS  = 11
FILL_THRESHOLD = 0.25    # fill minimo de la ganadora
MIN_CONTRAST   = 0.12    # contraste minimo fill_max - fill_2do


# ===========================================================================
# API PUBLICA
# ===========================================================================

def process_exam_image(image_path: str, debug: bool = False) -> dict:
    """
    Procesa foto de hoja y extrae 125 respuestas.

    Returns dict:
      success        (bool)
      answers        (list[str])  — 125 elementos: 'A','B','C','D' o '?'
      confidence     (float)      — promedio 0-100
      rows_detected  (int)        — filas calibradas por timing marks
      error          (str)        — solo si success=False
    """
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': 'No se pudo abrir la imagen.'}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    sheet, _        = _extract_sheet(gray)
    warped, p_ok    = _correct_perspective(sheet)
    binary          = _binarize(warped)
    y_rows, n_rows  = _detect_rows_from_timing(binary)
    answers, confs  = _read_answers(binary, y_rows)

    if debug:
        _save_debug(warped, binary, answers, y_rows, image_path)

    return {
        'success':               True,
        'answers':               answers,
        'confidence':            round(float(np.mean(confs)) if confs else 0, 1),
        'rows_detected':         n_rows,
        'perspective_corrected': p_ok,
    }


# ===========================================================================
# INTERNOS
# ===========================================================================

def _extract_sheet(gray):
    """Recorta hoja blanca del fondo oscuro."""
    h, w = gray.shape
    _, white = cv2.threshold(gray, 100, 255, cv2.THRESH_BINARY)
    cnts, _ = cv2.findContours(white, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return gray, (0, 0)
    lg = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(lg) > 0.70 * h * w:
        return gray, (0, 0)
    sx, sy, sw, sh = cv2.boundingRect(lg)
    return gray[sy:sy+sh, sx:sx+sw], (sx, sy)


def _correct_perspective(gray):
    """Detecta los 4 marcadores fiduciales y corrige perspectiva."""
    h, w = gray.shape
    _, thr = cv2.threshold(gray, 70, 255, cv2.THRESH_BINARY_INV)
    cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 300 or area > 80000:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        asp = bw / bh if bh else 0
        if 0.4 < asp < 2.5 and area / (bw * bh) > 0.55:
            cands.append({'cx': x + bw//2, 'cy': y + bh//2, 'area': area})

    corners = _find_corners(cands, w, h)
    if corners is None:
        return cv2.resize(gray, (WORK_W, WORK_H)), False

    tl, tr, br, bl = corners
    src = np.float32([tl, tr, br, bl])
    dst = np.float32([[0,0],[WORK_W-1,0],[WORK_W-1,WORK_H-1],[0,WORK_H-1]])
    M   = cv2.getPerspectiveTransform(src, dst)
    return cv2.warpPerspective(gray, M, (WORK_W, WORK_H)), True


def _find_corners(cands, w, h):
    if len(cands) < 4:
        return None
    mx, my = w/2, h/2
    def best(lst, px, py):
        return min(lst, key=lambda c: (c['cx']-px)**2+(c['cy']-py)**2)
    q = {
        'tl': [c for c in cands if c['cx']<mx and c['cy']<my],
        'tr': [c for c in cands if c['cx']>mx and c['cy']<my],
        'br': [c for c in cands if c['cx']>mx and c['cy']>my],
        'bl': [c for c in cands if c['cx']<mx and c['cy']>my],
    }
    if not all(q.values()):
        return None
    tl=best(q['tl'],0,0); tr=best(q['tr'],w,0)
    br=best(q['br'],w,h); bl=best(q['bl'],0,h)
    return ([tl['cx'],tl['cy']],[tr['cx'],tr['cy']],
            [br['cx'],br['cy']],[bl['cx'],bl['cy']])


def _binarize(gray):
    """Binarizacion adaptativa optimizada para lapiz sobre papel blanco."""
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=25, C=8
    )


def _detect_rows_from_timing(binary):
    """
    Detecta la Y de las 25 filas usando los timing marks.

    Los timing marks son rectangulos negros (en imagen invertida = blancos).
    Se proyectan verticalmente en la franja X del timing mark.
    """
    h, w = binary.shape
    y_top    = int(ANSWERS_TOP_F * h)
    y_bottom = int(ANSWERS_BOTTOM_F * h)

    best_rows  = []
    best_count = 0

    for col_idx in range(5):
        fx   = TIMING_FX[col_idx]
        x_tm = int(fx * w)
        x1   = max(0, x_tm - 3)
        x2   = min(w, x_tm + 18)

        strip   = binary[y_top:y_bottom, x1:x2].astype(np.float32)
        profile = strip.mean(axis=1)

        # Suavizado de caja
        kernel    = np.ones(5) / 5
        profile_s = np.convolve(profile, kernel, mode='same')

        min_dist = max(10, int((y_bottom - y_top) / 30))
        peaks, _ = find_peaks(
            profile_s,
            height=profile_s.max() * 0.35,
            distance=min_dist
        )

        if len(peaks) > best_count:
            best_count = len(peaks)
            best_rows  = [y_top + int(p) for p in peaks]

    # Ajustar a exactamente 25 filas
    if len(best_rows) >= 25:
        # tomar las 25 mas uniformemente espaciadas
        best_rows = _select_25(best_rows)
    elif len(best_rows) >= 10:
        best_rows = _interpolate_rows(best_rows, 25)
    else:
        # fallback uniforme
        best_rows = _uniform_rows(h)
        best_count = 0

    return best_rows[:25], min(best_count, 25)


def _select_25(rows):
    """De una lista de > 25 filas, selecciona las 25 mas igualmente espaciadas."""
    if len(rows) == 25:
        return rows
    # usar los extremos y tomar 25 puntos interpolados
    step = (rows[-1] - rows[0]) / 24
    return [int(rows[0] + i * step) for i in range(25)]


def _interpolate_rows(rows, n):
    """Interpola n filas a partir de las conocidas."""
    if len(rows) < 2:
        return _uniform_rows(WORK_H)[:n]
    step = (rows[-1] - rows[0]) / (len(rows) - 1)
    return [int(rows[0] + i * step) for i in range(n)]


def _uniform_rows(h):
    y_top    = int(ANSWERS_TOP_F * h)
    y_bottom = int(ANSWERS_BOTTOM_F * h)
    step = (y_bottom - y_top) / 24
    return [int(y_top + i * step) for i in range(25)]


def _read_answers(binary, y_rows):
    h, w = binary.shape
    answers, confs = [], []

    for col_idx in range(5):
        xs = [int(fx * w) for fx in BUBBLE_FX[col_idx]]
        for row_idx in range(25):
            y = y_rows[row_idx] if row_idx < len(y_rows) else 0
            fills = []
            for x in xs:
                y1,y2 = max(0,y-BUBBLE_RADIUS), min(h,y+BUBBLE_RADIUS)
                x1,x2 = max(0,x-BUBBLE_RADIUS), min(w,x+BUBBLE_RADIUS)
                roi = binary[y1:y2, x1:x2]
                fills.append(float(np.sum(roi>127))/roi.size if roi.size else 0.0)
            ans, conf = _pick_answer(fills)
            answers.append(ans)
            confs.append(conf)

    return answers, confs


def _pick_answer(fills):
    mx = max(fills)
    mi = fills.index(mx)
    if mx < FILL_THRESHOLD:
        return '?', 0.0
    sf = sorted(fills, reverse=True)
    contrast = sf[0] - sf[1]
    if contrast < MIN_CONTRAST:
        return '?', 5.0
    conf = min(100.0, (contrast / (sf[0]+1e-6)) * 130)
    return 'ABCD'[mi], round(conf, 1)


def _save_debug(warped, binary, answers, y_rows, original_path):
    h, w = warped.shape
    debug = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    colors = {'A':(50,180,50),'B':(50,50,230),'C':(230,130,0),'D':(180,0,180),'?':(80,80,80)}

    for col_idx in range(5):
        xs = [int(fx*w) for fx in BUBBLE_FX[col_idx]]
        for row_idx in range(25):
            q   = col_idx*25+row_idx
            ans = answers[q] if q < len(answers) else '?'
            y   = y_rows[row_idx] if row_idx < len(y_rows) else 0
            clr = colors.get(ans, (80,80,80))
            for oi, x in enumerate(xs):
                if 'ABCD'[oi] == ans:
                    cv2.circle(debug,(x,y),BUBBLE_RADIUS,clr,2)
                    cv2.putText(debug,'ABCD'[oi],(x-5,y+4),
                                cv2.FONT_HERSHEY_SIMPLEX,0.28,clr,1)
                else:
                    cv2.circle(debug,(x,y),BUBBLE_RADIUS,(170,170,170),1)

    for y in y_rows:
        cv2.line(debug,(0,y),(25,y),(0,220,220),1)

    base,_ = os.path.splitext(original_path)
    cv2.imwrite(base+'_debug.jpg', debug)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python omr_processor.py <foto.jpg>")
        sys.exit(1)
    r = process_exam_image(sys.argv[1], debug=True)
    if not r['success']:
        print("ERROR:", r['error']); sys.exit(1)
    print(f"Perspectiva OK: {r['perspective_corrected']}")
    print(f"Filas timing:   {r['rows_detected']}/25")
    print(f"Confianza:      {r['confidence']}%")
    answered = sum(1 for a in r['answers'] if a != '?')
    print(f"Respondidas:    {answered}/125")
    for col in range(5):
        s = col*25; print(f"P{s+1:3d}-P{s+25:3d}: {' '.join(r['answers'][s:s+25])}")
    print(f"Debug: {sys.argv[1].rsplit('.',1)[0]}_debug.jpg")
