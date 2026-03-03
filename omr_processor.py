"""
omr_processor.py  —  v7  (corrección EXIF + parámetros recalibrados)
=====================================================================
  1. Corrección de orientación EXIF  → la foto llega siempre derecha
  2. Binarización adaptativa directa (sin CLAHE)
  3. Parámetros recalibrados
"""

import cv2
import numpy as np
import os
from scipy.signal import find_peaks

# ---------------------------------------------------------------------------
# TAMANIO CANONICO
# ---------------------------------------------------------------------------
WORK_W = 1275
WORK_H = 1650

# ---------------------------------------------------------------------------
# POSICIONES X CALIBRADAS (fraccion de WORK_W)
# ---------------------------------------------------------------------------
BUBBLE_FX = {
    0: [0.110, 0.145, 0.180, 0.215],   # P1-P25
    1: [0.290, 0.325, 0.360, 0.395],   # P26-P50
    2: [0.470, 0.505, 0.540, 0.575],   # P51-P75
    3: [0.650, 0.685, 0.720, 0.755],   # P76-P100
    4: [0.830, 0.865, 0.900, 0.935],   # P101-P125
}

TIMING_FX = {0: 0.056, 1: 0.236, 2: 0.419, 3: 0.601, 4: 0.789}

ANSWERS_TOP_F    = 0.35
ANSWERS_BOTTOM_F = 0.90

# ---------------------------------------------------------------------------
# PARAMETROS DE DETECCION
# ---------------------------------------------------------------------------
BUBBLE_RADIUS  = 10
FILL_THRESHOLD = 0.18
MIN_CONTRAST   = 0.12
BINARIZE_BLOCK = 25
BINARIZE_C     = 8

# ===========================================================================
# API PUBLICA
# ===========================================================================

def process_exam_image(image_path: str, debug: bool = False) -> dict:
    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': 'No se pudo abrir la imagen.'}

    # ── MEJORA 1: corregir orientación EXIF ──────────────────────────────────
    img = _fix_exif_rotation(image_path, img)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    warped, p_ok   = _correct_perspective(gray)
    binary         = _binarize(warped)
    y_rows, n_rows = _detect_rows_from_timing(binary)
    answers, confs = _read_answers(binary, y_rows)

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
# MEJORA 1 — CORRECCIÓN DE ORIENTACIÓN EXIF
# ===========================================================================

def _fix_exif_rotation(image_path: str, img):
    """
    Los celulares guardan la foto con un tag EXIF de orientación.
    OpenCV ignora ese tag y carga la imagen "cruda", que puede aparecer
    girada 90°, 180° o 270°. Esta función la endereza.

    Tag EXIF 0x0112 (Orientation):
      1 = normal          → no hacer nada
      3 = 180°            → rotar 180
      6 = 90° CW          → rotar 90° CCW  (−90)
      8 = 90° CCW         → rotar 90° CW   (+90)
    """
    try:
        # Intentar con Pillow (más confiable para EXIF)
        from PIL import Image
        import io

        pil_img = Image.open(image_path)
        exif    = pil_img._getexif() if hasattr(pil_img, '_getexif') else None

        if exif is None:
            return img

        orientation = exif.get(274)   # 274 = tag Orientation

        if orientation == 3:
            img = cv2.rotate(img, cv2.ROTATE_180)
        elif orientation == 6:
            img = cv2.rotate(img, cv2.ROTATE_90_CLOCKWISE)
        elif orientation == 8:
            img = cv2.rotate(img, cv2.ROTATE_90_COUNTERCLOCKWISE)
        # orientation 1 (o None) → no hacer nada

    except Exception:
        # Si Pillow no está o no hay EXIF, continuar sin rotar
        pass

    return img


# ===========================================================================
# CORRECCION DE PERSPECTIVA
# ===========================================================================

def _find_sheet_corners(gray):
    h, w = gray.shape
    blurred = cv2.GaussianBlur(gray, (21, 21), 0)
    _, thr  = cv2.threshold(blurred, 100, 255, cv2.THRESH_BINARY)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (15, 15))
    closed  = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
    cnts, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    page = max(cnts, key=cv2.contourArea)
    if cv2.contourArea(page) < 0.15 * h * w:
        return None
    peri   = cv2.arcLength(page, True)
    approx = cv2.approxPolyDP(page, 0.02 * peri, True)
    pts    = approx.reshape(-1, 2).astype(np.float32) if len(approx) == 4 \
             else cv2.boxPoints(cv2.minAreaRect(page)).astype(np.float32)
    s    = pts.sum(axis=1)
    diff = np.diff(pts, axis=1).flatten()
    tl = pts[np.argmin(s)];  br = pts[np.argmax(s)]
    tr = pts[np.argmin(diff)]; bl = pts[np.argmax(diff)]
    if np.linalg.norm(tr - tl) < 50 or np.linalg.norm(bl - tl) < 50:
        return None
    return tl, tr, br, bl


def _correct_perspective(gray):
    h, w = gray.shape

    # Estrategia 1: contorno de la hoja blanca
    corners = _find_sheet_corners(gray)
    if corners is not None:
        tl, tr, br, bl = corners
        src = np.float32([tl, tr, br, bl])
        dst = np.float32([[0,0],[WORK_W-1,0],[WORK_W-1,WORK_H-1],[0,WORK_H-1]])
        M   = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(gray, M, (WORK_W, WORK_H)), True

    # Estrategia 2: fiduciales negros con varios umbrales
    for thresh_val in [50, 70, 90, 110]:
        _, thr = cv2.threshold(gray, thresh_val, 255, cv2.THRESH_BINARY_INV)
        kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))
        thr    = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
        cnts, _ = cv2.findContours(thr, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        cands = []
        for c in cnts:
            area = cv2.contourArea(c)
            if area < 200 or area > 200000:
                continue
            x, y, bw, bh = cv2.boundingRect(c)
            asp = bw / bh if bh else 0
            if 0.3 < asp < 3.5 and area / (bw * bh) > 0.40:
                cands.append({'cx': x+bw//2, 'cy': y+bh//2, 'area': area})
        result = _find_corners(cands, w, h)
        if result is not None:
            tl, tr, br, bl = result
            src = np.float32([tl, tr, br, bl])
            dst = np.float32([[0,0],[WORK_W-1,0],[WORK_W-1,WORK_H-1],[0,WORK_H-1]])
            M   = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(gray, M, (WORK_W, WORK_H)), True

    return cv2.resize(gray, (WORK_W, WORK_H)), False


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
    return ([tl['cx'],tl['cy']], [tr['cx'],tr['cy']],
            [br['cx'],br['cy']], [bl['cx'],bl['cy']])


# ===========================================================================
# BINARIZACION
# ===========================================================================

def _binarize(gray):
    """
    Umbralización adaptativa gaussiana sobre la imagen ya mejorada por CLAHE.
    blockSize: tamaño del vecindario local (px). Debe ser impar.
    C: constante que se resta al umbral calculado. Mayor = más estricto.
    """
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=BINARIZE_BLOCK, C=BINARIZE_C
    )


# ===========================================================================
# DETECCION DE FILAS (TIMING MARKS)
# ===========================================================================

def _detect_rows_from_timing(binary):
    h, w = binary.shape
    y_top    = int(ANSWERS_TOP_F * h)
    y_bottom = int(ANSWERS_BOTTOM_F * h)

    best_rows  = []
    best_count = 0

    for col_idx in range(5):
        x_tm = int(TIMING_FX[col_idx] * w)
        x1   = max(0, x_tm - 3)
        x2   = min(w, x_tm + 18)

        strip     = binary[y_top:y_bottom, x1:x2].astype(np.float32)
        profile   = strip.mean(axis=1)
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

    if len(best_rows) >= 25:
        step = (best_rows[-1] - best_rows[0]) / 24
        best_rows = [int(best_rows[0] + i * step) for i in range(25)]
    elif len(best_rows) >= 10:
        step = (best_rows[-1] - best_rows[0]) / (len(best_rows) - 1)
        best_rows = [int(best_rows[0] + i * step) for i in range(25)]
    else:
        step = (y_bottom - y_top) / 24
        best_rows = [int(y_top + i * step) for i in range(25)]
        best_count = 0

    return best_rows[:25], min(best_count, 25)


# ===========================================================================
# LECTURA DE BURBUJAS
# ===========================================================================

def _read_answers(binary, y_rows):
    h, w = binary.shape
    answers, confs = [], []

    for col_idx in range(5):
        xs = [int(fx * w) for fx in BUBBLE_FX[col_idx]]
        for row_idx in range(25):
            y = y_rows[row_idx] if row_idx < len(y_rows) else 0
            fills = []
            for x in xs:
                y1 = max(0, y - BUBBLE_RADIUS)
                y2 = min(h, y + BUBBLE_RADIUS)
                x1 = max(0, x - BUBBLE_RADIUS)
                x2 = min(w, x + BUBBLE_RADIUS)
                roi = binary[y1:y2, x1:x2]
                fills.append(float(np.sum(roi > 127)) / roi.size if roi.size else 0.0)
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
    conf = min(100.0, (contrast / (sf[0] + 1e-6)) * 130)
    return 'ABCD'[mi], round(conf, 1)


# ===========================================================================
# DEBUG
# ===========================================================================

def _save_debug(warped, binary, answers, y_rows, original_path):
    h, w = warped.shape
    debug  = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    colors = {
        'A': (50, 200, 50), 'B': (50, 50, 230),
        'C': (0, 180, 230),  'D': (200, 50, 200),
        '?': (80, 80, 80),
    }
    for col_idx in range(5):
        xs = [int(fx * w) for fx in BUBBLE_FX[col_idx]]
        for row_idx in range(25):
            q   = col_idx * 25 + row_idx
            ans = answers[q] if q < len(answers) else '?'
            y   = y_rows[row_idx] if row_idx < len(y_rows) else 0
            clr = colors.get(ans, (80, 80, 80))
            for oi, x in enumerate(xs):
                if 'ABCD'[oi] == ans:
                    cv2.circle(debug, (x, y), BUBBLE_RADIUS, clr, 2)
                    cv2.putText(debug, 'ABCD'[oi], (x-5, y+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.3, clr, 1)
                else:
                    cv2.circle(debug, (x, y), BUBBLE_RADIUS, (160, 160, 160), 1)
    for y in y_rows:
        cv2.line(debug, (0, y), (30, y), (0, 220, 220), 1)
    base, _ = os.path.splitext(original_path)
    cv2.imwrite(base + '_debug.jpg', debug)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    if len(sys.argv) < 2:
        print("Uso: python omr_processor.py <foto.jpg>")
        sys.exit(1)
    r = process_exam_image(sys.argv[1], debug=True)
    if not r['success']:
        print("ERROR:", r['error']); sys.exit(1)
    print(f"Perspectiva OK : {r['perspective_corrected']}")
    print(f"Filas timing   : {r['rows_detected']}/25")
    print(f"Confianza      : {r['confidence']}%")
    answered = sum(1 for a in r['answers'] if a != '?')
    print(f"Respondidas    : {answered}/125")
    for col in range(5):
        s = col * 25
        print(f"P{s+1:3d}-P{s+25:3d}: {' '.join(r['answers'][s:s+25])}")
    print(f"Debug          : {sys.argv[1].rsplit('.',1)[0]}_debug.jpg")
