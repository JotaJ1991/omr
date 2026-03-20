"""
omr_processor.py  —  v9  (multi-perfil)
========================================
Toda la geometría viene del perfil activo (exam_profiles.py).
Sin constantes fijas — 100 % configurable por tipo de examen.
"""

import cv2
import numpy as np
import os
from scipy.signal import find_peaks
from exam_profiles import get_profile, DEFAULT_PROFILE_ID


# ===========================================================================
# API PUBLICA
# ===========================================================================

def process_exam_image(image_path: str,
                       profile_id: str = DEFAULT_PROFILE_ID,
                       debug: bool = False) -> dict:
    profile = get_profile(profile_id)

    img = cv2.imread(image_path)
    if img is None:
        return {'success': False, 'error': 'No se pudo abrir la imagen.'}

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    warped, p_ok = _correct_perspective(gray, profile)

    # CLAHE opcional según el perfil
    if profile.get('clahe_clip', 0) > 0:
        warped = _apply_clahe(warped,
                              profile['clahe_clip'],
                              profile['clahe_grid'])

    binary         = _binarize(warped,
                               profile['binarize_block'],
                               profile['binarize_c'])
    y_rows, n_rows = _detect_rows(binary, profile)
    answers, confs = _read_answers(binary, y_rows, profile)

    if debug:
        _save_debug(warped, binary, answers, y_rows, profile, image_path)

    total_q  = profile['total_q']
    detected = len([a for a in answers if a != '?'])

    return {
        'success':               True,
        'answers':               answers,       # lista de total_q elementos
        'confidence':            round(float(np.mean(confs)) if confs else 0, 1),
        'rows_detected':         n_rows,
        'perspective_corrected': p_ok,
        'total_q':               total_q,
        'profile_id':            profile_id,
    }


# ===========================================================================
# CLAHE
# ===========================================================================

def _apply_clahe(gray, clip, grid):
    clahe     = cv2.createCLAHE(clipLimit=clip, tileGridSize=grid)
    equalized = clahe.apply(gray)
    return cv2.addWeighted(equalized, 0.70, gray, 0.30, 0)


# ===========================================================================
# BINARIZACIÓN
# ===========================================================================

def _binarize(gray, block, c):
    return cv2.adaptiveThreshold(
        gray, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=block, C=c
    )


# ===========================================================================
# CORRECCIÓN DE PERSPECTIVA  (igual para todos los perfiles)
# ===========================================================================

def _find_sheet_corners(gray):
    h, w    = gray.shape
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


def _correct_perspective(gray, profile):
    W, H = profile['work_w'], profile['work_h']
    h, w = gray.shape

    corners = _find_sheet_corners(gray)
    if corners is not None:
        tl, tr, br, bl = corners
        src = np.float32([tl, tr, br, bl])
        dst = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
        M   = cv2.getPerspectiveTransform(src, dst)
        return cv2.warpPerspective(gray, M, (W, H)), True

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
        result = _find_corners_from_cands(cands, w, h)
        if result is not None:
            tl, tr, br, bl = result
            src = np.float32([tl, tr, br, bl])
            dst = np.float32([[0,0],[W-1,0],[W-1,H-1],[0,H-1]])
            M   = cv2.getPerspectiveTransform(src, dst)
            return cv2.warpPerspective(gray, M, (W, H)), True

    return cv2.resize(gray, (W, H)), False


def _find_corners_from_cands(cands, w, h):
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
    tl = best(q['tl'],0,0); tr = best(q['tr'],w,0)
    br = best(q['br'],w,h); bl = best(q['bl'],0,h)
    return ([tl['cx'],tl['cy']], [tr['cx'],tr['cy']],
            [br['cx'],br['cy']], [bl['cx'],bl['cy']])


# ===========================================================================
# DETECCIÓN DE FILAS (timing marks)
# ===========================================================================

def _detect_rows(binary, profile):
    """
    Detecta las Y de cada fila usando los timing marks.

    Estrategia:
      1. Buscar picos con find_peaks en la franja de cada timing mark.
      2. Si detecta suficientes picos, estimar el paso real entre filas
         promediando los intervalos entre picos consecutivos.
      3. Distribuir las filas uniformemente dentro de [y_top, y_bottom]
         usando ese paso, anclando al primer pico detectado si es fiable,
         o al centro del rango si no hay datos.

    Esto garantiza espaciado uniforme y que las 35 filas queden
    exactamente dentro de los límites establecidos.
    """
    h, w     = binary.shape
    y_top    = int(profile['answers_top_f']    * h)
    y_bottom = int(profile['answers_bottom_f'] * h)

    max_rows = max(col['q_end'] - col['q_start'] + 1
                   for col in profile['columns'])

    best_peaks = []
    best_count = 0

    for col in profile['columns']:
        x_tm = int(col['timing_fx'] * w)
        x1   = max(0, x_tm - 3)
        x2   = min(w, x_tm + 18)

        strip     = binary[y_top:y_bottom, x1:x2].astype(np.float32)
        profile_v = strip.mean(axis=1)
        kernel    = np.ones(5) / 5
        profile_s = np.convolve(profile_v, kernel, mode='same')

        n_rows   = col['q_end'] - col['q_start'] + 1
        min_dist = max(8, int((y_bottom - y_top) / (n_rows + 5)))
        peaks, _ = find_peaks(
            profile_s,
            height=profile_s.max() * 0.35,
            distance=min_dist
        )

        if len(peaks) > best_count:
            best_count = len(peaks)
            best_peaks = [y_top + int(p) for p in peaks]

    # ── Distribución de filas ─────────────────────────────────────────────────
    if max_rows <= 25:
        # Lógica original — funciona bien para JMR-125
        if len(best_peaks) >= max_rows:
            step = (best_peaks[-1] - best_peaks[0]) / (max_rows - 1)
            best_rows = [int(best_peaks[0] + i * step) for i in range(max_rows)]
        elif len(best_peaks) >= max_rows // 2:
            step = (best_peaks[-1] - best_peaks[0]) / (len(best_peaks) - 1)
            best_rows = [int(best_peaks[0] + i * step) for i in range(max_rows)]
        else:
            step = (y_bottom - y_top) / (max_rows - 1)
            best_rows = [int(y_top + i * step) for i in range(max_rows)]
            best_count = 0
        best_rows = [max(y_top, min(y_bottom, y)) for y in best_rows]
    else:
        # Para perfiles con más filas (SIPAGRE-140): distribución de N+2 puntos
        # y_top y y_bottom son los bordes, las max_rows filas son los interiores
        step = (y_bottom - y_top) / (max_rows + 1)
        best_rows = [int(round(y_top + step * i)) for i in range(1, max_rows + 1)]

    return best_rows[:max_rows], min(best_count, max_rows)


# ===========================================================================
# LECTURA DE BURBUJAS
# ===========================================================================

def _read_answers(binary, y_rows, profile):
    h, w      = binary.shape
    answers   = []
    confs     = []
    radius    = profile['bubble_radius']
    threshold = profile['fill_threshold']
    contrast  = profile['min_contrast']

    for col in profile['columns']:
        xs      = [int(fx * w) for fx in col['bubble_fx']]
        options = col['options']
        n_rows  = col['q_end'] - col['q_start'] + 1

        for row_idx in range(n_rows):
            y = y_rows[row_idx] if row_idx < len(y_rows) else 0
            fills = []
            for x in xs:
                y1  = max(0, y - radius)
                y2  = min(h, y + radius)
                x1  = max(0, x - radius)
                x2  = min(w, x + radius)
                roi = binary[y1:y2, x1:x2]
                fills.append(
                    float(np.sum(roi > 127)) / roi.size if roi.size else 0.0
                )
            ans, conf = _pick_answer(fills, options, threshold, contrast)
            answers.append(ans)
            confs.append(conf)

    return answers, confs


def _pick_answer(fills, options, threshold, min_contrast):
    mx = max(fills)
    mi = fills.index(mx)
    if mx < threshold:
        return '?', 0.0
    sf       = sorted(fills, reverse=True)
    contrast = sf[0] - sf[1]
    if contrast < min_contrast:
        return '?', 5.0
    conf = min(100.0, (contrast / (sf[0] + 1e-6)) * 130)
    return options[mi], round(conf, 1)


# ===========================================================================
# DEBUG
# ===========================================================================

def _save_debug(warped, binary, answers, y_rows, profile, original_path):
    h, w  = warped.shape
    debug = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)

    palette = [
        (50,200,50),(50,50,230),(0,180,230),(200,50,200),
        (230,130,0),(0,180,130),(180,0,180),(100,100,230)
    ]
    radius = profile['bubble_radius']

    # ── Dibujar TODAS las posiciones de burbuja (detectada o no) ─────────────
    q_idx = 0
    for col in profile['columns']:
        xs      = [int(fx * w) for fx in col['bubble_fx']]
        options = col['options']
        n_rows  = col['q_end'] - col['q_start'] + 1

        for row_idx in range(n_rows):
            ans = answers[q_idx] if q_idx < len(answers) else '?'
            y   = y_rows[row_idx] if row_idx < len(y_rows) else 0

            for oi, x in enumerate(xs):
                opt = options[oi]
                if opt == ans:
                    # Detectada: círculo grueso con color
                    clr = palette[oi % len(palette)]
                    cv2.circle(debug, (x, y), radius, clr, 2)
                    cv2.putText(debug, opt, (x-4, y+4),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.28, clr, 1)
                else:
                    # No detectada: círculo fino gris para ver la posición
                    cv2.circle(debug, (x, y), radius, (180, 180, 180), 1)

            q_idx += 1

    # ── Zona de respuestas (top/bottom) — se dibuja primero como referencia ──
    y_top    = int(profile['answers_top_f']    * h)
    y_bottom = int(profile['answers_bottom_f'] * h)
    cv2.line(debug, (0, y_top),    (w, y_top),    (0, 120, 255), 2)
    cv2.line(debug, (0, y_bottom), (w, y_bottom), (0, 120, 255), 2)

    # Líneas horizontales de cada fila — solo dentro del rango visible
    margin = profile.get('bubble_radius', 10) + 2
    for y in y_rows:
        if y_top + margin <= y <= y_bottom - margin:
            cv2.line(debug, (0, y), (w, y), (0, 200, 200), 1)

    # ── Líneas verticales de timing marks ────────────────────────────────────
    for col in profile['columns']:
        x_tm = int(col['timing_fx'] * w)
        cv2.line(debug, (x_tm, y_top), (x_tm, y_bottom), (0, 200, 200), 1)

    # ── Textos informativos ───────────────────────────────────────────────────
    info_lines = [
        f"Perfil: {profile.get('id','?')}",
        f"Filas detectadas: {len(y_rows)}",
        f"top_f={profile['answers_top_f']}  bot_f={profile['answers_bottom_f']}",
        f"fill_thr={profile['fill_threshold']}  contrast={profile['min_contrast']}",
    ]
    for i, txt in enumerate(info_lines):
        cv2.putText(debug, txt, (10, 22 + i * 22),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 120, 255), 1, cv2.LINE_AA)

    base, _ = os.path.splitext(original_path)
    cv2.imwrite(base + '_debug.jpg', debug)


# ---------------------------------------------------------------------------
if __name__ == '__main__':
    import sys
    from exam_profiles import PROFILE_LIST
    if len(sys.argv) < 2:
        print("Uso: python omr_processor.py <foto.jpg> [profile_id]")
        print("Perfiles disponibles:")
        for p in PROFILE_LIST:
            print(f"  {p['id']}  —  {p['name']}")
        sys.exit(1)

    pid = sys.argv[2] if len(sys.argv) > 2 else DEFAULT_PROFILE_ID
    r   = process_exam_image(sys.argv[1], profile_id=pid, debug=True)

    if not r['success']:
        print("ERROR:", r['error']); sys.exit(1)

    print(f"Perfil        : {r['profile_id']}")
    print(f"Perspectiva OK: {r['perspective_corrected']}")
    print(f"Filas timing  : {r['rows_detected']}")
    print(f"Confianza     : {r['confidence']}%")
    answered = sum(1 for a in r['answers'] if a != '?')
    print(f"Respondidas   : {answered}/{r['total_q']}")
    profile = get_profile(pid)
    for col in profile['columns']:
        s   = col['q_start'] - 1
        e   = col['q_end']
        ans = r['answers'][s:e]
        print(f"  P{col['q_start']:3d}-P{col['q_end']:3d}: {' '.join(ans)}")
    print(f"Debug         : {sys.argv[1].rsplit('.',1)[0]}_debug.jpg")
