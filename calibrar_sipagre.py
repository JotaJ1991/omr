"""
calibrar_sipagre.py — Herramienta de calibración interactiva para SIPAGRE-140
==============================================================================
Uso:
    python calibrar_sipagre.py foto.jpg

Controles:
    Clic izquierdo  → registrar punto en el modo actual
    TAB             → avanzar al siguiente modo manualmente
    Z               → deshacer último clic
    R               → reiniciar todos los puntos
    S               → guardar/imprimir parámetros calibrados
    +  /  =         → zoom in
    -               → zoom out
    0               → restablecer zoom (ver imagen completa)
    Flechas         → desplazar la vista cuando hay zoom
    Q  /  ESC       → salir

Flujo de 6 pasos (el script avanza solo al completar cada uno):
    1. TIMING      — clic en el timing mark (rect negro) fila 1 de cada columna → 4 clics
    2. TOP/BOTTOM  — clic en timing mark fila 1 y fila 35 de cualquier columna  → 2 clics
    3. Col 1 A-D   — clic en el centro de A B C D de cualquier fila col 1       → 4 clics
    4. Col 2 A-D   — idem columna 2                                              → 4 clics
    5. Col 3 A-D   — idem columna 3                                              → 4 clics
    6. Col 4 A-H   — clic en A B C D E F G H de cualquier fila col 4            → 8 clics

Al pulsar S se genera sipagre_params_calibrados.txt listo para pegar
en exam_profiles.py.

Requiere: pip install opencv-python numpy
"""

import cv2
import numpy as np
import sys
import os
from datetime import datetime

COLUMNS_DEF = [
    {'id': 0, 'name': 'Col 1 (P1-35)',    'options': ['A','B','C','D'],                    'q_start':   1, 'q_end':  35},
    {'id': 1, 'name': 'Col 2 (P36-70)',   'options': ['A','B','C','D'],                    'q_start':  36, 'q_end':  70},
    {'id': 2, 'name': 'Col 3 (P71-105)',  'options': ['A','B','C','D','E','F','G','H'],    'q_start':  71, 'q_end': 105},
    {'id': 3, 'name': 'Col 4 (P106-140)', 'options': ['A','B','C','D'],                    'q_start': 106, 'q_end': 140},
]

WORK_W = 1275
WORK_H = 1650

MODES = ['TIMING', 'TOP_BOTTOM', 'BUBBLES_C0', 'BUBBLES_C1', 'BUBBLES_C2', 'BUBBLES_C3']

MODE_LABELS = {
    'TIMING':     '1/6 TIMING: clic en rect negro fila-1 de cada columna (4 clics de izq a der)',
    'TOP_BOTTOM': '2/6 TOP/BOT: clic en timing mark fila-1 y fila-35 de cualquier columna (2 clics)',
    'BUBBLES_C0': '3/6 Col1 A-D: clic en centro de A B C D en cualquier fila de col 1 (4 clics)',
    'BUBBLES_C1': '4/6 Col2 A-D: clic en centro de A B C D en cualquier fila de col 2 (4 clics)',
    'BUBBLES_C2': '5/6 Col3 A-H: clic en A B C D E F G H en cualquier fila de col 3 (8 clics)',
    'BUBBLES_C3': '6/6 Col4 A-D: clic en centro de A B C D en cualquier fila de col 4 (4 clics)',
}

MODE_EXPECTED = {'TIMING':4,'TOP_BOTTOM':2,'BUBBLES_C0':4,'BUBBLES_C1':4,'BUBBLES_C2':8,'BUBBLES_C3':4}

COLORS = {
    'TIMING':     (0,220,220),
    'TOP_BOTTOM': (30,165,255),
    'BUBBLES_C0': (50,220,50),
    'BUBBLES_C1': (50,50,230),
    'BUBBLES_C2': (200,50,200),
    'BUBBLES_C3': (0,200,180),
}

HUD_H    = 90
WIN_NAME = 'Calibrador SIPAGRE-140  |  S=guardar  TAB=modo  Z=deshacer  +/-=zoom  0=encajar  Q=salir'

state = {
    'mode':              'TIMING',
    'timing_clicks':     [],
    'top_bottom_clicks': [],
    'bubble_clicks':     {0:[], 1:[], 2:[], 3:[]},
    'history':           [],
    'img_orig':          None,
    'zoom':              1.0,
    'pan_x':             0.0,
    'pan_y':             0.0,
    'win_w':             1200,
    'win_h':             900,
    'base_scale':        1.0,
}


def _find_fiducials(binary, w, h):
    """Misma lógica que omr_processor.py — detecta los 4 cuadrados negros."""
    cnts, _ = cv2.findContours(binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not cnts:
        return None
    cands = []
    for c in cnts:
        area = cv2.contourArea(c)
        if area < 150 or area > 20000:
            continue
        x, y, bw, bh = cv2.boundingRect(c)
        if bw < 8 or bh < 8:
            continue
        asp      = bw / bh if bh else 0
        solidity = area / (bw * bh) if bw * bh > 0 else 0
        if 0.4 < asp < 2.5 and solidity > 0.35:
            cands.append({'cx': x + bw//2, 'cy': y + bh//2,
                          'area': area, 'w': bw, 'h': bh})
    return _assign_corners(cands, w, h)


def _assign_corners(cands, w, h):
    if len(cands) < 4:
        return None
    mx, my = w / 2, h / 2
    def closest(lst, px, py):
        return min(lst, key=lambda c: (c['cx']-px)**2 + (c['cy']-py)**2)
    q = {
        'tl': [c for c in cands if c['cx'] < mx and c['cy'] < my],
        'tr': [c for c in cands if c['cx'] > mx and c['cy'] < my],
        'br': [c for c in cands if c['cx'] > mx and c['cy'] > my],
        'bl': [c for c in cands if c['cx'] < mx and c['cy'] > my],
    }
    if not all(q.values()):
        return None
    tl = closest(q['tl'], 0, 0)
    tr = closest(q['tr'], w, 0)
    br = closest(q['br'], w, h)
    bl = closest(q['bl'], 0, h)
    pts = [(tl['cx'],tl['cy']), (tr['cx'],tr['cy']),
           (br['cx'],br['cy']), (bl['cx'],bl['cy'])]
    width  = max(pts[1][0], pts[2][0]) - min(pts[0][0], pts[3][0])
    height = max(pts[2][1], pts[3][1]) - min(pts[0][1], pts[1][1])
    if width < w * 0.3 or height < h * 0.3:
        return None
    return pts


def correct_perspective(gray):
    """
    Detecta fiduciales y corrige perspectiva.
    No depende del fondo — funciona con cualquier superficie.
    Devuelve (warped, ok, fiducial_pts_original).
    fiducial_pts_original: lista de 4 puntos (x,y) en la imagen ORIGINAL
    para dibujarlos como referencia en el visor.
    """
    h, w    = gray.shape
    blurred = cv2.GaussianBlur(gray, (5, 5), 0)
    kernel  = cv2.getStructuringElement(cv2.MORPH_RECT, (3, 3))

    # Estrategia 1: umbral adaptativo local
    adaptive = cv2.adaptiveThreshold(
        blurred, 255,
        cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
        cv2.THRESH_BINARY_INV,
        blockSize=51, C=10
    )
    adaptive = cv2.morphologyEx(adaptive, cv2.MORPH_CLOSE, kernel)
    corners  = _find_fiducials(adaptive, w, h)

    # Estrategia 2: umbral global multi-nivel
    if corners is None:
        for thresh_val in [40, 60, 80, 100, 120]:
            _, thr  = cv2.threshold(blurred, thresh_val, 255, cv2.THRESH_BINARY_INV)
            thr     = cv2.morphologyEx(thr, cv2.MORPH_CLOSE, kernel)
            corners = _find_fiducials(thr, w, h)
            if corners is not None:
                break

    if corners is not None:
        src = np.float32(corners)
        dst = np.float32([[0,0],[WORK_W-1,0],[WORK_W-1,WORK_H-1],[0,WORK_H-1]])
        M   = cv2.getPerspectiveTransform(src, dst)
        print('[OK] Fiduciales detectados - perspectiva corregida')
        for i, (px, py) in enumerate(corners):
            labels = ['TL','TR','BR','BL']
            print(f'   {labels[i]}: ({px}, {py})')
        return cv2.warpPerspective(gray, M, (WORK_W, WORK_H)), True, corners

    print('[WARN] Fiduciales no detectados - usando imagen redimensionada')
    print('   Asegúrate de que los 4 cuadrados negros de las esquinas sean visibles')
    return cv2.resize(gray, (WORK_W, WORK_H)), False, None


def _get_scale_and_offsets():
    img_h, img_w = state['img_orig'].shape[:2]
    scale    = state['base_scale'] * state['zoom']
    view_w   = state['win_w']
    view_h   = state['win_h'] - HUD_H
    excess_x = max(0, img_w*scale - view_w)
    excess_y = max(0, img_h*scale - view_h)
    ox = state['pan_x'] * excess_x
    oy = state['pan_y'] * excess_y
    return scale, ox, oy


def display_to_img(dx, dy):
    scale, ox, oy = _get_scale_and_offsets()
    img_h, img_w  = state['img_orig'].shape[:2]
    nx = max(0.0, min(1.0, (dx + ox) / (img_w * scale)))
    ny = max(0.0, min(1.0, (dy + oy) / (img_h * scale)))
    return nx, ny


def render():
    img_h, img_w = state['img_orig'].shape[:2]
    scale, ox, oy = _get_scale_and_offsets()

    disp_w = max(1, int(img_w * scale))
    disp_h = max(1, int(img_h * scale))
    canvas = cv2.resize(state['img_orig'], (disp_w, disp_h), interpolation=cv2.INTER_LINEAR)

    # Factor de escala para radio y fuente de anotaciones
    ann_r    = max(6,  int(10 * scale))
    ann_font = max(0.35, 0.45 * scale)

    def draw_pt(nx, ny, color, label):
        px = int(nx * img_w * scale)
        py = int(ny * img_h * scale)
        cv2.circle(canvas, (px, py), ann_r, color, 2)
        cv2.circle(canvas, (px, py), 3,     color, -1)
        cv2.putText(canvas, label, (px+ann_r+2, py+4),
                    cv2.FONT_HERSHEY_SIMPLEX, ann_font, color, 1, cv2.LINE_AA)

    for i,(nx,ny) in enumerate(state['timing_clicks']):
        draw_pt(nx, ny, COLORS['TIMING'], f'TM{i+1}')
    for i,(nx,ny) in enumerate(state['top_bottom_clicks']):
        draw_pt(nx, ny, COLORS['TOP_BOTTOM'], 'TOP' if i==0 else 'BOT')
    for col_id, clicks in state['bubble_clicks'].items():
        clr  = COLORS[f'BUBBLES_C{col_id}']
        opts = COLUMNS_DEF[col_id]['options']
        for oi,(nx,ny) in enumerate(clicks):
            draw_pt(nx, ny, clr, opts[oi] if oi<len(opts) else '?')

    # Recortar viewport
    view_w = state['win_w']
    view_h = state['win_h'] - HUD_H
    x1 = int(ox); y1 = int(oy)
    x2 = min(x1 + view_w, disp_w)
    y2 = min(y1 + view_h, disp_h)
    view = canvas[y1:y2, x1:x2]

    # Padding si imagen es menor que ventana
    ph = view_h - view.shape[0]
    pw = view_w - view.shape[1]
    if ph > 0 or pw > 0:
        view = cv2.copyMakeBorder(view, 0, ph, 0, pw, cv2.BORDER_CONSTANT, value=(15,15,15))

    # HUD
    hud = np.zeros((HUD_H, view_w, 3), dtype=np.uint8)
    hud[:] = (28,28,28)
    mode    = state['mode']
    cur     = _mode_count(mode)
    exp     = MODE_EXPECTED[mode]
    bar_w   = int((view_w-20) * cur / exp)

    cv2.putText(hud, MODE_LABELS[mode], (10,20),
                cv2.FONT_HERSHEY_SIMPLEX, 0.50, (255,255,255), 1, cv2.LINE_AA)
    cv2.putText(hud, f'Clics: {cur}/{exp}', (10,42),
                cv2.FONT_HERSHEY_SIMPLEX, 0.44, COLORS[mode], 1, cv2.LINE_AA)
    cv2.rectangle(hud, (10,50), (view_w-10,58), (60,60,60), -1)
    if bar_w > 0:
        cv2.rectangle(hud, (10,50), (10+bar_w,58), COLORS[mode], -1)

    seg = (view_w-20) // len(MODES)
    for i,m in enumerate(MODES):
        color = (50,200,50) if _mode_complete(m) else (90,90,90)
        x0 = 10 + i*seg
        cv2.rectangle(hud, (x0,62),(x0+seg-3,72), color, -1)
        lbl = m.replace('BUBBLES_','C').replace('TOP_BOTTOM','TOP')[:5]
        cv2.putText(hud, lbl, (x0+2,71), cv2.FONT_HERSHEY_SIMPLEX, 0.28,(255,255,255),1)

    cv2.putText(hud, 'TAB=sig  Z=deshacer  R=reiniciar  S=guardar  +/-=zoom  0=encajar  flechas=mover  Q=salir',
                (10,84), cv2.FONT_HERSHEY_SIMPLEX, 0.34, (150,150,150), 1, cv2.LINE_AA)

    cv2.imshow(WIN_NAME, np.vstack([hud, view]))


def _mode_count(m):
    if m == 'TIMING':     return len(state['timing_clicks'])
    if m == 'TOP_BOTTOM': return len(state['top_bottom_clicks'])
    return len(state['bubble_clicks'][int(m[-1])])

def _mode_complete(m):
    return _mode_count(m) >= MODE_EXPECTED[m]


def mouse_cb(event, sx, sy, flags, param):
    if event != cv2.EVENT_LBUTTONDOWN:
        return
    dy = sy - HUD_H
    if dy < 0:
        return
    nx, ny = display_to_img(sx, dy)
    mode   = state['mode']
    if _mode_count(mode) >= MODE_EXPECTED[mode]:
        print(f'  Modo {mode} completo. Pulsa TAB para continuar.')
        return

    if mode == 'TIMING':
        state['timing_clicks'].append((nx,ny))
        state['history'].append(('TIMING',))
        print(f'  TM{len(state["timing_clicks"])}: x={nx:.4f}  y={ny:.4f}')
    elif mode == 'TOP_BOTTOM':
        state['top_bottom_clicks'].append((nx,ny))
        state['history'].append(('TOP_BOTTOM',))
        print(f'  {"TOP" if len(state["top_bottom_clicks"])==1 else "BOTTOM"}: x={nx:.4f}  y={ny:.4f}')
    else:
        col_id = int(mode[-1])
        clicks = state['bubble_clicks'][col_id]
        clicks.append((nx,ny))
        state['history'].append(('BUBBLE', col_id))
        opt = COLUMNS_DEF[col_id]['options'][len(clicks)-1]
        print(f'  Col{col_id+1} {opt}: x={nx:.4f}  y={ny:.4f}')

    if _mode_complete(mode):
        idx = MODES.index(mode)
        if idx < len(MODES)-1:
            state['mode'] = MODES[idx+1]
            print(f'\nPaso {mode} completo -> {state["mode"]}')
            print(f'  {MODE_LABELS[state["mode"]]}\n')
        else:
            print('\nTodos los pasos completos. Pulsa S para guardar.')


def compute_and_save(image_path):
    tf = {i: round(state['timing_clicks'][i][0], 4)
          for i in range(len(state['timing_clicks']))}

    if len(state['top_bottom_clicks']) >= 2:
        ys    = sorted(c[1] for c in state['top_bottom_clicks'])
        step  = (ys[1] - ys[0]) / 34
        top_f = round(max(0.0, ys[0] - step*0.6), 4)
        bot_f = round(min(1.0, ys[1] + step*0.6), 4)
    else:
        top_f, bot_f = 0.28, 0.97

    col_lines = []
    for col in COLUMNS_DEF:
        i      = col['id']
        clicks = state['bubble_clicks'][i]
        opts   = col['options']
        tx     = tf.get(i, '???')
        xs     = [round(c[0],4) for c in sorted(clicks, key=lambda c: c[0])] if clicks else None
        bx_str = str(xs) if xs else '[SIN CALIBRAR]'
        col_lines.append(
            f"        # {col['name']}\n"
            f"        {{'q_start': {col['q_start']:3d}, 'q_end': {col['q_end']:3d}, "
            f"'options': {opts},\n"
            f"         'bubble_fx': {bx_str},\n"
            f"         'timing_fx': {tx}}},"
        )

    block = (
        '\n# ' + '='*71 + '\n'
        f'# PARAMETROS SIPAGRE-140  —  {datetime.now().strftime("%Y-%m-%d %H:%M")}\n'
        f'# Fuente: {os.path.basename(image_path)}\n'
        '# Pega en SIPAGRE_140 de exam_profiles.py\n'
        '# ' + '='*71 + '\n\n'
        "    'columns': [\n"
        + '\n'.join(col_lines) + '\n'
        "    ],\n\n"
        f"    'answers_top_f':    {top_f},\n"
        f"    'answers_bottom_f': {bot_f},\n\n"
        '# ' + '='*71 + '\n'
    )

    print(block)
    out = 'sipagre_params_calibrados.txt'
    with open(out, 'w', encoding='utf-8') as f:
        f.write(block)
    print(f'Guardado: {os.path.abspath(out)}')
    cv2.imwrite('sipagre_calibracion_anotada.jpg', state['img_orig'])
    print('Imagen anotada: sipagre_calibracion_anotada.jpg')


def main():
    if len(sys.argv) < 2:
        print('Uso: python calibrar_sipagre.py <foto.jpg>')
        sys.exit(1)
    path = sys.argv[1]
    if not os.path.exists(path):
        print(f'Archivo no encontrado: {path}'); sys.exit(1)
    img_bgr = cv2.imread(path)
    if img_bgr is None:
        print(f'No se pudo abrir: {path}'); sys.exit(1)

    gray, p_ok, fid_pts = correct_perspective(cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY))
    canvas = cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR)

    # Dibujar los fiduciales detectados como referencia visual
    if fid_pts:
        labels = ['TL','TR','BR','BL']
        colors = [(0,220,220),(0,220,220),(0,220,220),(0,220,220)]
        for i, (px, py) in enumerate(fid_pts):
            # Transformar punto original al espacio warped
            src_pts = np.float32(fid_pts)
            dst_pts = np.float32([[0,0],[WORK_W-1,0],[WORK_W-1,WORK_H-1],[0,WORK_H-1]])
            M       = cv2.getPerspectiveTransform(src_pts, dst_pts)
            pt_w    = cv2.perspectiveTransform(
                        np.float32([[[px, py]]]), M)[0][0]
            wx, wy  = int(pt_w[0]), int(pt_w[1])
            cv2.drawMarker(canvas, (wx, wy), (0,200,200),
                           cv2.MARKER_CROSS, 30, 2)
            cv2.putText(canvas, labels[i], (wx+8, wy-8),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0,200,200), 2)

    state['img_orig'] = canvas

    print(f'\nImagen: {WORK_W}x{WORK_H}  |  Fiduciales: {"[OK] detectados" if p_ok else "[WARN] NO detectados - calibracion manual necesaria"}')
    print(f'\nPASO 1: {MODE_LABELS["TIMING"]}\n')

    cv2.namedWindow(WIN_NAME, cv2.WINDOW_NORMAL)

    # Usar tamaño de pantalla si está disponible, sino valores conservadores
    try:
        import tkinter as tk
        root = tk.Tk(); root.withdraw()
        sw = root.winfo_screenwidth()
        sh = root.winfo_screenheight()
        root.destroy()
        win_w = min(sw - 60, 1400)
        win_h = min(sh - 80, 900)
    except Exception:
        win_w, win_h = 1200, 850

    cv2.resizeWindow(WIN_NAME, win_w, win_h)
    state['win_w'] = win_w
    state['win_h'] = win_h

    img_h, img_w = state['img_orig'].shape[:2]
    state['base_scale'] = min(win_w / img_w, (win_h - HUD_H) / img_h)
    state['zoom'] = 1.0

    cv2.setMouseCallback(WIN_NAME, mouse_cb)

    while True:
        render()

        # Detectar redimensionado manual de la ventana
        try:
            rect = cv2.getWindowImageRect(WIN_NAME)
            if rect[2] > 200 and rect[3] > 200:
                if abs(rect[2] - state['win_w']) > 5 or abs(rect[3] - state['win_h']) > 5:
                    state['win_w'] = rect[2]
                    state['win_h'] = rect[3]
                    state['base_scale'] = min(
                        state['win_w'] / img_w,
                        (state['win_h'] - HUD_H) / img_h
                    )
        except Exception:
            pass

        key = cv2.waitKey(30) & 0xFF

        if key in (ord('q'), 27):
            break
        elif key == ord('s'):
            compute_and_save(path)
        elif key == ord('r'):
            state.update({'timing_clicks':[],'top_bottom_clicks':[],
                          'bubble_clicks':{0:[],1:[],2:[],3:[]},'history':[],
                          'mode':'TIMING','zoom':1.0,'pan_x':0.0,'pan_y':0.0})
            print('Reiniciado')
        elif key == ord('z'):
            if state['history']:
                last = state['history'].pop()
                if last[0]=='TIMING':        state['timing_clicks'].pop()
                elif last[0]=='TOP_BOTTOM':  state['top_bottom_clicks'].pop()
                else:                        state['bubble_clicks'][last[1]].pop()
                print('Deshecho')
        elif key == 9:
            idx = MODES.index(state['mode'])
            state['mode'] = MODES[(idx+1) % len(MODES)]
            print(f'Modo: {state["mode"]}  —  {MODE_LABELS[state["mode"]]}')
        elif key in (ord('+'), ord('=')):
            state['zoom'] = min(state['zoom']*1.25, 8.0)
        elif key == ord('-'):
            state['zoom'] = max(state['zoom']/1.25, 0.5)
        elif key == ord('0'):
            state['zoom']=1.0; state['pan_x']=0.0; state['pan_y']=0.0
        elif key == 82:  state['pan_y'] = max(0.0, state['pan_y']-0.05)
        elif key == 84:  state['pan_y'] = min(1.0, state['pan_y']+0.05)
        elif key == 81:  state['pan_x'] = max(0.0, state['pan_x']-0.05)
        elif key == 83:  state['pan_x'] = min(1.0, state['pan_x']+0.05)

    cv2.destroyAllWindows()
    print('Finalizado.')


if __name__ == '__main__':
    main()
