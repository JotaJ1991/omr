"""
calibrar.py
-----------
Script de calibración: te ayuda a encontrar los valores correctos de la grilla
para TU impresión específica del formulario.

Uso:
    python calibrar.py mi_hoja_de_respuestas.jpg

Genera una imagen con la grilla superpuesta para que veas si está alineada.
Si no está alineada, ajusta las constantes en omr_processor.py
"""

import sys
import cv2
import numpy as np

def calibrate(image_path: str):
    from omr_processor import (
        _correct_perspective, WORK_W, WORK_H,
        GRID_TOP, GRID_BOTTOM, GRID_LEFT, GRID_RIGHT,
        N_COLS_PAGE, QUESTIONS_PER_COL, N_OPTIONS
    )

    img = cv2.imread(image_path)
    if img is None:
        print(f"Error: no se pudo leer '{image_path}'")
        sys.exit(1)

    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    print("Corrigiendo perspectiva…")
    warped = _correct_perspective(gray)
    if warped is None:
        print("⚠ No se detectaron marcadores. Usando imagen completa redimensionada.")
        warped = cv2.resize(gray, (WORK_W, WORK_H))
    else:
        print("✓ Marcadores fiduciales detectados correctamente")

    # Dibujar grilla sobre imagen a color
    debug = cv2.cvtColor(warped, cv2.COLOR_GRAY2BGR)
    h, w = warped.shape

    grid_top    = int(GRID_TOP    * h)
    grid_bottom = int(GRID_BOTTOM * h)
    grid_left   = int(GRID_LEFT   * w)
    grid_right  = int(GRID_RIGHT  * w)
    grid_h = grid_bottom - grid_top
    grid_w = grid_right  - grid_left
    col_w  = grid_w / N_COLS_PAGE
    row_h  = grid_h / QUESTIONS_PER_COL

    # Rectángulo del área total de la grilla
    cv2.rectangle(debug, (grid_left, grid_top), (grid_right, grid_bottom), (255, 0, 0), 2)

    # Líneas de columnas
    for c in range(N_COLS_PAGE + 1):
        x = int(grid_left + c * col_w)
        cv2.line(debug, (x, grid_top), (x, grid_bottom), (0, 200, 0), 1)

    # Líneas de filas
    for r in range(QUESTIONS_PER_COL + 1):
        y = int(grid_top + r * row_h)
        cv2.line(debug, (grid_left, y), (grid_right, y), (0, 200, 0), 1)

    # Mostrar número de pregunta en cada celda
    for c in range(N_COLS_PAGE):
        for r in range(QUESTIONS_PER_COL):
            q = c * QUESTIONS_PER_COL + r + 1
            cx = int(grid_left + c * col_w + 5)
            cy = int(grid_top  + r * row_h + row_h * 0.6)
            cv2.putText(debug, str(q), (cx, cy),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.28, (0, 0, 255), 1)

    output_path = image_path.rsplit('.', 1)[0] + '_calibracion.jpg'
    cv2.imwrite(output_path, debug)
    print(f"\n✓ Imagen de calibración guardada en: {output_path}")
    print("\nRevisar:")
    print("  - ¿La caja azul rodea toda el área de respuestas?")
    print("  - ¿Las líneas verdes separan bien cada pregunta?")
    print("  - ¿Los números rojos están dentro de cada celda?")
    print("\nSi hay desajustes, edita las constantes en omr_processor.py:")
    print("  GRID_TOP, GRID_BOTTOM, GRID_LEFT, GRID_RIGHT")


if __name__ == '__main__':
    if len(sys.argv) < 2:
        print("Uso: python calibrar.py <ruta_imagen>")
        sys.exit(1)
    calibrate(sys.argv[1])
