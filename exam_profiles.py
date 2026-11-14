"""
exam_profiles.py — Perfiles de examen
======================================
Cada perfil define la geometría completa de una hoja OMR.
Para agregar un nuevo tipo: añadir una entrada al dict PROFILES
y registrarlo en PROFILE_LIST.

Estructura de un perfil
-----------------------
  name          str   — Nombre para mostrar en la UI
  total_q       int   — Número total de preguntas
  work_w        int   — Ancho canónico (px) tras corrección de perspectiva
  work_h        int   — Alto canónico (px)
  columns       list  — Una entrada por columna de respuestas:
      q_start   int   — Primera pregunta de la columna (1-based)
      q_end     int   — Última pregunta (incluida)
      options   list  — Letras válidas, ej. ['A','B','C','D']
      bubble_fx list  — Fracción X de cada opción respecto a work_w
      timing_fx float — Fracción X del timing mark de esta columna
  answers_top_f    float — Fracción Y donde empiezan las filas de respuestas
  answers_bottom_f float — Fracción Y donde terminan
  bubble_radius    int   — Radio del ROI de cada burbuja (px)
  fill_threshold   float — Mínimo fill para considerar una burbuja marcada
  min_contrast     float — Mínimo contraste fill_max − fill_2nd
  binarize_block   int   — blockSize para adaptiveThreshold (impar)
  binarize_c       int   — Constante C para adaptiveThreshold
  clahe_clip       float — clipLimit CLAHE (0 = sin CLAHE)
  clahe_grid       tuple — tileGridSize CLAHE
"""

# ---------------------------------------------------------------------------
# JMR-125  —  Hoja original 5 columnas × 25 filas, opciones A-D
# ---------------------------------------------------------------------------
JMR_125 = {
    'id':    'JMR125',
    'name':  'JMR — 125 preguntas (5 col × 25)',
    'total_q': 125,
    'work_w':  1275,
    'work_h':  1650,

    # Cada columna: q_start, q_end, options, bubble_fx, timing_fx
    'columns': [
        {'q_start':   1, 'q_end':  25, 'options': ['A','B','C','D'],
         'bubble_fx': [0.110, 0.145, 0.180, 0.215], 'timing_fx': 0.056},
        {'q_start':  26, 'q_end':  50, 'options': ['A','B','C','D'],
         'bubble_fx': [0.290, 0.325, 0.360, 0.395], 'timing_fx': 0.236},
        {'q_start':  51, 'q_end':  75, 'options': ['A','B','C','D'],
         'bubble_fx': [0.470, 0.505, 0.540, 0.575], 'timing_fx': 0.419},
        {'q_start':  76, 'q_end': 100, 'options': ['A','B','C','D'],
         'bubble_fx': [0.650, 0.685, 0.720, 0.755], 'timing_fx': 0.601},
        {'q_start': 101, 'q_end': 125, 'options': ['A','B','C','D'],
         'bubble_fx': [0.830, 0.865, 0.900, 0.935], 'timing_fx': 0.789},
    ],

    'answers_top_f':    0.35,
    'answers_bottom_f': 0.87,
    'bubble_radius':    10,
    'fill_threshold':   0.12,
    'min_contrast':     0.10,
    'binarize_block':   25,
    'binarize_c':        8,
    'clahe_clip':        2.5,
    'clahe_grid':       (8, 8),
}


# ---------------------------------------------------------------------------
# SIPAGRE-140  —  4 columnas × 35 filas; cols 1-3: A-D, col 4: A-H
#
# NOTA: Las posiciones X son ESTIMADAS basadas en la estructura del LaTeX.
# Deben calibrarse con fotos reales antes de usar en producción.
#
# LaTeX layout (letterpaper 215.9 × 279.4 mm, márgenes 1.1/0.9/1.2/1.1 cm):
#   Área útil ≈ 193.6 × 267.4 mm, 4 columnas con columnsep=6pt≈2.1mm
#   Ancho por columna ≈ (193.6 − 3×2.1) / 4 ≈ 46.8 mm
#   TMw=6pt, Nw=14pt, Bdia=12pt, Bsep=1.5pt
#   Col1 inicio X ≈ 12mm / 193.6mm ≈ 0.062 (fracción del área útil)
#   TM en col i: 0.062 + i * (46.8+2.1)/193.6
# ---------------------------------------------------------------------------
# ---------------------------------------------------------------------------
# 1S SIPAGRE-140  —  4 col × 35 filas
#   Col 1-2: A-D  |  Col 3-4: A-H
# ---------------------------------------------------------------------------
SIPAGRE_1S = {
    'id':    '1SSIPAGRE',
    'name':  '1S SIPAGRE — 140 preguntas',
    'total_q': 140,
    'work_w':  1275,
    'work_h':  1650,

    # Hoja física 4 columnas × 35 filas = 140 burbujas.
    # Col 1-3: A-D  |  Col 4: A-H
    # (Cuántas se evalúan por grado lo define la distribución en Sheets:
    #  ej. 6° usa 1-120, 7°/8° usan 1-125, simulacro Completo usa 1-140.)
    'columns': [
        # Col 1: P1-P35  — A B C D
        {'q_start':   1, 'q_end':  35, 'options': ['A','B','C','D'],
         'bubble_fx': [0.068, 0.093, 0.117, 0.142],
         'timing_fx': 0.020},
        # Col 2: P36-P70  — A B C D
        {'q_start':  36, 'q_end':  70, 'options': ['A','B','C','D'],
         'bubble_fx': [0.310, 0.335, 0.360, 0.385],
         'timing_fx': 0.262},
        # Col 3: P71-P105 — A B C D
        {'q_start':  71, 'q_end': 105, 'options': ['A','B','C','D'],
         'bubble_fx': [0.550, 0.574, 0.599, 0.624],
         'timing_fx': 0.503},
        # Col 4: P106-P140 — A B C D E F G H (35 filas, igual al LaTeX original)
        {'q_start': 106, 'q_end': 140, 'options': ['A','B','C','D','E','F','G','H'],
         'bubble_fx': [0.790, 0.815, 0.840, 0.865, 0.890, 0.914, 0.940, 0.963],
         'timing_fx': 0.746},
    ],

    'answers_top_f':    0.175,
    'answers_bottom_f': 0.977,
    'bubble_radius':     9,
    'snap_range':        3,
    'fill_threshold':   0.10,
    'min_contrast':     0.06,
    'binarize_block':   25,
    'binarize_c':        6,
    'clahe_clip':        2.5,
    'clahe_grid':       (8, 8),
    'mask_inner_ratio':  0.75,
}


# ---------------------------------------------------------------------------
# 2S SIPAGRE-140  —  4 col × 35 filas
#   Col 1-2: A-D  |  Col 3: A-H  |  Col 4: A-D
#   Misma geometría que 1S SIPAGRE, solo cambian las opciones de col 3 y 4
# ---------------------------------------------------------------------------
SIPAGRE_2S = {
    'id':    '2SSIPAGRE',
    'name':  '2S SIPAGRE — 134 preguntas',
    'total_q': 134,
    'work_w':  1275,
    'work_h':  1650,

    'columns': [
        # Col 1: P1-P35  — A B C D
        {'q_start':   1, 'q_end':  35, 'options': ['A','B','C','D'],
         'bubble_fx': [0.068, 0.093, 0.118, 0.143],
         'timing_fx': 0.02},
        # Col 2: P36-P70  — A B C D
        {'q_start':  36, 'q_end':  70, 'options': ['A','B','C','D'],
         'bubble_fx': [0.309, 0.333, 0.359, 0.383],
         'timing_fx': 0.262},
        # Col 3: P71-P105 — A B C D E F G H
        {'q_start':  71, 'q_end': 105, 'options': ['A','B','C','D','E','F','G','H'],
         'bubble_fx': [0.551, 0.575, 0.599, 0.624, 0.648, 0.673, 0.697, 0.722],
         'timing_fx': 0.503},
        # Col 4: P106-P134 — A B C D
        {'q_start': 106, 'q_end': 134, 'options': ['A','B','C','D'],
         'bubble_fx': [0.871, 0.897, 0.920, 0.944],
         'timing_fx': 0.823},
    ],

    'answers_top_f':    0.175,
    'answers_bottom_f': 0.977,
    'bubble_radius':     9,
    'snap_range':        3,
    'fill_threshold':   0.10,
    'min_contrast':     0.06,
    'binarize_block':   25,
    'binarize_c':        6,
    'clahe_clip':        2.5,
    'clahe_grid':       (8, 8),
    'mask_inner_ratio':  0.75,   # mide solo el 60% interior, ignora el borde impreso
}


# ---------------------------------------------------------------------------
# 2S SIPAGRE A-H  —  Hoja de la 2.ª sesión de 10° (IETECI OCTUBRE 2026)
#   impresa por error con el formato de 140 burbujas:
#   Col 1-2: A-D  |  Col 3-4: A-H  (P1-P140)
# ---------------------------------------------------------------------------
SIPAGRE_2S_AH = {
    'id':    '2SSIPAGRE_AH',
    'name':  '2S SIPAGRE A-H — 140 preguntas (col 3-4 A-H)',
    'total_q': 140,
    'work_w':  1275,
    'work_h':  1650,

    # Posiciones medidas sobre una foto real (Respuestas 10° S2, oct-2026)
    # y ajustadas a mano por el docente con el calibrador visual:
    # las columnas 3-4 son más anchas (8 opciones), así que las columnas
    # 1-3 quedan más a la izquierda que en el 1S/2S SIPAGRE.
    'columns': [
        {'q_start':   1, 'q_end':  35, 'options': ['A','B','C','D'],
         'bubble_fx': [0.0689, 0.0939, 0.1189, 0.1439],
         'timing_fx': 0.017},
        {'q_start':  36, 'q_end':  70, 'options': ['A','B','C','D'],
         'bubble_fx': [0.2818, 0.3068, 0.3308, 0.3558],
         'timing_fx': 0.233},
        {'q_start':  71, 'q_end': 105, 'options': ['A','B','C','D','E','F','G','H'],
         'bubble_fx': [0.497, 0.522, 0.547, 0.571, 0.596, 0.621, 0.646, 0.671],
         'timing_fx': 0.452},
        {'q_start': 106, 'q_end': 140, 'options': ['A','B','C','D','E','F','G','H'],
         'bubble_fx': [0.7916, 0.8166, 0.8426, 0.8676, 0.8926, 0.9166, 0.9416, 0.9666],
         'timing_fx': 0.746},
    ],

    'answers_top_f':    0.175,
    'answers_bottom_f': 0.9776,
    'bubble_radius':    11,
    'snap_range':        3,
    'fill_threshold':   0.10,
    'min_contrast':     0.06,
    'binarize_block':   25,
    'binarize_c':        6,
    'clahe_clip':        2.5,
    'clahe_grid':       (8, 8),
    'mask_inner_ratio':  0.75,
    # Marca real: núcleo >= 0.08 más oscuro que la mediana de su fila
    # (relativo a la foto: marcas fuertes ~+0.45, lápiz suave ~+0.10;
    # filas vacías <= +0.06 aun con foto oscura o con sombra)
    'min_mark_gap':      0.08,
}


# ---------------------------------------------------------------------------
# 2S SIPAGRE 6°/9° — formato impreso por error en la 2.ª sesión de 6° y 9°
#   (IETECI OCTUBRE 2026): columnas 1 y 4 A-D, columnas 2 y 3 A-H.
#   Geometría medida sobre el PDF original (HOJA DE RESPUESTA 9° 2 SESION)
#   pasado por el mismo enderezado que las fotos.
# ---------------------------------------------------------------------------
SIPAGRE_2S_AH9 = {
    'id':    '2SSIPAGRE_AH9',
    'name':  '2S SIPAGRE 6°/9° — 140 preguntas (col 2-3 A-H)',
    'total_q': 140,
    'work_w':  1275,
    'work_h':  1650,

    'columns': [
        {'q_start':   1, 'q_end':  35, 'options': ['A','B','C','D'],
         'bubble_fx': [0.0694, 0.0937, 0.1188, 0.1431],
         'timing_fx': 0.0165},
        {'q_start':  36, 'q_end':  70, 'options': ['A','B','C','D','E','F','G','H'],
         'bubble_fx': [0.2851, 0.3094, 0.3345, 0.3588, 0.3839, 0.4082, 0.4325, 0.4576],
         'timing_fx': 0.2322},
        {'q_start':  71, 'q_end': 105, 'options': ['A','B','C','D','E','F','G','H'],
         'bubble_fx': [0.5753, 0.6004, 0.6247, 0.6498, 0.6741, 0.6992, 0.7235, 0.7486],
         'timing_fx': 0.5184},
        {'q_start': 106, 'q_end': 140, 'options': ['A','B','C','D'],
         'bubble_fx': [0.8663, 0.8914, 0.9157, 0.9408],
         'timing_fx': 0.8141},
    ],

    'answers_top_f':    0.1756,
    'answers_bottom_f': 0.9777,
    'bubble_radius':    11,
    'snap_range':        3,
    'fill_threshold':   0.10,
    'min_contrast':     0.06,
    'binarize_block':   25,
    'binarize_c':        6,
    'clahe_clip':        2.5,
    'clahe_grid':       (8, 8),
    'mask_inner_ratio':  0.75,
    'min_mark_gap':      0.08,   # igual que el A-H de 10°
}


# ---------------------------------------------------------------------------
# PRIMARIA 60 — hoja para primaria: 60 preguntas A-D en 3 columnas de 20,
#   burbujas de 7 mm con la letra dentro (templates_latex/primaria_60.tex).
#   Geometría medida sobre el PDF pasado por el enderezado de las fotos.
# ---------------------------------------------------------------------------
PRIMARIA_60 = {
    'id':    'PRIMARIA60',
    'name':  'Primaria — 60 preguntas (burbuja grande)',
    'total_q': 60,
    'work_w':  1275,
    'work_h':  1650,

    'columns': [
        {'q_start':  1, 'q_end': 20, 'options': ['A','B','C','D'],
         'bubble_fx': [0.1204, 0.1745, 0.2294, 0.2843],
         'timing_fx': 0.0173},
        {'q_start': 21, 'q_end': 40, 'options': ['A','B','C','D'],
         'bubble_fx': [0.4608, 0.5157, 0.5698, 0.6247],
         'timing_fx': 0.3576},
        {'q_start': 41, 'q_end': 60, 'options': ['A','B','C','D'],
         'bubble_fx': [0.8020, 0.8561, 0.9110, 0.9659],
         'timing_fx': 0.6988},
    ],

    'answers_top_f':    0.1951,
    'answers_bottom_f': 0.9619,
    'bubble_radius':    16,      # la burbuja real mide ~22 px; se lee el interior
    'snap_range':        3,
    'fill_threshold':   0.10,
    'min_contrast':     0.06,
    'binarize_block':   25,
    'binarize_c':        6,
    'clahe_clip':        2.5,
    'clahe_grid':       (8, 8),
    'mask_inner_ratio':  0.75,
    # Niños: marcas pequeñas, descentradas o garabateadas → buscar el núcleo
    # más oscuro en toda la burbuja (±8 px). Umbral más alto que en
    # bachillerato para que un borrón no se lea como respuesta.
    'core_search':       8,
    'min_mark_gap':      0.16,
}


# ---------------------------------------------------------------------------
# M SIPAGRE  —  Una sola jornada, 125 preguntas
#   Geometría: 4 columnas. La columna 4 lee solo hasta P125.
# ---------------------------------------------------------------------------
M_SIPAGRE = {
    'id':    'MSIPAGRE',
    'name':  'M SIPAGRE — 125 preguntas',
    'total_q': 125,
    'work_w':  1275,
    'work_h':  1650,

    'columns': [
        {'q_start':   1, 'q_end':  35, 'options': ['A','B','C','D'],
         'bubble_fx': [0.069, 0.094, 0.119, 0.144],
         'timing_fx': 0.02},
        {'q_start':  36, 'q_end':  70, 'options': ['A','B','C','D'],
         'bubble_fx': [0.309, 0.333, 0.359, 0.383],
         'timing_fx': 0.262},
        {'q_start':  71, 'q_end': 105, 'options': ['A','B','C','D','E','F','G','H'],
         'bubble_fx': [0.551, 0.575, 0.599, 0.624, 0.648, 0.673, 0.697, 0.722],
         'timing_fx': 0.503},
        # Col 4: lee 29 espacios físicos para que la geometría se calibre,
        # pero la imagen de debug y el scoring solo usan hasta P125 (total_q).
        {'q_start': 106, 'q_end': 134, 'options': ['A','B','C','D'],
         'bubble_fx': [0.870, 0.896, 0.919, 0.943],
         'timing_fx': 0.823},
    ],

    'answers_top_f':    0.175,
    'answers_bottom_f': 0.977,
    'bubble_radius':     9,
    'snap_range':        3,
    'fill_threshold':   0.10,
    'min_contrast':     0.06,
    'binarize_block':   25,
    'binarize_c':        6,
    'clahe_clip':        2.5,
    'clahe_grid':       (8, 8),
    'mask_inner_ratio':  0.75,
}


# ---------------------------------------------------------------------------
# IETECI 6°  —  Hoja 1S SIPAGRE (120 burbujas), TODAS las opciones A-D
#   Formato físico idéntico al 1S SIPAGRE excepto en la columna 4:
#   en 1S original esa columna es A-H (P106-P120); aquí es A-D para que
#   el OMR no espere las opciones E-H.
#   Distribución de asignaturas (definida en hoja "Distribucion"):
#     Mat 1-20 | Lect 21-40 | Soc 41-60 | Nat 61-80 | Ing 81-120
# ---------------------------------------------------------------------------
IETECI_6 = {
    'id':    'IETECI6',
    'name':  'IETECI 6° — 120 preguntas (todas A-D)',
    'total_q': 120,
    'work_w':  1275,
    'work_h':  1650,

    # Geometría IDÉNTICA al 1S SIPAGRE original (las 4 columnas en las
    # mismas posiciones x) — solo cambia que la col 4 lee A-D en lugar de A-H.
    'columns': [
        {'q_start':   1, 'q_end':  35, 'options': ['A','B','C','D'],
         'bubble_fx': [0.068, 0.093, 0.117, 0.142],
         'timing_fx': 0.020},
        {'q_start':  36, 'q_end':  70, 'options': ['A','B','C','D'],
         'bubble_fx': [0.310, 0.335, 0.360, 0.385],
         'timing_fx': 0.262},
        {'q_start':  71, 'q_end': 105, 'options': ['A','B','C','D'],
         'bubble_fx': [0.550, 0.574, 0.599, 0.624],
         'timing_fx': 0.503},
        # Col 4: P106-P120, A-D solamente (las E-H del 1S original se ignoran)
        {'q_start': 106, 'q_end': 120, 'options': ['A','B','C','D'],
         'bubble_fx': [0.790, 0.815, 0.840, 0.865],
         'timing_fx': 0.789},
    ],

    'answers_top_f':    0.175,
    'answers_bottom_f': 0.977,
    'bubble_radius':     9,
    'snap_range':        3,
    'fill_threshold':   0.10,
    'min_contrast':     0.06,
    'binarize_block':   25,
    'binarize_c':        6,
    'clahe_clip':        2.5,
    'clahe_grid':       (8, 8),
    'mask_inner_ratio':  0.75,
}

# IETECI 7°/8° ya no requiere perfil aparte: la hoja física 1S SIPAGRE
# tiene 140 burbujas (4 col × 35 filas). Cualquier grado puede usar el
# perfil 1S SIPAGRE y la distribución (en hoja "Distribucion") define
# cuántas preguntas se evalúan realmente:
#   - 6°:     1-120 (Ing 81-120, A-D)
#   - 7°/8°:  1-125 (Ing 101-125, A-D hasta 105 + A-H 106-125)
#   - Comp.:  1-140 (Completo SIPAGRE original)


# ---------------------------------------------------------------------------
# Registro central
# (JMR-125 desactivado a pedido del usuario; sigue como referencia histórica
# en este archivo pero no se expone en PROFILES ni en PROFILE_LIST.)
# ---------------------------------------------------------------------------
PROFILES = {
    SIPAGRE_1S['id']: SIPAGRE_1S,
    SIPAGRE_2S['id']: SIPAGRE_2S,
    M_SIPAGRE['id']:  M_SIPAGRE,
    SIPAGRE_2S_AH['id']: SIPAGRE_2S_AH,
    SIPAGRE_2S_AH9['id']: SIPAGRE_2S_AH9,
    PRIMARIA_60['id']: PRIMARIA_60,     # aún sin listar en la UI
}

# Lista ordenada para la UI
PROFILE_LIST = [
    {'id': SIPAGRE_1S['id'], 'name': SIPAGRE_1S['name']},
    {'id': SIPAGRE_2S['id'], 'name': SIPAGRE_2S['name']},
    {'id': M_SIPAGRE['id'],  'name': M_SIPAGRE['name']},
    {'id': SIPAGRE_2S_AH['id'], 'name': SIPAGRE_2S_AH['name']},
    {'id': SIPAGRE_2S_AH9['id'], 'name': SIPAGRE_2S_AH9['name']},
]

# Formatos físicos distintos al habitual en un (simulacro, grado, sesión):
# el servidor cambia el perfil solo, así los escáneres no eligen nada.
FORMAT_OVERRIDES = {
    # 2.ª sesión de 10° impresa por error en el formato A-H de 140 burbujas
    ('IETECI OCTUBRE 2026', '10', '2S'): SIPAGRE_2S_AH['id'],
    # 2.ª sesión de 6° y 9°: columnas 2 y 3 A-H (1 y 4 A-D)
    ('IETECI OCTUBRE 2026', '6', '2S'): SIPAGRE_2S_AH9['id'],
    ('IETECI OCTUBRE 2026', '9', '2S'): SIPAGRE_2S_AH9['id'],
}


def profile_override(simulacro: str, grado: str, sesion: str) -> str:
    """ID de perfil que reemplaza al habitual para esa hoja, o ''."""
    return FORMAT_OVERRIDES.get(((simulacro or '').strip(), str(grado or '').strip(),
                                 (sesion or '').strip().upper()), '')

DEFAULT_PROFILE_ID = SIPAGRE_1S['id']


def get_profile(profile_id: str) -> dict:
    """Retorna el perfil por ID. Fallback al perfil por defecto."""
    return PROFILES.get(profile_id, PROFILES[DEFAULT_PROFILE_ID])
