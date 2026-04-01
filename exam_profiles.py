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
    'name':  '1S SIPAGRE — 120 preguntas',
    'total_q': 120,
    'work_w':  1275,
    'work_h':  1650,

    # Posiciones calibradas (paso=0.025, gap timing→A=0.046)
    # Col 1-2: A B C D  |  Col 3-4: A B C D E F G H
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
         'bubble_fx': [0.551, 0.575, 0.600, 0.625],
         'timing_fx': 0.504},
        # Col 4: P106-P120 — A B C D E F G H
        {'q_start': 106, 'q_end': 120, 'options': ['A','B','C','D','E','F','G','H'],
         'bubble_fx': [0.790, 0.815, 0.840, 0.865, 0.890, 0.914, 0.940, 0.963],
         'timing_fx': 0.746},
    ],

    'answers_top_f':    0.175,
    'answers_bottom_f': 0.977,
    'bubble_radius':     9,
    'snap_range':        3,
    'fill_threshold':   0.02,
    'min_contrast':     0.01,
    'binarize_block':   25,
    'binarize_c':        4,
    'clahe_clip':        3.5,
    'clahe_grid':       (8, 8),
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
    'fill_threshold':   0.02,
    'min_contrast':     0.01,
    'binarize_block':   25,
    'binarize_c':        4,
    'clahe_clip':        3.5,
    'clahe_grid':       (8, 8),
}


# ---------------------------------------------------------------------------
# Registro central
# ---------------------------------------------------------------------------
PROFILES = {
    JMR_125['id']:    JMR_125,
    SIPAGRE_1S['id']: SIPAGRE_1S,
    SIPAGRE_2S['id']: SIPAGRE_2S,
}

# Lista ordenada para la UI
PROFILE_LIST = [
    {'id': SIPAGRE_1S['id'], 'name': SIPAGRE_1S['name']},
    {'id': SIPAGRE_2S['id'], 'name': SIPAGRE_2S['name']},
    {'id': JMR_125['id'],    'name': JMR_125['name']},
]

DEFAULT_PROFILE_ID = SIPAGRE_1S['id']


def get_profile(profile_id: str) -> dict:
    """Retorna el perfil por ID. Fallback al perfil por defecto."""
    return PROFILES.get(profile_id, PROFILES[DEFAULT_PROFILE_ID])
