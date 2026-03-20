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
SIPAGRE_140 = {
    'id':    'SIPAGRE140',
    'name':  'SIPAGRE — 140 preguntas (4 col × 35)',
    'total_q': 140,
    'work_w':  1275,
    'work_h':  1650,

    # Posiciones calibradas con foto real (2025)
    # Col 1-3: A B C D (4 opciones)
    # Col 4:   A B C D E F G H (8 opciones)
    'columns': [
        # Col 1: P1-P35  — A B C D
        {'q_start':   1, 'q_end':  35, 'options': ['A','B','C','D'],
         'bubble_fx': [0.105, 0.125, 0.150, 0.170],
         'timing_fx': 0.060},
        # Col 2: P36-P70  — A B C D
        {'q_start':  36, 'q_end':  70, 'options': ['A','B','C','D'],
         'bubble_fx': [0.330, 0.350, 0.375, 0.395],
         'timing_fx': 0.285},
        # Col 3: P71-P105 — A B C D
        {'q_start':  71, 'q_end': 105, 'options': ['A','B','C','D'],
         'bubble_fx': [0.555, 0.575, 0.600, 0.620],
         'timing_fx': 0.510},
        # Col 4: P106-P140 — A B C D E F G H
        {'q_start': 106, 'q_end': 140, 'options': ['A','B','C','D','E','F','G','H'],
         'bubble_fx': [0.780, 0.800, 0.825, 0.845, 0.865, 0.890, 0.910, 0.935],
         'timing_fx': 0.735},
    ],

    'answers_top_f':    0.210,
    'answers_bottom_f': 0.87,
    'bubble_radius':     9,     # burbujas ligeramente más pequeñas (12pt vs 15pt)
    'fill_threshold':   0.12,
    'min_contrast':     0.10,
    'binarize_block':   25,
    'binarize_c':        8,
    'clahe_clip':        2.5,
    'clahe_grid':       (8, 8),
}


# ---------------------------------------------------------------------------
# Registro central — agregar aquí nuevos perfiles en el futuro
# ---------------------------------------------------------------------------
PROFILES = {
    JMR_125['id']:     JMR_125,
    SIPAGRE_140['id']: SIPAGRE_140,
}

# Lista ordenada para la UI
PROFILE_LIST = [
    {'id': JMR_125['id'],     'name': JMR_125['name']},
    {'id': SIPAGRE_140['id'], 'name': SIPAGRE_140['name']},
]

DEFAULT_PROFILE_ID = JMR_125['id']


def get_profile(profile_id: str) -> dict:
    """Retorna el perfil por ID. Fallback al perfil por defecto."""
    return PROFILES.get(profile_id, PROFILES[DEFAULT_PROFILE_ID])
