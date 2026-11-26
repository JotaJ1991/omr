"""Lectura del texto del QR de las hojas personalizadas (sin OpenCV)."""


def parse_qr_payload(data: str) -> dict:
    """Parsea 'OMR|<sim>|<ses>|<id>|<curso>' → dict, o None si no es válido."""
    if not data:
        return None
    parts = data.split('|')
    if len(parts) < 5 or parts[0] != 'OMR':
        return None
    sim   = (parts[1] or '').strip()
    ses   = (parts[2] or '').strip().upper()
    sid   = (parts[3] or '').strip()
    curso = (parts[4] or '').strip()
    if not (sim and ses and sid):
        return None
    return {
        'raw':       data,
        'simulacro': sim,
        'sesion':    ses,
        'student_id': sid,
        'curso':     curso,
    }
