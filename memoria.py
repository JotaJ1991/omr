"""
memoria.py — control de memoria para la instancia de 512 MB de Render.

Se importa PRIMERO en app.py (antes de crear hilos) porque:

1. glibc crea una "arena" de memoria por hilo y conserva lo que se libera en
   ella. Con 4 hilos de gunicorn (más los de OpenCV), cada escaneo deja
   decenas de MB retenidos en arenas distintas y la memoria del proceso sube
   escaneo tras escaneo hasta que Render lo mata ('Ran out of memory').
   - M_ARENA_MAX = 2: pocas arenas compartidas.
   - M_MMAP_THRESHOLD = 1 MB fijo: las imágenes (varios MB) se piden con
     mmap y vuelven al sistema al liberarse (por defecto glibc sube ese
     umbral solo y empieza a retenerlas en el heap).
2. liberar() devuelve al sistema la memoria libre que quede tras un escaneo.
3. reporte() deja en el log cuánta memoria usa el contenedor (lo que Render
   compara con el límite) para poder diagnosticar.

En Windows/macOS no hace nada (solo aplica a glibc/Linux).
"""
import os

_libc = None
try:
    import ctypes
    _libc = ctypes.CDLL('libc.so.6')
    _M_TRIM_THRESHOLD, _M_MMAP_THRESHOLD, _M_ARENA_MAX = -1, -3, -8
    _libc.mallopt(_M_ARENA_MAX, int(os.environ.get('OMR_MALLOC_ARENAS', '2')))
    _libc.mallopt(_M_MMAP_THRESHOLD, 1 << 20)
    _libc.mallopt(_M_TRIM_THRESHOLD, 4 << 20)
except Exception:
    _libc = None


def liberar():
    """Devuelve al sistema operativo la memoria libre retenida por malloc."""
    try:
        import gc
        gc.collect()
        if _libc is not None:
            _libc.malloc_trim(0)
    except Exception:
        pass


def _leer_mb(path):
    try:
        with open(path) as f:
            v = f.read().strip()
        return int(v) / 2**20 if v.isdigit() else None
    except Exception:
        return None


def _status_mb(pid, campo):
    try:
        with open(f'/proc/{pid}/status') as f:
            for linea in f:
                if linea.startswith(campo + ':'):
                    return int(linea.split()[1]) / 1024
    except Exception:
        pass
    return None


def reporte() -> str:
    """'contenedor=… MB · worker=… MB (pico …) · maestro=… MB' (Linux)."""
    partes = []
    cg = (_leer_mb('/sys/fs/cgroup/memory.current')
          or _leer_mb('/sys/fs/cgroup/memory/memory.usage_in_bytes'))
    if cg is not None:
        partes.append(f'contenedor={cg:.0f} MB')
    rss = _status_mb('self', 'VmRSS')
    if rss is not None:
        hwm = _status_mb('self', 'VmHWM')
        partes.append(f'worker={rss:.0f} MB (pico {hwm:.0f})')
    ppid = os.getppid()
    m = _status_mb(ppid, 'VmRSS') if ppid > 1 else None
    if m is not None:
        partes.append(f'maestro={m:.0f} MB')
    return ' · '.join(partes) or 'sin datos (no es Linux)'
