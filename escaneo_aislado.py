"""
escaneo_aislado.py — el análisis de cada foto corre en un PROCESO APARTE.

Con algunas fotos el análisis dispara la memoria y Render mataba la app
entera ('Ran out of memory'): todos los ayudantes perdían la conexión y el
servidor tardaba ~1 min en volver. Ahora:

  · Un proceso hijo (persistente, con OpenCV cargado) analiza las fotos de a
    una. Un vigilante dentro del hijo mide su memoria cada 20 ms y, si pasa
    de OMR_SCAN_MAX_MB (300 MB por defecto), lo termina en el acto.
  · Si el hijo muere o se cuelga (OMR_SCAN_TIMEOUT s), solo esa foto falla:
    el teléfono recibe "toma la foto de nuevo", el servidor sigue en pie y el
    siguiente escaneo levanta un hijo nuevo.
  · La foto que causó el problema se guarda en uploads/fallidas/ (máx. 20)
    junto con la etapa en que ocurrió, para poder diagnosticarla.
"""
import os
import time
import shutil
import threading
import multiprocessing as mp
import concurrent.futures as cf

_LIMITE_MB = int(os.environ.get('OMR_SCAN_MAX_MB', '300'))
_TIMEOUT_S = int(os.environ.get('OMR_SCAN_TIMEOUT', '60'))
CARPETA_FALLIDAS = os.path.join('uploads', 'fallidas')
_MAX_FALLIDAS = 20

_ctx = mp.get_context('spawn')
_pool = None
_pool_lock = threading.Lock()   # crear / descartar el proceso hijo
_turno = threading.Lock()       # una foto a la vez dentro del hijo


# ── Lado HIJO ──────────────────────────────────────────────────────────────

def _iniciar_hijo(limite_mb):
    import memoria                       # noqa: F401  (ajusta malloc)
    import omr_processor as op

    def vigilar():
        try:
            pagina = os.sysconf('SC_PAGE_SIZE')
        except (ValueError, AttributeError, OSError):
            return                       # no es Linux: sin vigilante
        while True:
            try:
                with open('/proc/self/statm') as f:
                    rss = int(f.read().split()[1]) * pagina / 2**20
            except Exception:
                return
            if rss > limite_mb:
                try:
                    with open(os.path.join(CARPETA_FALLIDAS, '_ultima_etapa.txt'), 'w') as f:
                        f.write(op.ETAPA)
                except Exception:
                    pass
                print(f'[escaneo] ABORTADO: el análisis llegó a {rss:.0f} MB '
                      f'en la etapa "{op.ETAPA}"', flush=True)
                os._exit(86)
            time.sleep(0.02)

    threading.Thread(target=vigilar, daemon=True).start()


def _tarea(image_path, kwargs):
    import memoria
    import omr_processor as op
    try:
        return op.process_exam_image(image_path, **kwargs)
    finally:
        op.ETAPA = 'libre'
        memoria.liberar()
        print('[mem-hijo] ' + memoria.reporte(), flush=True)


# ── Lado SERVIDOR ──────────────────────────────────────────────────────────

def _obtener_pool():
    global _pool
    with _pool_lock:
        if _pool is None:
            os.makedirs(CARPETA_FALLIDAS, exist_ok=True)
            _pool = cf.ProcessPoolExecutor(
                max_workers=1, mp_context=_ctx,
                initializer=_iniciar_hijo, initargs=(_LIMITE_MB,))
        return _pool


def _descartar_pool(pool):
    global _pool
    with _pool_lock:
        if _pool is pool:
            _pool = None
    for p in list(getattr(pool, '_processes', {}).values()):
        try:
            p.kill()
        except Exception:
            pass
    try:
        pool.shutdown(wait=False, cancel_futures=True)
    except Exception:
        pass


def _guardar_fallida(image_path, etiqueta, motivo):
    try:
        os.makedirs(CARPETA_FALLIDAS, exist_ok=True)
        etapa = ''
        p_etapa = os.path.join(CARPETA_FALLIDAS, '_ultima_etapa.txt')
        if os.path.exists(p_etapa):
            with open(p_etapa) as f:
                etapa = f.read().strip()
            os.remove(p_etapa)
        limpio = ''.join(c if c.isalnum() or c in '-_' else '_' for c in
                         f'{etiqueta}_{motivo}_{etapa}')[:80]
        destino = os.path.join(CARPETA_FALLIDAS,
                               time.strftime('%Y%m%d-%H%M%S') + f'_{limpio}.jpg')
        shutil.copyfile(image_path, destino)
        fotos = sorted(f for f in os.listdir(CARPETA_FALLIDAS) if f.endswith('.jpg'))
        for viejo in fotos[:-_MAX_FALLIDAS]:
            os.remove(os.path.join(CARPETA_FALLIDAS, viejo))
        print(f'[escaneo] foto guardada para diagnóstico: {os.path.basename(destino)}',
              flush=True)
    except Exception as e:
        print(f'[escaneo] no se pudo guardar la foto fallida: {e}', flush=True)


def procesar(image_path, etiqueta='', **kwargs):
    """Igual que omr_processor.process_exam_image, pero aislado."""
    with _turno:
        motivo = 'hijo'
        for _ in range(2):
            pool = _obtener_pool()
            try:
                fut = pool.submit(_tarea, image_path, kwargs)
            except (cf.process.BrokenProcessPool, RuntimeError):
                # El hijo ya estaba muerto ANTES de esta foto (no es culpa de
                # ella): se levanta uno nuevo y se reintenta.
                _descartar_pool(pool)
                continue
            try:
                return fut.result(timeout=_TIMEOUT_S)
            except cf.TimeoutError:
                motivo = 'tiempo'
            except cf.process.BrokenProcessPool:
                motivo = 'memoria'
            _descartar_pool(pool)
            break
    print(f'[escaneo] el análisis falló ({motivo}); el servidor sigue en pie',
          flush=True)
    _guardar_fallida(image_path, etiqueta, motivo)
    return {'success': False, 'retake': True,
            'error': ('No se pudo analizar esta foto. Tómala de nuevo con la '
                      'hoja completa, plana y bien iluminada.')}
