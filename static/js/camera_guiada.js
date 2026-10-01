/**
 * camera_guiada.js — Cámara guiada con detección de bordes (OpenCV.js)
 * ====================================================================
 * Módulo compartido entre la app de administración (index.html) y la
 * página de escaneo para ayudantes (escaner.html).
 *
 * Contrato con la página que lo incluye:
 *   - Debe existir el modal con ids: cam-modal, cam-video, cam-canvas,
 *     cam-error, cam-loading, cam-hint.
 *   - Debe existir <input type=file id="file-input">: al capturar,
 *     sendBlob() coloca ahí la foto y dispara el evento 'change'
 *     (la página procesa la imagen en su handler de ese evento).
 *   - Expone además _pendingQrHint: QR decodificado en el cliente desde
 *     el frame completo ANTES del recorte (respaldo para /process).
 */

// ── Estado global de la cámara ──────────────────────────────────────────────
let camStream      = null;
let camFacing      = 'environment';
let cvLoaded       = false;
let cvLoading      = false;
let detectionLoop  = null;
let lastDetected   = null;
let videoReady     = false;
// OpenCV.js se sirve LOCALMENTE (static/vendor/) porque docs.opencv.org
// eliminó las versiones antiguas (la 4.10.0 empezó a dar 404 y rompió la
// cámara guiada). El respaldo remoto apunta a "4.x" (siempre la última).
const OPENCV_URL          = '/static/vendor/opencv.js';
const OPENCV_URL_FALLBACK = 'https://docs.opencv.org/4.x/opencv.js';



// Pista de QR decodificada en el cliente desde el frame COMPLETO (antes del
// recorte de la cámara guiada). El recorte por perspectiva puede dejar el QR
// del encabezado fuera de la imagen enviada, así que lo leemos aquí y lo
// pasamos al servidor como respaldo.
let _pendingQrHint = '';

// Decodifica un QR desde un canvas usando la API nativa BarcodeDetector
// (Chrome/Android). Devuelve el texto del QR o '' si no se detecta / no
// está soportada la API.
async function _decodeQrFromCanvas(canvas) {
  try {
    if (!('BarcodeDetector' in window)) return '';
    const det = new BarcodeDetector({ formats: ['qr_code'] });
    const codes = await det.detect(canvas);
    if (codes && codes.length) {
      for (const c of codes) {
        const v = (c.rawValue || '').trim();
        if (v.startsWith('OMR|')) return v;
      }
      return (codes[0].rawValue || '').trim();
    }
  } catch (e) { /* no soportado o sin QR */ }
  return '';
}



async function loadOpenCV() {
  if (cvLoaded) return true;
  if (cvLoading) {
    // Esperar a que termine la carga en curso
    return new Promise(res => {
      const wait = setInterval(() => {
        if (cvLoaded) { clearInterval(wait); res(true); }
      }, 200);
    });
  }
  cvLoading = true;
  return new Promise((resolve) => {
    if (window.cv && window.cv.Mat) { cvLoaded = true; cvLoading = false; resolve(true); return; }

    const tryLoad = (url, onFail) => {
      const script = document.createElement('script');
      script.src = url;
      script.async = true;
      script.onload = () => {
        // OpenCV.js inicializa de forma asíncrona
        const checkReady = () => {
          if (window.cv && window.cv.Mat) {
            cvLoaded = true; cvLoading = false; resolve(true);
          } else if (window.cv) {
            window.cv.onRuntimeInitialized = () => { cvLoaded = true; cvLoading = false; resolve(true); };
          } else {
            setTimeout(checkReady, 100);
          }
        };
        checkReady();
      };
      script.onerror = onFail;
      document.head.appendChild(script);
    };

    // 1º el archivo local (mismo dominio, cacheable, sin filtros externos);
    // 2º el sitio oficial como respaldo.
    tryLoad(OPENCV_URL, () => {
      tryLoad(OPENCV_URL_FALLBACK, () => { cvLoading = false; resolve(false); });
    });
  });
}

async function openCameraModal() {
  if (!navigator.mediaDevices || !navigator.mediaDevices.getUserMedia) {
    alert('Tu navegador no soporta acceso a cámara. Usa el botón Cámara normal.');
    return;
  }
  const modal = document.getElementById('cam-modal');
  modal.classList.add('show');
  document.getElementById('cam-error').style.display = 'none';
  document.getElementById('cam-loading').classList.remove('hidden');
  document.getElementById('cam-hint').textContent = 'Inicializando detector…';
  document.getElementById('cam-hint').classList.remove('detected');

  // Iniciar cámara y cargar OpenCV en paralelo
  const [_, ok] = await Promise.all([startCameraStream(), loadOpenCV()]);
  document.getElementById('cam-loading').classList.add('hidden');

  if (!ok) {
    document.getElementById('cam-hint').textContent = 'Detector no disponible. Puedes capturar manualmente.';
    return;
  }
  document.getElementById('cam-hint').textContent = 'Apunta al documento';
  startDetectionLoop();
}

async function startCameraStream() {
  stopCameraStream();
  videoReady = false;
  try {
    // Pedimos la máxima resolución posible (4K). El detector en vivo igual
    // procesa a 480px (ver detectFrame), así que subir la resolución NO
    // ralentiza la guía, pero SÍ mejora la nitidez de la captura y la
    // lectura del QR. El dispositivo hace fallback al máximo que soporte.
    camStream = await navigator.mediaDevices.getUserMedia({
      video: { facingMode: camFacing, width: {ideal: 3840}, height: {ideal: 2160} },
      audio: false
    });
    const v = document.getElementById('cam-video');
    v.srcObject = camStream;
    v.onloadedmetadata = () => {
      videoReady = true;
      const cv2 = document.getElementById('cam-canvas');
      cv2.width  = v.videoWidth;
      cv2.height = v.videoHeight;
      const mp = (v.videoWidth * v.videoHeight / 1e6).toFixed(1);
      console.log(`[OMR] Cámara guiada: ${v.videoWidth}×${v.videoHeight} (${mp} MP)`);
    };
  } catch (e) {
    const err = document.getElementById('cam-error');
    err.style.display = 'flex';
    err.textContent = 'No se pudo acceder a la cámara. Verifica permisos. (' + e.message + ')';
  }
}

function stopCameraStream() {
  if (camStream) {
    camStream.getTracks().forEach(t => t.stop());
    camStream = null;
  }
  videoReady = false;
}

async function switchCamera() {
  stopDetectionLoop();
  camFacing = (camFacing === 'environment') ? 'user' : 'environment';
  await startCameraStream();
  if (cvLoaded) startDetectionLoop();
}

function closeCameraModal() {
  stopDetectionLoop();
  stopCameraStream();
  document.getElementById('cam-modal').classList.remove('show');
}

// ── Loop de detección ──
function startDetectionLoop() {
  stopDetectionLoop();
  detectionLoop = setInterval(detectFrame, 150);   // ~6-7 fps
}
function stopDetectionLoop() {
  if (detectionLoop) { clearInterval(detectionLoop); detectionLoop = null; }
}

// ── Detección de la hoja ────────────────────────────────────────────────────
// 1º por las GUÍAS: los 4 cuadrados negros de las esquinas. Funcionan sobre
//    cualquier fondo (mesa blanca, madera, tela), porque no dependen del
//    contraste entre el papel y lo que hay detrás.
// 2º respaldo: el borde del papel (requiere un fondo que contraste).
// Devuelve {pts:[{x,y}×4] en coordenadas del canvas recibido, mode} o null.
function _findFiducialQuad(gray, W, H, method) {
  let bin = null, cnts = null, hier = null;
  try {
    bin = new cv.Mat(); cnts = new cv.MatVector(); hier = new cv.Mat();
    // 1er intento: umbral local (tolera sombras). 2º: umbral global (Otsu),
    // para fondos muy oscuros donde el umbral local "se apaga" cerca del
    // borde del papel.
    if (method === 'otsu') {
      cv.threshold(gray, bin, 0, 255, cv.THRESH_BINARY_INV + cv.THRESH_OTSU);
    } else {
      let bs = Math.round(W / 10) | 1;
      if (bs < 31) bs = 31;
      cv.adaptiveThreshold(gray, bin, 255, cv.ADAPTIVE_THRESH_MEAN_C,
                           cv.THRESH_BINARY_INV, bs, 18);
    }
    // RETR_LIST (no EXTERNAL): sobre un fondo oscuro, el fondo forma un
    // contorno que ENVUELVE la hoja y las guías quedan anidadas dentro.
    cv.findContours(bin, cnts, hier, cv.RETR_LIST, cv.CHAIN_APPROX_SIMPLE);
    const frame = W * H, minA = frame * 0.00008, maxA = frame * 0.02;
    const cands = [];
    for (let i = 0; i < cnts.size(); i++) {
      const c = cnts.get(i);
      const a = cv.contourArea(c);
      if (a >= minA && a <= maxA) {
        const r = cv.boundingRect(c);
        const asp = r.width / r.height;
        const fill = a / (r.width * r.height);
        // Cuadrado sólido: casi tan ancho como alto (descarta las marcas de
        // tiempo, que son rectángulos angostos) y bien relleno.
        if (asp > 0.6 && asp < 1.65 && fill > 0.55) {
          const ap = new cv.Mat();
          cv.approxPolyDP(c, ap, 0.03 * cv.arcLength(c, true), true);
          // 4 vértices convexos: descarta burbujas rellenas (círculos)
          if (ap.rows === 4 && cv.isContourConvex(ap)) {
            cands.push({x: r.x + r.width / 2, y: r.y + r.height / 2, a});
          }
          ap.delete();
        }
      }
      c.delete();
    }
    if (cands.length < 4) return null;
    // Las guías son los cuadrados más extremos en cada diagonal
    const pick = (f, wantMax) => cands.reduce((b, p) =>
      (b === null || (wantMax ? f(p) > f(b) : f(p) < f(b))) ? p : b, null);
    const tl = pick(p => p.x + p.y, false), br = pick(p => p.x + p.y, true);
    const tr = pick(p => p.x - p.y, true),  bl = pick(p => p.x - p.y, false);
    const q = [tl, tr, br, bl];
    if (new Set(q).size < 4) return null;
    const areas = q.map(p => p.a);
    if (Math.max(...areas) / Math.min(...areas) > 5) return null;   // tamaños parecidos
    let polyA = 0;
    for (let i = 0; i < 4; i++) { const n = q[(i + 1) % 4]; polyA += q[i].x * n.y - n.x * q[i].y; }
    if (Math.abs(polyA) / 2 < frame * 0.10) return null;            // hoja muy pequeña
    const wT = dist(tl, tr), wB = dist(bl, br), hL = dist(tl, bl), hR = dist(tr, br);
    if (Math.min(wT, wB) / Math.max(wT, wB) < 0.6) return null;      // perspectiva absurda
    if (Math.min(hL, hR) / Math.max(hL, hR) < 0.6) return null;
    const ratio = Math.min(wT + wB, hL + hR) / Math.max(wT + wB, hL + hR);
    if (ratio < 0.4) return null;                                    // no parece una hoja
    // Ampliar hacia afuera: las guías deben quedar COMPLETAS dentro del
    // recorte (el servidor las vuelve a buscar para enderezar con precisión).
    const cx = q.reduce((s, p) => s + p.x, 0) / 4, cy = q.reduce((s, p) => s + p.y, 0) / 4;
    const fid = Math.sqrt(areas.reduce((s, x) => s + x, 0) / 4);
    return q.map(p => {
      const dx = p.x - cx, dy = p.y - cy, d = Math.hypot(dx, dy) || 1;
      const k = 1 + Math.max(0.05, (fid * 1.6) / d);
      return {x: Math.min(W - 1, Math.max(0, cx + dx * k)),
              y: Math.min(H - 1, Math.max(0, cy + dy * k))};
    });
  } catch (e) {
    return null;
  } finally {
    if (bin) bin.delete(); if (cnts) cnts.delete(); if (hier) hier.delete();
  }
}

function _findPaperEdgeQuad(gray, W, H) {
  let blur = null, edges = null, cnts = null, hier = null;
  try {
    blur = new cv.Mat(); edges = new cv.Mat();
    cnts = new cv.MatVector(); hier = new cv.Mat();
    cv.GaussianBlur(gray, blur, new cv.Size(5, 5), 0);
    cv.Canny(blur, edges, 60, 180);
    const kernel = cv.Mat.ones(3, 3, cv.CV_8U);   // cerrar líneas rotas
    cv.dilate(edges, edges, kernel);
    kernel.delete();
    cv.findContours(edges, cnts, hier, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);
    let best = null, bestArea = 0;
    const minArea = W * H * 0.15;
    for (let i = 0; i < cnts.size(); i++) {
      const c = cnts.get(i);
      const area = cv.contourArea(c);
      if (area >= minArea) {
        const approx = new cv.Mat();
        cv.approxPolyDP(c, approx, 0.02 * cv.arcLength(c, true), true);
        if (approx.rows === 4 && area > bestArea) {
          bestArea = area;
          best = [];
          for (let j = 0; j < 4; j++) {
            best.push({x: approx.data32S[j * 2], y: approx.data32S[j * 2 + 1]});
          }
        }
        approx.delete();
      }
      c.delete();
    }
    return best;
  } catch (e) {
    return null;
  } finally {
    if (blur) blur.delete(); if (edges) edges.delete();
    if (cnts) cnts.delete(); if (hier) hier.delete();
  }
}

function _detectSheetQuad(canvasEl) {
  let src = null, gray = null;
  try {
    src = cv.imread(canvasEl);
    gray = new cv.Mat();
    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    const W = canvasEl.width, H = canvasEl.height;
    const fid = _findFiducialQuad(gray, W, H, 'adaptive') ||
                _findFiducialQuad(gray, W, H, 'otsu');
    if (fid) return {pts: fid, mode: 'guias'};
    const edge = _findPaperEdgeQuad(gray, W, H);
    if (edge) return {pts: edge, mode: 'borde'};
    return null;
  } finally {
    if (src) src.delete(); if (gray) gray.delete();
  }
}

function detectFrame() {
  if (!videoReady || !cvLoaded) return;
  const video  = document.getElementById('cam-video');
  const canvas = document.getElementById('cam-canvas');
  if (!video.videoWidth) return;

  // Frame reducido para acelerar (640 px: suficiente para ver las guías)
  const procW = 640;
  const scale = procW / video.videoWidth;
  const procH = Math.round(video.videoHeight * scale);
  const tmp = document.createElement('canvas');
  tmp.width = procW; tmp.height = procH;
  tmp.getContext('2d').drawImage(video, 0, 0, procW, procH);

  try {
    const det = _detectSheetQuad(tmp);
    const pts = det ? det.pts.map(p => ({x: p.x / scale, y: p.y / scale})) : null;
    drawOverlay(canvas, pts);
    updateHint(pts, det ? det.mode : null);
  } catch (e) {
    console.warn('Detección falló:', e);
  }
}

function drawOverlay(canvas, pts) {
  const ctx = canvas.getContext('2d');
  ctx.clearRect(0, 0, canvas.width, canvas.height);
  if (!pts) return;
  // Polígono semitransparente verde
  ctx.fillStyle   = 'rgba(16, 185, 129, 0.22)';
  ctx.strokeStyle = '#10b981';
  ctx.lineWidth   = 6;
  ctx.beginPath();
  ctx.moveTo(pts[0].x, pts[0].y);
  for (let i = 1; i < pts.length; i++) ctx.lineTo(pts[i].x, pts[i].y);
  ctx.closePath();
  ctx.fill();
  ctx.stroke();
  // Vértices
  ctx.fillStyle = '#10b981';
  pts.forEach(p => {
    ctx.beginPath(); ctx.arc(p.x, p.y, 10, 0, Math.PI*2); ctx.fill();
  });
}

function updateHint(pts, mode) {
  const hint = document.getElementById('cam-hint');
  if (pts) {
    hint.textContent = mode === 'guias' ? '✓ Hoja detectada' : '✓ Documento detectado';
    hint.classList.add('detected');
    lastDetected = pts;
  } else {
    hint.textContent = 'Encuadra los 4 cuadros negros de las esquinas';
    hint.classList.remove('detected');
    lastDetected = null;
  }
}

// Ordenar 4 puntos en orden: TL, TR, BR, BL
function orderQuadPoints(pts) {
  // Suma x+y: el menor es TL, el mayor es BR
  // Diferencia x-y: el menor es TR, el mayor es BL  (ojo: signo)
  const sums  = pts.map(p => p.x + p.y);
  const diffs = pts.map(p => p.x - p.y);
  const tl = pts[sums.indexOf(Math.min(...sums))];
  const br = pts[sums.indexOf(Math.max(...sums))];
  const tr = pts[diffs.indexOf(Math.max(...diffs))];
  const bl = pts[diffs.indexOf(Math.min(...diffs))];
  return [tl, tr, br, bl];
}

function dist(a, b) { return Math.hypot(a.x - b.x, a.y - b.y); }

async function captureFromCamera() {
  const video = document.getElementById('cam-video');
  if (!video.videoWidth) { alert('La cámara aún no está lista.'); return; }

  // Capturar frame completo a un canvas en tamaño original
  const fullCanvas = document.createElement('canvas');
  fullCanvas.width  = video.videoWidth;
  fullCanvas.height = video.videoHeight;
  fullCanvas.getContext('2d').drawImage(video, 0, 0);

  // ── Leer el QR del FRAME COMPLETO antes de recortar ──
  // El recorte por perspectiva (warpPerspective) puede dejar fuera el QR del
  // encabezado, así que lo decodificamos aquí y lo enviamos como pista.
  _pendingQrHint = await _decodeQrFromCanvas(fullCanvas);

  // Si tenemos detección, recortar y enderezar el documento
  if (cvLoaded && lastDetected && lastDetected.length === 4) {
    try {
      const ordered = orderQuadPoints(lastDetected);
      const [tl, tr, br, bl] = ordered;
      const widthA  = dist(br, bl);
      const widthB  = dist(tr, tl);
      const heightA = dist(tr, br);
      const heightB = dist(tl, bl);
      const outW = Math.round(Math.max(widthA, widthB));
      const outH = Math.round(Math.max(heightA, heightB));

      const src = cv.imread(fullCanvas);
      const srcPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
        tl.x, tl.y, tr.x, tr.y, br.x, br.y, bl.x, bl.y
      ]);
      const dstPts = cv.matFromArray(4, 1, cv.CV_32FC2, [
        0, 0, outW, 0, outW, outH, 0, outH
      ]);
      const M = cv.getPerspectiveTransform(srcPts, dstPts);
      const dst = new cv.Mat();
      cv.warpPerspective(src, dst, M, new cv.Size(outW, outH),
        cv.INTER_LINEAR, cv.BORDER_CONSTANT, new cv.Scalar());

      const outCanvas = document.createElement('canvas');
      outCanvas.width  = outW;
      outCanvas.height = outH;
      cv.imshow(outCanvas, dst);

      src.delete(); srcPts.delete(); dstPts.delete(); M.delete(); dst.delete();

      outCanvas.toBlob(blob => sendBlob(blob), 'image/jpeg', 0.92);
      return;
    } catch (e) {
      console.warn('Recorte falló, usando frame completo:', e);
    }
  }

  // Fallback: enviar el frame completo
  fullCanvas.toBlob(blob => sendBlob(blob), 'image/jpeg', 0.92);
}

function sendBlob(blob) {
  if (!blob) { alert('No se pudo capturar la imagen.'); return; }
  const file = new File([blob], `camara_${Date.now()}.jpg`, {type: 'image/jpeg'});
  const dt = new DataTransfer();
  dt.items.add(file);
  document.getElementById('file-input').files = dt.files;
  closeCameraModal();
  document.getElementById('file-input').dispatchEvent(new Event('change'));
}
