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
const OPENCV_URL   = 'https://docs.opencv.org/4.10.0/opencv.js';



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
    const script = document.createElement('script');
    script.src = OPENCV_URL;
    script.async = true;
    script.onload = () => {
      // OpenCV.js inicializa de forma asíncrona
      const checkReady = () => {
        if (window.cv && window.cv.Mat) {
          cvLoaded = true; cvLoading = false; resolve(true);
        } else if (window.cv && typeof window.cv.onRuntimeInitialized === 'function') {
          window.cv.onRuntimeInitialized = () => { cvLoaded = true; cvLoading = false; resolve(true); };
        } else if (window.cv) {
          window.cv.onRuntimeInitialized = () => { cvLoaded = true; cvLoading = false; resolve(true); };
        } else {
          setTimeout(checkReady, 100);
        }
      };
      checkReady();
    };
    script.onerror = () => { cvLoading = false; resolve(false); };
    document.head.appendChild(script);
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

function detectFrame() {
  if (!videoReady || !cvLoaded) return;
  const video  = document.getElementById('cam-video');
  const canvas = document.getElementById('cam-canvas');
  if (!video.videoWidth) return;

  // Capturar frame a un canvas auxiliar pequeño para acelerar
  const procW = 480;
  const scale = procW / video.videoWidth;
  const procH = Math.round(video.videoHeight * scale);
  const tmp = document.createElement('canvas');
  tmp.width = procW; tmp.height = procH;
  tmp.getContext('2d').drawImage(video, 0, 0, procW, procH);

  let src, gray, blur, edges, contours, hierarchy;
  try {
    src      = cv.imread(tmp);
    gray     = new cv.Mat();
    blur     = new cv.Mat();
    edges    = new cv.Mat();
    contours = new cv.MatVector();
    hierarchy = new cv.Mat();

    cv.cvtColor(src, gray, cv.COLOR_RGBA2GRAY);
    cv.GaussianBlur(gray, blur, new cv.Size(5,5), 0);
    cv.Canny(blur, edges, 60, 180);
    // Dilatar para cerrar líneas rotas
    const kernel = cv.Mat.ones(3,3, cv.CV_8U);
    cv.dilate(edges, edges, kernel);
    kernel.delete();

    cv.findContours(edges, contours, hierarchy, cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE);

    // Buscar el contorno cuadrilátero más grande
    let bestPts = null;
    let bestArea = 0;
    const minArea = procW * procH * 0.15;  // al menos 15% de la imagen
    for (let i = 0; i < contours.size(); i++) {
      const c = contours.get(i);
      const area = cv.contourArea(c);
      if (area < minArea) { c.delete(); continue; }
      const peri = cv.arcLength(c, true);
      const approx = new cv.Mat();
      cv.approxPolyDP(c, approx, 0.02 * peri, true);
      if (approx.rows === 4 && area > bestArea) {
        bestArea = area;
        bestPts = [];
        for (let j = 0; j < 4; j++) {
          bestPts.push({
            x: approx.data32S[j*2]   / scale,
            y: approx.data32S[j*2+1] / scale,
          });
        }
      }
      approx.delete();
      c.delete();
    }

    drawOverlay(canvas, bestPts);
    updateHint(bestPts);
  } catch (e) {
    console.warn('Detección falló:', e);
  } finally {
    if (src) src.delete();
    if (gray) gray.delete();
    if (blur) blur.delete();
    if (edges) edges.delete();
    if (contours) contours.delete();
    if (hierarchy) hierarchy.delete();
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

function updateHint(pts) {
  const hint = document.getElementById('cam-hint');
  if (pts) {
    hint.textContent = '✓ Documento detectado';
    hint.classList.add('detected');
    lastDetected = pts;
  } else {
    hint.textContent = 'Buscando documento…';
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
