/* pdf_individual.js — Genera el PDF de "Reporte individual" del estudiante.
 * Usado tanto por el dashboard admin (templates/index.html) como por
 * el portal estudiantil (templates/portal_estudiante.html).
 *
 * API pública:
 *   - renderStudentPage(doc, student, pcts, addPage, simulacroLabel)
 *   - calcPercentiles(results)
 *   - _loadLogo()  (precarga los logos)
 *   - getJsPDF()
 *
 * Espera que jsPDF esté cargado vía <script> antes de este archivo.
 */

(function (window) {
  'use strict';

  const SUBJECTS = [
    {name:'Matematicas',          key:'mat'},
    {name:'Lectura Critica',      key:'lect'},
    {name:'Sociales y Ciudadanas',key:'soc'},
    {name:'Ciencias Naturales',   key:'nat'},
    {name:'Ingles',               key:'ing'},
  ];

  // Helper para derivar grado del curso ("1101" → "11", "11A" → "11", "902" → "9")
  // Resolver de grado inyectable: la app de admin registra su propia versión
  // (que consulta el catálogo de cursos) vía setGradoResolver(). El default
  // cubre los formatos comunes: "1101"→"11", "601"→"6", "11A"→"11".
  let _gradoResolver = null;
  function setGradoResolver(fn) { _gradoResolver = fn; }

  function gradoDeCurso(curso) {
    // Cursos de sede ("CV-301"): el prefijo de la sede no cuenta para el grado
    const raw = String(curso || '').trim();
    const c = raw.replace(/^[A-Za-z]{1,4}-/, '');
    if (!c) return '';
    if (_gradoResolver) {
      try {
        const g = _gradoResolver(raw);
        if (g) return String(g);
      } catch (e) { /* caer al default */ }
    }
    if (/^\d{4}/.test(c)) return c.slice(0, 2);   // 1101 → 11
    if (/^\d{3}/.test(c)) return c.slice(0, 1);   // 601  → 6
    const m = c.match(/^(\d{1,2})/);              // 11A  → 11
    return m ? m[1] : '';
  }

  // ── Primaria (1° a 5°) ──
  // El reporte muestra solo las áreas que evalúa el grado (las demás llegan
  // vacías), el área de lectura se llama "Lenguaje" y en lugar del percentil
  // se muestra el nivel de desempeño ICFES de cada área.
  function esPrimaria(curso) {
    const g = parseInt(gradoDeCurso(curso), 10);
    return g >= 1 && g <= 5;
  }
  // Cortes 0-100 (tope de cada nivel) — los mismos del panel de resultados.
  const ICFES_CORTES = {
    mat: [35, 50, 70], lect: [35, 50, 65], soc: [40, 55, 70],
    nat: [40, 55, 70], ing: [36, 57, 70],
  };
  const NIVEL_NOMBRES = ['Insuficiente', 'Mínimo', 'Satisfactorio', 'Avanzado'];
  const NIVEL_FUERTE = [null, [226, 75, 74], [239, 159, 39], [234, 190, 40], [76, 175, 80]];
  const NIVEL_TENUE  = [null, [248, 205, 205], [251, 222, 178], [252, 238, 180], [205, 233, 206]];
  const NIVEL_TEXTO  = [null, [121, 31, 31], [110, 60, 5], [100, 74, 5], [27, 94, 32]];
  function nivelIcfes(key, score) {
    const c = ICFES_CORTES[key];
    if (!c || score === '' || score == null || isNaN(Number(score))) return 0;
    const v = Number(score);
    return v <= c[0] ? 1 : v <= c[1] ? 2 : v <= c[2] ? 3 : 4;
  }
  function rangoNivel(key, n) {
    const c = ICFES_CORTES[key];
    if (!c || !n) return '';
    return `${n === 1 ? 0 : c[n - 2] + 1}-${n === 4 ? 100 : c[n - 1]}`;
  }
  const _evaluada = v => v !== '' && v != null && !isNaN(Number(v));

  // Escalera de 4 niveles con el del estudiante resaltado
  function _drawNivelCell(doc, x, y, w, h, key, score) {
    const n = nivelIcfes(key, score);
    const segW = Math.min(4.2, (w * 0.42 - 3) / 4), segH = 2.6;
    const ladderW = segW * 4 + 1.5 * 3;
    const lx0 = x + w - ladderW - 1;
    for (let k = 1; k <= 4; k++) {
      const c = k === n ? NIVEL_FUERTE[k] : k < n ? NIVEL_TENUE[k] : [226, 232, 240];
      doc.setFillColor(...c);
      doc.roundedRect(lx0 + (k - 1) * (segW + 1.5), y + h / 2 - segH / 2, segW, segH, 0.8, 0.8, 'F');
    }
    if (!n) return;
    const pillW = lx0 - x - 2.5;
    doc.setFillColor(...NIVEL_TENUE[n]);
    doc.roundedRect(x, y + 1, pillW, h - 2, 2, 2, 'F');
    doc.setTextColor(...NIVEL_TEXTO[n]);
    doc.setFont('helvetica', 'bold'); doc.setFontSize(8.2);
    doc.text(NIVEL_NOMBRES[n - 1], x + pillW / 2, y + h / 2 + 0.2, {align: 'center'});
    doc.setFont('helvetica', 'normal'); doc.setFontSize(5.8);
    doc.text(`Nivel ${n} · ${rangoNivel(key, n)} puntos`, x + pillW / 2, y + h / 2 + 3.3, {align: 'center'});
  }

  // Tarjeta con la explicación de los 4 niveles (reemplaza al percentil)
  function _drawNivelesCard(doc, x, y, w, h) {
    doc.setFillColor(248, 250, 252); doc.setDrawColor(226, 232, 240); doc.setLineWidth(0.3);
    doc.roundedRect(x, y, w, h, 3, 3, 'FD');
    doc.setFont('helvetica', 'bold'); doc.setFontSize(8); doc.setTextColor(60, 52, 137);
    doc.text('NIVELES DE DESEMPEÑO ICFES', x + w / 2, y + 6.5, {align: 'center'});
    const desc = ['Aún no alcanza lo básico del área.', 'Logra lo básico del área.',
                  'Logra lo esperado para el grado.', 'Supera lo esperado para el grado.'];
    const rowH = (h - 11) / 4;
    for (let k = 4; k >= 1; k--) {
      const ry = y + 9.5 + (4 - k) * rowH;
      doc.setFillColor(...NIVEL_FUERTE[k]);
      doc.roundedRect(x + 4, ry + rowH / 2 - 2.2, 4.4, 4.4, 1, 1, 'F');
      doc.setFont('helvetica', 'bold'); doc.setFontSize(7.8); doc.setTextColor(...NIVEL_TEXTO[k]);
      doc.text(`${NIVEL_NOMBRES[k - 1]}`, x + 11, ry + rowH / 2 - 0.4);
      doc.setFont('helvetica', 'normal'); doc.setFontSize(6.3); doc.setTextColor(100, 116, 139);
      doc.text(desc[k - 1], x + 11, ry + rowH / 2 + 2.9);
    }
  }

  // Percentiles POR GRADO
  function calcPercentiles(results) {
    const fields = ['mat','lect','soc','nat','ing','general'];
    if (!results.length) return {};
    const groups = {};
    results.forEach((r, idx) => {
      const g = gradoDeCurso(r.curso) || 'sin_grado';
      if (!groups[g]) groups[g] = [];
      groups[g].push({ r, idx });
    });
    const pcts = {};
    Object.entries(groups).forEach(([_g, members]) => {
      const sorted = {};
      fields.forEach(f => {
        sorted[f] = members.map(({r}) => Number(r[f])||0).sort((a,b) => a-b);
      });
      const n = members.length;
      members.forEach(({r, idx}) => {
        const key = r.id ? String(r.id) : `_idx${idx}`;
        pcts[key] = {};
        fields.forEach(f => {
          const val = Number(r[f])||0;
          const count = sorted[f].filter(v => v <= val).length;
          pcts[key][f] = Math.round(count / n * 100);
        });
      });
    });
    return pcts;
  }

  function _newCanvas(w, h) {
    const c = document.createElement('canvas');
    c.width = w; c.height = h;
    return c;
  }

  let _logoDataUrl = null;
  let _sipagreLogoDataUrl = null;
  let _ietagroLogoDataUrl = null;

  // Elección de logo del colegio: 'auto' | 'sipagre' | 'ietagro'.
  // El portal usa 'auto' (decide por el nombre del simulacro); la app de
  // admin la fija desde su selector vía setLogoChoice().
  let _logoChoice = 'auto';
  function setLogoChoice(choice) {
    _logoChoice = (choice === 'sipagre' || choice === 'ietagro') ? choice : 'auto';
  }

  // Resuelve qué colegio aplica para un simulacro dado ('ietagro'|'sipagre').
  // Útil también para mostrar pistas en la UI sin duplicar esta lógica.
  function resolveLogoChoice(simulacroLabel) {
    if (_logoChoice !== 'auto') return _logoChoice;
    const name = (simulacroLabel || '').toUpperCase();
    return name.includes('IETAGRO') ? 'ietagro' : 'sipagre';
  }

  function _headerLogoForSim(simulacroLabel) {
    if (resolveLogoChoice(simulacroLabel) === 'ietagro' && _ietagroLogoDataUrl) {
      return _ietagroLogoDataUrl;
    }
    return _sipagreLogoDataUrl;
  }

  function _loadImageAsDataUrl(url) {
    return new Promise(resolve => {
      const img = new Image();
      img.crossOrigin = 'anonymous';
      img.onload = () => {
        const c = _newCanvas(img.naturalWidth, img.naturalHeight);
        c.getContext('2d').drawImage(img, 0, 0);
        try { resolve(c.toDataURL('image/png')); }
        catch (e) { resolve(''); }
      };
      img.onerror = () => resolve('');
      img.src = url;
    });
  }

  async function _loadLogo() {
    // Versiones _sm (256px): a tamaño de impresión se ven idénticas y el PDF
    // pasa de ~12 MB a <1 MB (las originales pesan hasta 1.2 MB cada una).
    if (_logoDataUrl === null)        _logoDataUrl        = await _loadImageAsDataUrl('/static/logo_sm.png');
    if (_sipagreLogoDataUrl === null) _sipagreLogoDataUrl = await _loadImageAsDataUrl('/static/sipagre_logo_sm.png');
    if (_ietagroLogoDataUrl === null) _ietagroLogoDataUrl = await _loadImageAsDataUrl('/static/ietagro_logo_sm.png');
    await _loadKidsFont();   // tipografía del reporte de primaria
    return _logoDataUrl;
  }

  // ── Iconos de asignatura (emoji → imagen) y bandera UK ─────────────────────
  // Los emojis no existen en las fuentes del PDF, pero el canvas del navegador
  // sí los dibuja a color; se convierten a PNG e incrustan como imágenes.
  // La bandera 🇬🇧 NO se usa como emoji porque Windows no la renderiza
  // (muestra "GB"): se dibuja una Union Jack en canvas, idéntica en todo OS.
  const SUBJECT_ICONS = { mat:'🧮', lect:'📖', soc:'🌍', nat:'🔬', ing:'__FLAG__' };
  const _emojiCache = {};
  function _emojiDataUrl(ch, size = 96) {
    const key = ch + '@' + size;
    if (_emojiCache[key] !== undefined) return _emojiCache[key];
    let url = '';
    try {
      const c = _newCanvas(size, size);
      const ctx = c.getContext('2d');
      ctx.font = Math.round(size * 0.8) +
        'px "Segoe UI Emoji","Apple Color Emoji","Noto Color Emoji",sans-serif';
      ctx.textAlign = 'center'; ctx.textBaseline = 'middle';
      ctx.fillText(ch, size / 2, size / 2 + size * 0.05);
      url = c.toDataURL('image/png');
    } catch (e) { url = ''; }
    _emojiCache[key] = url;
    return url;
  }

  function _drawFlagUK(ctx, x, y, w, h) {
    ctx.save();
    ctx.beginPath(); ctx.rect(x, y, w, h); ctx.clip();
    ctx.fillStyle = '#012169'; ctx.fillRect(x, y, w, h);
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = h * 0.20;
    ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x + w, y + h);
    ctx.moveTo(x + w, y); ctx.lineTo(x, y + h); ctx.stroke();
    ctx.strokeStyle = '#C8102E'; ctx.lineWidth = h * 0.10;
    ctx.beginPath();
    ctx.moveTo(x, y); ctx.lineTo(x + w, y + h);
    ctx.moveTo(x + w, y); ctx.lineTo(x, y + h); ctx.stroke();
    ctx.strokeStyle = '#ffffff'; ctx.lineWidth = h * 0.33;
    ctx.beginPath();
    ctx.moveTo(x + w / 2, y); ctx.lineTo(x + w / 2, y + h);
    ctx.moveTo(x, y + h / 2); ctx.lineTo(x + w, y + h / 2); ctx.stroke();
    ctx.strokeStyle = '#C8102E'; ctx.lineWidth = h * 0.20;
    ctx.beginPath();
    ctx.moveTo(x + w / 2, y); ctx.lineTo(x + w / 2, y + h);
    ctx.moveTo(x, y + h / 2); ctx.lineTo(x + w, y + h / 2); ctx.stroke();
    ctx.restore();
  }

  let _flagUrl = null;
  function _flagDataUrl() {
    if (_flagUrl !== null) return _flagUrl;
    try {
      const c = _newCanvas(120, 72);
      _drawFlagUK(c.getContext('2d'), 0, 0, 120, 72);
      _flagUrl = c.toDataURL('image/png');
    } catch (e) { _flagUrl = ''; }
    return _flagUrl;
  }

  function _subjectIconUrl(key) {
    const ic = SUBJECT_ICONS[key];
    if (!ic) return '';
    return ic === '__FLAG__' ? _flagDataUrl() : _emojiDataUrl(ic);
  }

  // Medalla + mensaje motivacional según el puntaje general (escala 0-500).
  // Rangos y textos definidos por el docente; solo 3 niveles/iconos.
  function _medalFor(general) {
    if (general >= 300) return { e: '🏆',
      msg: 'Con esfuerzo, disciplina, perseverancia y constancia podrás cumplir tus metas' };
    if (general >= 221) return { e: '💪',
      msg: 'La clave del éxito es enfocarse en metas, no en obstáculos' };
    return { e: '📈',
      msg: 'El esfuerzo que pones hoy, será tu orgullo mañana' };
  }

  function drawGaugeCanvas(value, max, label, medalChar) {
    const W = 700, H = 480;
    const c = _newCanvas(W, H);
    const ctx = c.getContext('2d');
    const cx = W/2, cy = H*0.62, r = 200;
    const lw = 44;
    const startA = Math.PI, endA = 2*Math.PI;
    const total = endA - startA;
    const zones = [
      {from: 0,    to: 0.25, color: '#ef4444'},
      {from: 0.25, to: 0.50, color: '#f59e0b'},
      {from: 0.50, to: 0.75, color: '#eab308'},
      {from: 0.75, to: 1.00, color: '#22c55e'},
    ];
    ctx.lineWidth = lw;
    ctx.lineCap = 'butt';
    zones.forEach(z => {
      ctx.beginPath();
      ctx.strokeStyle = z.color;
      ctx.arc(cx, cy, r, startA + total*z.from, startA + total*z.to);
      ctx.stroke();
    });
    ctx.fillStyle = '#1e293b';
    ctx.font = 'bold 28px Inter, Arial';
    ctx.textAlign = 'center';
    ctx.fillText('0',          cx - r,     cy + 45);
    ctx.fillText(String(max),  cx + r,     cy + 45);
    const v = Math.min(Math.max(value, 0), max);
    const ang = startA + total * (v / max);
    ctx.save();
    ctx.translate(cx, cy);
    ctx.rotate(ang);
    ctx.fillStyle = '#334155';
    ctx.beginPath();
    ctx.moveTo(0, -14);
    ctx.lineTo(r - 8, 0);
    ctx.lineTo(0, 14);
    ctx.closePath();
    ctx.fill();
    ctx.beginPath();
    ctx.arc(0, 0, 22, 0, Math.PI*2);
    ctx.fillStyle = '#1e293b';
    ctx.fill();
    ctx.beginPath();
    ctx.arc(0, 0, 8, 0, Math.PI*2);
    ctx.fillStyle = '#94a3b8';
    ctx.fill();
    ctx.restore();
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 72px Inter, Arial';
    ctx.fillText(String(value), cx, cy + 110);
    if (medalChar) {
      // Medalla al lado del número (el canvas sí renderiza emoji a color)
      const numW = ctx.measureText(String(value)).width;
      ctx.font = '52px "Segoe UI Emoji","Apple Color Emoji","Noto Color Emoji",sans-serif';
      ctx.textBaseline = 'alphabetic';
      ctx.fillText(medalChar, cx + numW / 2 + 44, cy + 106);
    }
    ctx.fillStyle = '#64748b';
    ctx.font = '500 24px Inter, Arial';
    ctx.fillText(`/ ${max}`, cx, cy + 140);
    if (label) {
      ctx.fillStyle = '#1e293b';
      ctx.font = 'bold 30px Inter, Arial';
      ctx.fillText(label, cx, 50);
    }
    return c;
  }

  function drawDonutCanvas(percent, label, color) {
    const W = 380, H = 380;
    const c = _newCanvas(W, H);
    const ctx = c.getContext('2d');
    const cx = W/2, cy = H/2, r = 130;
    const lw = 38;
    ctx.lineWidth = lw;
    ctx.lineCap = 'round';
    ctx.strokeStyle = '#e2e8f0';
    ctx.beginPath();
    ctx.arc(cx, cy, r, 0, Math.PI*2);
    ctx.stroke();
    const p = Math.max(0, Math.min(100, percent)) / 100;
    ctx.strokeStyle = color || '#6366f1';
    ctx.beginPath();
    ctx.arc(cx, cy, r, -Math.PI/2, -Math.PI/2 + Math.PI*2 * p);
    ctx.stroke();
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 92px Inter, Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(String(Math.round(percent)), cx, cy - 8);
    ctx.fillStyle = '#64748b';
    ctx.font = 'bold 22px Inter, Arial';
    ctx.fillText(label || 'PERCENTIL', cx, cy + 50);
    return c;
  }

  function drawPercentileBarCanvas(percent) {
    // Barra espectro rojo→verde con marcador ▼ y número pequeños encima
    // (más discretos para no interponerse con la fila superior).
    const W = 700, H = 84;
    const c = _newCanvas(W, H);
    const ctx = c.getContext('2d');
    const x = 14, y = 54, w = W - 28, h = 18;
    const grad = ctx.createLinearGradient(x, 0, x + w, 0);
    grad.addColorStop(0,    '#ef4444');
    grad.addColorStop(0.4,  '#f59e0b');
    grad.addColorStop(0.7,  '#eab308');
    grad.addColorStop(1,    '#22c55e');
    ctx.fillStyle = grad;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, 9);
    else ctx.rect(x, y, w, h);
    ctx.fill();
    const pc = Math.max(0, Math.min(100, percent));
    const px = x + (w * pc / 100);
    ctx.fillStyle = '#475569';
    ctx.beginPath();
    ctx.moveTo(px - 7, y - 11);
    ctx.lineTo(px + 7, y - 11);
    ctx.lineTo(px, y - 2);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#475569';
    ctx.font = 'bold 27px Inter, Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    // El número no se sale del lienzo en los extremos (0 o 100)
    const tx = Math.max(x + 18, Math.min(x + w - 18, px));
    ctx.fillText(String(percent), tx, y - 18);
    return c;
  }

  function drawRadarCanvas(labels, values, maxVal) {
    const W = 700, H = 700;
    const c = _newCanvas(W, H);
    const ctx = c.getContext('2d');
    const cx = W/2, cy = H/2, R = 230;
    const N = labels.length;
    ctx.strokeStyle = '#cbd5e1';
    ctx.lineWidth = 1.5;
    for (let lvl = 1; lvl <= 4; lvl++) {
      const r = R * lvl / 4;
      ctx.beginPath();
      for (let i = 0; i < N; i++) {
        const a = -Math.PI/2 + 2*Math.PI*i/N;
        const x = cx + r*Math.cos(a);
        const y = cy + r*Math.sin(a);
        if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
      }
      ctx.closePath();
      ctx.stroke();
    }
    ctx.setLineDash([4,4]);
    for (let i = 0; i < N; i++) {
      const a = -Math.PI/2 + 2*Math.PI*i/N;
      ctx.beginPath();
      ctx.moveTo(cx, cy);
      ctx.lineTo(cx + R*Math.cos(a), cy + R*Math.sin(a));
      ctx.stroke();
    }
    ctx.setLineDash([]);
    ctx.beginPath();
    for (let i = 0; i < N; i++) {
      const a = -Math.PI/2 + 2*Math.PI*i/N;
      const v = Math.min(values[i] / maxVal, 1);
      const x = cx + R*v*Math.cos(a);
      const y = cy + R*v*Math.sin(a);
      if (i === 0) ctx.moveTo(x,y); else ctx.lineTo(x,y);
    }
    ctx.closePath();
    ctx.fillStyle = 'rgba(34, 197, 94, 0.35)';
    ctx.fill();
    ctx.strokeStyle = '#16a34a';
    ctx.lineWidth = 2.5;
    ctx.stroke();
    for (let i = 0; i < N; i++) {
      const a = -Math.PI/2 + 2*Math.PI*i/N;
      const v = Math.min(values[i] / maxVal, 1);
      const x = cx + R*v*Math.cos(a);
      const y = cy + R*v*Math.sin(a);
      ctx.fillStyle = '#16a34a';
      ctx.beginPath(); ctx.arc(x, y, 6, 0, Math.PI*2); ctx.fill();
      const valStr = String(Math.round(values[i]));
      ctx.font = 'bold 20px Inter, Arial';
      const tw = ctx.measureText(valStr).width;
      const offR = 18;
      const tx = cx + (R*v + offR)*Math.cos(a);
      const ty = cy + (R*v + offR)*Math.sin(a);
      ctx.fillStyle = 'rgba(255,255,255,0.92)';
      ctx.fillRect(tx - tw/2 - 4, ty - 12, tw + 8, 22);
      ctx.fillStyle = '#16a34a';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillText(valStr, tx, ty);
    }
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < N; i++) {
      const a = -Math.PI/2 + 2*Math.PI*i/N;
      const lblR = R + 52;
      const x = cx + lblR*Math.cos(a);
      const y = cy + lblR*Math.sin(a);
      if (labels[i] === '__FLAG__') {
        // Inglés: bandera británica dibujada (los emoji de bandera no
        // se renderizan en Windows)
        _drawFlagUK(ctx, x - 27, y - 16, 54, 32);
      } else {
        ctx.fillStyle = '#1e293b';
        ctx.font = '44px "Segoe UI Emoji","Apple Color Emoji","Noto Color Emoji",sans-serif';
        ctx.fillText(labels[i], x, y);
      }
    }
    return c;
  }

  function _drawSectionHeader(doc, x, y, w, title) {
    const steps = 60;
    const h = 9;
    for (let i = 0; i < steps; i++) {
      const t = i / steps;
      const r = Math.round(99  + (6   - 99 ) * t);
      const g = Math.round(102 + (182 - 102) * t);
      const b = Math.round(241 + (212 - 241) * t);
      doc.setFillColor(r, g, b);
      doc.rect(x + (w * i / steps), y, w / steps + 0.3, h, 'F');
    }
    doc.setFillColor(255, 255, 255);
    doc.rect(x + 2, y + 2, 2.5, h - 4, 'F');
    doc.setFont('helvetica','bold'); doc.setFontSize(11);
    doc.setTextColor(255, 255, 255);
    doc.text(title, x + 9, y + 6);
  }

  function _drawScoreBadge(doc, x, y, w, h, score, label) {
    doc.setFillColor(99, 102, 241);
    doc.roundedRect(x, y, w, h, 2.5, 2.5, 'F');
    doc.setFont('helvetica','bold'); doc.setFontSize(7);
    doc.setTextColor(255, 255, 255);
    doc.text(label, x + w/2, y + 3.5, {align: 'center'});
    doc.setFont('helvetica','bold'); doc.setFontSize(15);
    doc.setTextColor(255, 255, 255);
    doc.text(String(score), x + w/2, y + h - 2.2, {align: 'center'});
  }

  // ═══════════════════════════════════════════════════════════════════════
  // REPORTE INDIVIDUAL DE PRIMARIA — estética infantil
  // Mismo contenido que el de bachillerato adaptado a primaria: datos del
  // estudiante, puntaje general (0-500) con medalla y mensaje, explicación
  // de los niveles ICFES y, por área evaluada, su puntaje, nivel (nombre,
  // número y rango) y la mejor área. Tipografía Comic Neue (SIL OFL,
  // /static/fonts) incrustada en el PDF; si no carga, se usa Helvetica.
  // ═══════════════════════════════════════════════════════════════════════
  let _kidsFont = null;   // {r, b} en base64 · false si no se pudo cargar
  async function _fetchB64(url) {
    const buf = new Uint8Array(await (await fetch(url)).arrayBuffer());
    let bin = '';
    for (let i = 0; i < buf.length; i += 0x8000) bin += String.fromCharCode.apply(null, buf.subarray(i, i + 0x8000));
    return btoa(bin);
  }
  async function _loadKidsFont() {
    if (_kidsFont !== null) return;
    try {
      const [r, b] = await Promise.all([_fetchB64('/static/fonts/ComicNeue-Regular.ttf'),
                                        _fetchB64('/static/fonts/ComicNeue-Bold.ttf')]);
      _kidsFont = {r, b};
      // La misma letra para los textos que se dibujan en canvas (arcoíris)
      if (window.FontFace && document.fonts) {
        const ff = new FontFace('Comic Neue KJ', 'url(data:font/ttf;base64,' + b + ')', {weight: 'bold'});
        document.fonts.add(await ff.load());
      }
    } catch (e) { _kidsFont = false; }
  }
  function _kidsFontOn(doc) {
    if (!_kidsFont) return 'helvetica';
    if (!(doc.getFontList() || {}).ComicNeue) {
      doc.addFileToVFS('ComicNeue-Regular.ttf', _kidsFont.r); doc.addFont('ComicNeue-Regular.ttf', 'ComicNeue', 'normal');
      doc.addFileToVFS('ComicNeue-Bold.ttf', _kidsFont.b);    doc.addFont('ComicNeue-Bold.ttf', 'ComicNeue', 'bold');
    }
    return 'ComicNeue';
  }

  const KIDS = {
    tinta: [30, 58, 95], suave: [100, 116, 139], papel: [255, 251, 240],
    cielo1: [79, 195, 247], cielo2: [149, 117, 205], sol: [255, 217, 61],
  };
  const KIDS_AREAS = {
    mat:  {nombre: 'Matemáticas', emoji: '🧮', color: [66, 133, 244], claro: [227, 238, 255]},
    lect: {nombre: 'Lenguaje',    emoji: '📖', color: [255, 145, 60], claro: [255, 238, 222]},
    nat:  {nombre: 'Naturales',   emoji: '🔬', color: [46, 184, 114], claro: [222, 246, 233]},
    soc:  {nombre: 'Sociales',    emoji: '🌍', color: [156, 98, 205], claro: [240, 230, 250]},
    ing:  {nombre: 'Inglés',      emoji: '💬', color: [236, 72, 140], claro: [253, 228, 239]},
  };
  const KIDS_NIVELES = [
    {emoji: '🌱', color: [239, 83, 80],  claro: [253, 226, 226], frase: 'Estoy empezando: ¡a practicar!'},
    {emoji: '🌿', color: [255, 167, 38], claro: [255, 239, 214], frase: 'Ya sé lo básico.'},
    {emoji: '🌳', color: [102, 187, 106], claro: [226, 244, 227], frase: 'Sé lo esperado para mi grado.'},
    {emoji: '🏆', color: [41, 121, 255], claro: [222, 234, 255], frase: '¡Voy más allá de lo esperado!'},
  ];
  function _kidsMedalla(g) {
    if (g >= 400) return {e: '🏆', msg: '¡Increíble! Eres una estrella brillante.'};
    if (g >= 300) return {e: '🥇', msg: '¡Muy bien! Tu esfuerzo se nota.'};
    if (g >= 200) return {e: '🥈', msg: '¡Vas por buen camino! Sigue practicando.'};
    return {e: '🌟', msg: 'Cada día aprendes más. ¡Tú puedes!'};
  }
  const _kEmoji = ch => _emojiDataUrl(ch, 120);

  // Arcoíris con aguja: puntaje general 0-500
  function _kidsArcoiris(valor, max) {
    const W = 760, H = 530, c = _newCanvas(W, H);
    const x = c.getContext('2d'), cx = W / 2, cy = 360, R = 300;
    ['#ef5350', '#ffa726', '#ffd54f', '#66bb6a', '#42a5f5', '#7e57c2'].forEach((col, i) => {
      x.beginPath(); x.lineWidth = 34; x.strokeStyle = col; x.lineCap = 'butt';
      x.arc(cx, cy, R - i * 34, Math.PI, 2 * Math.PI); x.stroke();
    });
    const nube = (nx, ny, s) => {
      x.fillStyle = '#fff';
      [[0, 0, 1], [-0.9, 0.25, 0.75], [0.9, 0.25, 0.75], [0, 0.35, 0.8]]
        .forEach(([dx, dy, r]) => { x.beginPath(); x.arc(nx + dx * s, ny + dy * s, r * s, 0, 7); x.fill(); });
    };
    nube(cx - R + 60, cy + 10, 46); nube(cx + R - 60, cy + 10, 46);
    const t = Math.max(0, Math.min(1, valor / max)), a = Math.PI + t * Math.PI;
    x.strokeStyle = '#1e3a5f'; x.lineWidth = 12; x.lineCap = 'round';
    x.beginPath(); x.moveTo(cx, cy); x.lineTo(cx + Math.cos(a) * (R - 150), cy + Math.sin(a) * (R - 150)); x.stroke();
    x.fillStyle = '#1e3a5f'; x.beginPath(); x.arc(cx, cy, 20, 0, 7); x.fill();
    x.fillStyle = '#fff'; x.beginPath(); x.arc(cx, cy, 8, 0, 7); x.fill();
    x.fillStyle = '#1e3a5f'; x.textAlign = 'center';
    x.font = 'bold 96px "Comic Neue KJ", "Comic Sans MS", sans-serif';
    x.fillText(String(valor), cx, cy + 100);
    x.font = 'bold 34px "Comic Neue KJ", "Comic Sans MS", sans-serif'; x.fillStyle = '#64748b';
    x.fillText('de ' + max + ' puntos', cx, cy + 140);
    x.font = 'bold 30px "Comic Neue KJ", "Comic Sans MS", sans-serif';
    x.fillText('0', cx - R + 4, cy + 44); x.fillText(String(max), cx + R - 4, cy + 44);
    return c;
  }
  function _kidsNube(doc, x, y, s) {
    doc.setFillColor(255, 255, 255);
    [[0, 0, 1], [-0.95, 0.3, 0.7], [0.95, 0.3, 0.72], [0.45, 0.45, 0.6], [-0.45, 0.45, 0.6]]
      .forEach(([dx, dy, r]) => doc.circle(x + dx * s, y + dy * s, r * s, 'F'));
  }
  function _kidsEstrella(doc, x, y, r, color) {
    const pts = [];
    for (let k = 0; k < 10; k++) {
      const rr = k % 2 ? r * 0.45 : r, a = -Math.PI / 2 + k * Math.PI / 5;
      pts.push([x + rr * Math.cos(a), y + rr * Math.sin(a)]);
    }
    doc.setFillColor(...color);
    doc.lines(pts.slice(1).map((p, i) => [p[0] - pts[i][0], p[1] - pts[i][1]]), pts[0][0], pts[0][1], [1, 1], 'F', true);
  }
  // Escalera de 4 escalones con ⭐ sobre el nivel alcanzado
  function _kidsEscalera(doc, x, y, w, h, n) {
    const sw = w / 4;
    for (let k = 1; k <= 4; k++) {
      const sh = h * (0.3 + 0.7 * k / 4);
      doc.setFillColor(...(k <= n ? KIDS_NIVELES[k - 1].color : [226, 232, 240]));
      doc.roundedRect(x + (k - 1) * sw + 0.4, y + h - sh, sw - 0.8, sh, 1, 1, 'F');
    }
    const sh = h * (0.3 + 0.7 * n / 4);
    doc.addImage(_kEmoji('⭐'), 'PNG', x + (n - 1) * sw + sw / 2 - 2.8, y + h - sh - 5.9, 5.6, 5.6, undefined, 'FAST');
  }

  function _renderPrimariaKids(doc, student, addPage, simLabel) {
    if (addPage) doc.addPage('letter', 'p');
    const PW = 215.9, M = 10, W = PW - 2 * M, F = _kidsFontOn(doc);
    const s = student;
    const name = String(s.name || '').toUpperCase();
    const grado = gradoDeCurso(s.curso);
    const general = Number(s.general) || 0;
    const areas = ['mat', 'lect', 'nat', 'soc', 'ing'].filter(k => _evaluada(s[k]));

    doc.setFillColor(...KIDS.papel); doc.rect(0, 0, PW, 279.4, 'F');
    [[18, 60, [255, 205, 210]], [200, 70, [200, 230, 201]], [12, 150, [255, 236, 179]], [205, 160, [187, 222, 251]],
     [15, 245, [225, 190, 231]], [203, 250, [255, 205, 210]], [110, 268, [187, 222, 251]]]
      .forEach(([cx, cy, col]) => { doc.setFillColor(...col); doc.circle(cx, cy, 1.6, 'F'); });

    // ── ENCABEZADO: cielo con nubes ──
    const hy = 8, hh = 38;
    doc.saveGraphicsState();
    doc.roundedRect(M, hy, W, hh, 6, 6, null); doc.clip(); doc.discardPath();
    for (let i = 0; i < 60; i++) {
      const t = i / 59, c = KIDS.cielo1.map((v, j) => Math.round(v + (KIDS.cielo2[j] - v) * t));
      doc.setFillColor(...c); doc.rect(M + W * i / 60, hy, W / 60 + 0.3, hh, 'F');
    }
    _kidsNube(doc, M + 112, hy + hh - 3, 5); _kidsNube(doc, M + 125, hy + 7, 3.6); _kidsNube(doc, M + 8, hy + hh + 1, 4.2);
    doc.restoreGraphicsState();
    _kidsEstrella(doc, M + 60, hy + 7, 2.2, KIDS.sol); _kidsEstrella(doc, M + 150, hy + 30, 1.6, KIDS.sol);
    _kidsEstrella(doc, M + 135, hy + 12, 1.2, [255, 255, 255]);
    const logo = _headerLogoForSim(simLabel);
    if (logo) { try { doc.addImage(logo, 'PNG', M + W - 42, hy - 1, 40, 40, undefined, 'FAST'); } catch (e) {} }
    doc.setFont(F, 'bold'); doc.setFontSize(27);
    doc.setTextColor(40, 60, 120); doc.text('¡Mis resultados!', M + 9.6, hy + 19.6);
    doc.setTextColor(255, 255, 255); doc.text('¡Mis resultados!', M + 9, hy + 19);
    doc.setFontSize(12); doc.text(`Simulacro ${simLabel || ''}`, M + 9.5, hy + 29);

    // ── TARJETA DEL ESTUDIANTE ──
    let y = hy + hh + 5;
    doc.setFillColor(255, 255, 255); doc.setDrawColor(...KIDS.cielo1); doc.setLineWidth(0.8);
    doc.setLineDashPattern([2, 1.4], 0); doc.roundedRect(M, y, W, 25, 5, 5, 'FD'); doc.setLineDashPattern([], 0);
    const ini = name.split(/\s+/).filter(Boolean).slice(0, 2).map(w => w[0]).join('') || '·';
    doc.setFillColor(255, 167, 38); doc.circle(M + 14, y + 12.5, 9, 'F');
    doc.setFillColor(255, 213, 79); doc.circle(M + 14, y + 12.5, 7.2, 'F');
    doc.setFont(F, 'bold'); doc.setFontSize(13); doc.setTextColor(...KIDS.tinta);
    doc.text(ini, M + 14, y + 14.6, {align: 'center'});
    doc.setFontSize(8.5); doc.setTextColor(...KIDS.suave); doc.text('ESTE ES EL REPORTE DE', M + 27, y + 7);
    let ns = 14.5; doc.setFontSize(ns); doc.setTextColor(...KIDS.tinta);
    while (doc.getTextWidth(name) > W - 30 && ns > 9) { ns -= 0.5; doc.setFontSize(ns); }
    doc.text(name, M + 27, y + 13.4);
    const chips = [];
    if (s.id) chips.push(['🪪', String(s.id), [227, 238, 255]]);
    if (s.curso) chips.push(['🏫', 'Curso ' + s.curso, [255, 238, 222]]);
    if (grado) chips.push(['🎒', `Grado ${grado}°`, [222, 246, 233]]);
    let cx = M + 27; doc.setFontSize(9);
    chips.forEach(([e, t, col]) => {
      const w = doc.getTextWidth(t) + 11;
      doc.setFillColor(...col); doc.roundedRect(cx, y + 16, w, 6.4, 3.2, 3.2, 'F');
      try { doc.addImage(_kEmoji(e), 'PNG', cx + 1.6, y + 16.8, 4.6, 4.6, undefined, 'FAST'); } catch (er) {}
      doc.setTextColor(...KIDS.tinta); doc.text(t, cx + 7.4, y + 20.6);
      cx += w + 3;
    });

    // ── MI PUNTAJE GENERAL + ¿QUÉ SIGNIFICA MI NIVEL? ──
    y += 31;
    const md = _kidsMedalla(general);
    const bw = 96, bh = areas.length <= 2 ? 74 : 64;
    doc.setFillColor(255, 255, 255); doc.setDrawColor(255, 213, 79); doc.setLineWidth(1.1);
    doc.roundedRect(M, y, bw, bh, 6, 6, 'FD');
    doc.setFont(F, 'bold'); doc.setFontSize(13); doc.setTextColor(...KIDS.tinta);
    doc.text('Mi puntaje general', M + bw / 2, y + 8, {align: 'center'});
    const gc = _kidsArcoiris(general, 500), gh = bh - 12, gw = Math.min(84, gh * gc.width / gc.height);
    doc.addImage(gc.toDataURL('image/png'), 'PNG', M + (bw - gw) / 2, y + 10, gw, gh, undefined, 'FAST');
    try { doc.addImage(_kEmoji(md.e), 'PNG', M + bw - 15, y + 2.5, 11, 11, undefined, 'FAST'); } catch (e) {}

    const nx = M + bw + 4, nw = W - bw - 4;
    doc.setFillColor(255, 255, 255); doc.setDrawColor(149, 117, 205); doc.roundedRect(nx, y, nw, bh, 6, 6, 'FD');
    doc.setFontSize(12.5); doc.setTextColor(...KIDS.tinta);
    doc.text('¿Qué significa mi nivel?', nx + nw / 2, y + 8, {align: 'center'});
    doc.setFont(F, 'normal'); doc.setFontSize(7.6); doc.setTextColor(...KIDS.suave);
    doc.text('Niveles de desempeño ICFES', nx + nw / 2, y + 12.4, {align: 'center'});
    const rh = (bh - 17) / 4;
    for (let k = 4; k >= 1; k--) {
      const ry = y + 15 + (4 - k) * rh, L = KIDS_NIVELES[k - 1];
      doc.setFillColor(...L.claro); doc.roundedRect(nx + 4, ry + 0.6, nw - 8, rh - 1.2, 3, 3, 'F');
      try { doc.addImage(_kEmoji(L.emoji), 'PNG', nx + 6, ry + rh / 2 - 3.6, 7.2, 7.2, undefined, 'FAST'); } catch (e) {}
      doc.setFont(F, 'bold'); doc.setFontSize(9.6); doc.setTextColor(...L.color.map(v => Math.max(0, v - 60)));
      doc.text(`Nivel ${k} · ${NIVEL_NOMBRES[k - 1]}`, nx + 15.5, ry + rh / 2 - 0.4);
      doc.setFont(F, 'normal'); doc.setFontSize(7.8); doc.setTextColor(...KIDS.tinta);
      doc.text(L.frase, nx + 15.5, ry + rh / 2 + 3.6);
    }
    y += bh + 4;
    doc.setFont(F, 'bold'); doc.setFontSize(11);
    const mw = doc.getTextWidth(md.msg) + 22;
    doc.setFillColor(225, 245, 238); doc.roundedRect(PW / 2 - mw / 2, y, mw, 9, 4.5, 4.5, 'F');
    try { doc.addImage(_kEmoji('🎉'), 'PNG', PW / 2 - mw / 2 + 3, y + 1.6, 5.8, 5.8, undefined, 'FAST'); } catch (e) {}
    doc.setTextColor(15, 110, 86); doc.text(md.msg, PW / 2 + 3, y + 6.2, {align: 'center'});

    // ── MIS ÁREAS ──
    y += 15;
    doc.setFont(F, 'bold'); doc.setFontSize(15); doc.setTextColor(...KIDS.tinta);
    try { doc.addImage(_kEmoji('🎒'), 'PNG', M, y - 5.6, 7, 7, undefined, 'FAST'); } catch (e) {}
    doc.text('Mis áreas', M + 9, y);
    y += 4;
    let best = null;
    areas.forEach(k => { if (best === null || Number(s[k]) > Number(s[best])) best = k; });
    if (best !== null && Number(s[best]) <= 0) best = null;
    const cw = (W - 5) / 2, ch = areas.length <= 2 ? 60 : 44;
    areas.forEach((k, i) => {
      const A = KIDS_AREAS[k], v = Number(s[k]) || 0, n = nivelIcfes(k, v) || 1, L = KIDS_NIVELES[n - 1];
      const x0 = M + (i % 2) * (cw + 5), y0 = y + Math.floor(i / 2) * (ch + 4);
      doc.setFillColor(255, 255, 255); doc.setDrawColor(...A.color); doc.setLineWidth(1);
      doc.roundedRect(x0, y0, cw, ch, 5, 5, 'FD');
      doc.setFillColor(...A.color); doc.roundedRect(x0, y0, cw, 11, 5, 5, 'F'); doc.rect(x0, y0 + 6, cw, 5, 'F');
      try { doc.addImage(_kEmoji(A.emoji), 'PNG', x0 + 3, y0 + 1.6, 7.8, 7.8, undefined, 'FAST'); } catch (e) {}
      doc.setFont(F, 'bold'); doc.setFontSize(12.5); doc.setTextColor(255, 255, 255);
      doc.text(A.nombre, x0 + 12.5, y0 + 7.6);
      if (k === best) {
        doc.setFillColor(...KIDS.sol); const tw = 34;
        doc.roundedRect(x0 + cw - tw - 3, y0 + 2.2, tw, 6.6, 3.3, 3.3, 'F');
        try { doc.addImage(_kEmoji('👑'), 'PNG', x0 + cw - tw - 1.6, y0 + 2.8, 5.4, 5.4, undefined, 'FAST'); } catch (e) {}
        doc.setFontSize(8); doc.setTextColor(120, 70, 0); doc.text('¡Mi mejor área!', x0 + cw - 3.6, y0 + 6.8, {align: 'right'});
      }
      const cy = y0 + 11 + (ch - 11) / 2 - (areas.length <= 2 ? 4 : 2);
      doc.setFillColor(...A.claro); doc.circle(x0 + 17, cy, 11.5, 'F');
      doc.setDrawColor(...A.color); doc.setLineWidth(1.2); doc.circle(x0 + 17, cy, 11.5, 'S');
      doc.setFont(F, 'bold'); doc.setFontSize(19); doc.setTextColor(...KIDS.tinta);
      doc.text(String(v), x0 + 17, cy + 2.6, {align: 'center'});
      doc.setFontSize(6.6); doc.setTextColor(...KIDS.suave); doc.text('de 100', x0 + 17, cy + 6.6, {align: 'center'});
      const tx = x0 + 33;
      doc.setFillColor(...L.claro); doc.roundedRect(tx, cy - 10.5, cw - 36 - 19, 10, 3, 3, 'F');
      try { doc.addImage(_kEmoji(L.emoji), 'PNG', tx + 1.6, cy - 9.4, 7.6, 7.6, undefined, 'FAST'); } catch (e) {}
      doc.setFont(F, 'bold'); doc.setFontSize(11); doc.setTextColor(...L.color.map(c => Math.max(0, c - 70)));
      doc.text(NIVEL_NOMBRES[n - 1], tx + 10.5, cy - 4.4);
      doc.setFont(F, 'normal'); doc.setFontSize(7.2); doc.setTextColor(...KIDS.suave);
      doc.text(`Nivel ${n} · ${rangoNivel(k, n)} puntos`, tx + 10.5, cy - 1.0);
      // la escalera (y su ⭐) no debe montarse sobre la cinta del título
      const ey = Math.max(cy - 13, y0 + 18);
      _kidsEscalera(doc, x0 + cw - 20, ey, 17, cy + 0.5 - ey, n);
      const by = cy + 4.5, bwid = cw - 36 - 3, fill = Math.max(4.2, bwid * Math.min(100, v) / 100);
      doc.setFillColor(238, 242, 247); doc.roundedRect(tx, by, bwid, 4.2, 2.1, 2.1, 'F');
      doc.setFillColor(...A.color); doc.roundedRect(tx, by, fill, 4.2, 2.1, 2.1, 'F');
      doc.setFillColor(255, 255, 255); doc.circle(tx + fill - 2.1, by + 2.1, 0.9, 'F');
      if (areas.length <= 2) {
        doc.setFont(F, 'normal'); doc.setFontSize(8.4); doc.setTextColor(...KIDS.tinta);
        doc.text(L.frase, x0 + cw / 2, y0 + ch - 5, {align: 'center'});
      }
    });
    y += Math.ceil(areas.length / 2) * (ch + 4);

    // ── CIERRE ──
    if (y + 18 < 268) {
      doc.setFillColor(255, 243, 224); doc.roundedRect(M, y + 1, W, 16, 6, 6, 'F');
      ['📚', '✏️', '🌈', '⭐'].forEach((e, i) => {
        try { doc.addImage(_kEmoji(e), 'PNG', M + 5 + i * 9, y + 5, 8, 8, undefined, 'FAST'); } catch (er) {}
      });
      doc.setFont(F, 'bold'); doc.setFontSize(12); doc.setTextColor(150, 80, 0);
      doc.text('¡Sigue leyendo, jugando y aprendiendo todos los días!', M + 44, y + 10.6);
    }
    doc.setFont(F, 'normal'); doc.setFontSize(7.5); doc.setTextColor(150, 150, 150);
    doc.text(`Generado automáticamente · Simulacro ${simLabel || ''}`, PW / 2, 272, {align: 'center'});
    doc.setFont('helvetica', 'normal');
  }

  function renderStudentPage(doc, student, pcts, addPage, simulacroLabel) {
    // Primaria (1° a 5°): reporte con estética infantil
    if (esPrimaria(student.curso)) return _renderPrimariaKids(doc, student, addPage, simulacroLabel);
    if (addPage) doc.addPage('letter','p');
    const pw = 215.9 - 20;
    const maxGeneral = 500;
    const subTitle = simulacroLabel || 'SIPAGRE';

    // ── HEADER ──
    const titleH = 32;
    const titleSteps = 100;
    for (let i = 0; i < titleSteps; i++) {
      const t = i / titleSteps;
      const r = Math.round(99  + (6   - 99 ) * t);
      const g = Math.round(102 + (182 - 102) * t);
      const b = Math.round(241 + (212 - 241) * t);
      doc.setFillColor(r, g, b);
      doc.rect(10 + (pw * i / titleSteps), 10, pw / titleSteps + 0.3, titleH, 'F');
    }
    doc.setFillColor(255,255,255);
    doc.rect(13, 14, 3, titleH - 8, 'F');
    const _headerLogo = _headerLogoForSim(subTitle);
    if (_headerLogo) {
      const logoSize = 46;
      const logoX = 10 + pw - logoSize + 2;
      const logoY = 10 + (titleH - logoSize) / 2;
      try { doc.addImage(_headerLogo, 'PNG', logoX, logoY, logoSize, logoSize, undefined, 'FAST'); }
      catch(e) {}
    }
    doc.setFont('helvetica','bold'); doc.setFontSize(15);
    doc.setTextColor(255,255,255);
    doc.text('REPORTE DE RESULTADOS', 20, 23);
    doc.setFont('helvetica','normal'); doc.setFontSize(10);
    doc.setTextColor(220, 235, 255);
    doc.text(`Simulacro ${subTitle}  ·  Reporte individual`, 20, 30);

    // ── DATOS ESTUDIANTE (tarjeta con avatar de iniciales y píldoras) ──
    let y = 47;
    const cardH = 24;
    doc.setFillColor(244, 244, 251);
    doc.setDrawColor(224, 222, 246);
    doc.setLineWidth(0.3);
    doc.roundedRect(10, y, pw, cardH, 3, 3, 'FD');

    const stuName = String(student.name || '').toUpperCase();
    const initials = (stuName.split(/\s+/).filter(Boolean).slice(0, 2)
                      .map(w => w[0]).join('')) || '·';
    const avR = 8, avX = 10 + 6 + avR, avY = y + cardH / 2;
    doc.setFillColor(99, 102, 241);
    doc.circle(avX, avY, avR, 'F');
    doc.setFont('helvetica', 'bold'); doc.setFontSize(11);
    doc.setTextColor(255, 255, 255);
    doc.text(initials, avX, avY + 1.6, {align: 'center'});

    const tx0 = avX + avR + 4.5;
    doc.setFontSize(6.5);
    doc.setTextColor(127, 119, 221);
    doc.text('ESTUDIANTE', tx0, y + 5.6);
    doc.setTextColor(15, 23, 42);
    let ns = 12.5;
    doc.setFontSize(ns);
    while (doc.getTextWidth(stuName) > pw - (tx0 - 10) - 6 && ns > 8) {
      ns -= 0.5; doc.setFontSize(ns);
    }
    doc.text(stuName, tx0, y + 11.4);

    const pills = [];
    if (student.id)    pills.push({icon: '🪪', text: String(student.id)});
    if (student.curso) pills.push({icon: '🏫', text: 'Curso ' + student.curso});
    const stuGrado = gradoDeCurso(student.curso);
    if (stuGrado)      pills.push({icon: '🎓', text: 'Grado ' + stuGrado + '°'});
    let pillX = tx0;
    doc.setFontSize(7.5);
    pills.forEach(p => {
      const icW = 3.6;
      const pillW = icW + doc.getTextWidth(p.text) + 7.5;
      doc.setFillColor(238, 237, 254);
      doc.roundedRect(pillX, y + 14.2, pillW, 6.2, 3.1, 3.1, 'F');
      const iu = _emojiDataUrl(p.icon);
      if (iu) { try { doc.addImage(iu, 'PNG', pillX + 2, y + 15.4, icW, icW, undefined, 'FAST'); } catch (e) {} }
      doc.setTextColor(60, 52, 137);
      doc.text(p.text, pillX + icW + 4.2, y + 18.4);
      pillX += pillW + 3;
    });

    y += cardH + 4;

    // ── RESULTADOS GLOBALES ──
    _drawSectionHeader(doc, 10, y, pw, 'RESULTADOS GLOBALES');
    y += 13;

    const general = Number(student.general)||0;
    const pctGen  = (pcts||{}).general||0;
    const primaria = esPrimaria(student.curso);
    // Primaria: solo las áreas que evalúa el grado (las demás llegan vacías)
    const subjects = primaria ? SUBJECTS.filter(s => _evaluada(student[s.key])) : SUBJECTS;
    const medal   = _medalFor(general);

    // Medidor un poco más compacto (110→92) para dar espacio al mensaje
    // motivacional sin desbordar la página.
    const gaugeCanvas = drawGaugeCanvas(general, maxGeneral, 'PUNTAJE GENERAL', medal.e);
    const gW = 92;
    const gH = gW * (gaugeCanvas.height/gaugeCanvas.width);
    doc.addImage(gaugeCanvas.toDataURL('image/png'), 'PNG', 10, y, gW, gH, undefined, 'FAST');

    if (primaria) {
      _drawNivelesCard(doc, 10 + gW + 8, y + 2, pw - gW - 8, gH - 4);
    } else {
      const donutCanvas = drawDonutCanvas(pctGen, 'PERCENTIL', '#6366f1');
      const dSize = Math.min(gH, pw - gW - 8);
      const dx = 10 + gW + 8 + (pw - gW - 8 - dSize) / 2;
      doc.addImage(donutCanvas.toDataURL('image/png'), 'PNG', dx, y + (gH - dSize)/2, dSize, dSize, undefined, 'FAST');

      doc.setFont('helvetica','normal'); doc.setFontSize(7.5);
      doc.setTextColor(100, 116, 139);
      doc.text('Posicion respecto a los evaluados', dx + dSize/2, y + (gH + dSize)/2 + 2.5, {align:'center'});
    }

    y += gH + 3;

    // Mensaje motivacional (píldora verde centrada)
    doc.setFont('helvetica','bold'); doc.setFontSize(8.5);
    const msgW = doc.getTextWidth(medal.msg) + 12;
    doc.setFillColor(225, 245, 238);
    doc.roundedRect(10 + pw/2 - msgW/2, y, msgW, 7, 3.5, 3.5, 'F');
    doc.setTextColor(15, 110, 86);
    doc.text(medal.msg, 10 + pw/2, y + 4.7, {align:'center'});
    y += 10;

    // ── RESULTADOS POR PRUEBA ──
    _drawSectionHeader(doc, 10, y, pw, 'RESULTADOS POR PRUEBA');
    if (!primaria) {
      // Leyenda del percentil (dentro de la franja, a la derecha)
      doc.setFont('helvetica','normal'); doc.setFontSize(6.2);
      doc.setTextColor(232, 236, 255);
      const lgT1 = 'Percentil: 0 ', lgT2 = ' 100';
      const lgW = doc.getTextWidth(lgT1) + 2.6 + doc.getTextWidth(lgT2);
      let lgX = 10 + pw - 4 - lgW;
      const lgY = y + 6;
      doc.text(lgT1, lgX, lgY);
      lgX += doc.getTextWidth(lgT1);
      doc.setFillColor(232, 236, 255);
      doc.triangle(lgX, lgY - 2.6, lgX + 2.6, lgY - 2.6, lgX + 1.3, lgY - 0.3, 'F');
      doc.text(lgT2, lgX + 2.6, lgY);
    }
    y += 13;

    // Vértices del radar: iconos de asignatura (bandera dibujada para Inglés)
    // El radar necesita al menos 3 áreas (3° y 4° de primaria tienen 2)
    const conRadar = subjects.length >= 3;
    const rW = conRadar ? 75 : 0, rH = 75;
    if (conRadar) {
      const labels = subjects.map(s => SUBJECT_ICONS[s.key] || '');
      const values = subjects.map(s => Number(student[s.key])||0);
      const radarCanvas = drawRadarCanvas(labels, values, 100);
      doc.addImage(radarCanvas.toDataURL('image/png'), 'PNG', 10, y + 2, rW, rH, undefined, 'FAST');
    }

    const lx = conRadar ? 10 + rW + 5 : 10;
    const lw = conRadar ? pw - rW - 7 : pw;
    let ly = y;
    const rowH = Math.min(18, (rH + 2) / subjects.length);
    const colNameW   = lw * 0.30;
    const colScoreW  = lw * 0.16;
    const colBarW    = lw - colNameW - colScoreW - 3;

    doc.setFont('helvetica','bold'); doc.setFontSize(7);
    doc.setTextColor(99,102,241);
    doc.text('PRUEBA',     lx + 1,                                  ly - 1);
    doc.text('PUNTAJE',    lx + colNameW + colScoreW/2,             ly - 1, {align:'center'});
    doc.text(primaria ? 'NIVEL DE DESEMPEÑO ICFES' : 'PERCENTIL',
             lx + colNameW + colScoreW + 2 + colBarW/2, ly - 1, {align:'center'});

    const shortNames = {
      'Matematicas':           'Matemáticas',
      'Lectura Critica':       primaria ? 'Lenguaje' : 'Lectura Crítica',
      'Sociales y Ciudadanas': 'Sociales',
      'Ciencias Naturales':    'Naturales',
      'Ingles':                'Inglés',
    };

    // "Tu mejor prueba": la asignatura con mayor puntaje se resalta con ⭐
    let bestIdx = -1, bestScore = -1;
    subjects.forEach((s, i) => {
      const v = Number(student[s.key]) || 0;
      if (v > bestScore) { bestScore = v; bestIdx = i; }
    });
    if (bestScore <= 0) bestIdx = -1;

    subjects.forEach((subj, idx) => {
      const score = Number(student[subj.key])||0;
      const pct   = (pcts||{})[subj.key]||0;
      const displayName = shortNames[subj.name] || subj.name;
      const isBest = idx === bestIdx;

      if (isBest) {
        doc.setFillColor(250, 238, 218);   // dorado suave
      } else {
        doc.setFillColor(idx % 2 === 0 ? 248 : 241,
                         idx % 2 === 0 ? 250 : 245,
                         idx % 2 === 0 ? 252 : 249);
      }
      doc.roundedRect(lx, ly + 0.5, lw, rowH - 1, 1.5, 1.5, 'F');

      // Icono de la asignatura (la bandera es más ancha que alta)
      const isFlag = SUBJECT_ICONS[subj.key] === '__FLAG__';
      const icH = 4.6, icW = isFlag ? icH * 1.6 : icH;
      const iconUrl = _subjectIconUrl(subj.key);
      if (iconUrl) {
        try { doc.addImage(iconUrl, 'PNG', lx + 1.8, ly + rowH/2 - icH/2, icW, icH, undefined, 'FAST'); }
        catch (e) {}
      }
      const nameX = lx + 1.8 + icW + 1.8;

      doc.setFont('helvetica','bold'); doc.setFontSize(8.5);
      doc.setTextColor(30, 41, 59);
      const maxNameW = colNameW - (nameX - lx) - 5;
      let nameSize = 8.5;
      while (doc.getTextWidth(displayName) > maxNameW && nameSize > 6) {
        nameSize -= 0.5;
        doc.setFontSize(nameSize);
      }
      const nameY = isBest ? ly + rowH/2 - 0.6 : ly + rowH/2 + 1.2;
      doc.text(displayName, nameX, nameY);
      if (isBest) {
        const starUrl = _emojiDataUrl('⭐');
        if (starUrl) {
          const sx = Math.min(nameX + doc.getTextWidth(displayName) + 1.2,
                              lx + colNameW - 4.5);
          try { doc.addImage(starUrl, 'PNG', sx, nameY - 3.2, 3.8, 3.8, undefined, 'FAST'); }
          catch (e) {}
        }
        doc.setFont('helvetica','normal'); doc.setFontSize(5.8);
        doc.setTextColor(133, 79, 11);
        doc.text('tu mejor prueba', nameX, nameY + 3.6);
      }

      const badgeW = colScoreW - 1, badgeH = rowH - 3;
      const badgeX = lx + colNameW + 0.5;
      _drawScoreBadge(doc, badgeX, ly + 1.5, badgeW, badgeH, score, 'PUNTAJE');

      if (primaria) {
        _drawNivelCell(doc, lx + colNameW + colScoreW + 2, ly + 0.8, colBarW - 1, rowH - 1.6,
                       subj.key, score);
      } else {
        const barCanvas = drawPercentileBarCanvas(pct);
        const barW = colBarW;
        const barH = barW * (barCanvas.height/barCanvas.width);
        const barX = lx + colNameW + colScoreW + 2;
        const barY = ly + (rowH - barH)/2;
        doc.addImage(barCanvas.toDataURL('image/png'), 'PNG', barX, barY, barW, barH, undefined, 'FAST');
      }

      ly += rowH;
    });

    if (_logoDataUrl) {
      const fLogoSize = 12;
      const fLogoX = 10 + pw/2 - fLogoSize/2;
      const fLogoY = 256;
      try { doc.addImage(_logoDataUrl, 'PNG', fLogoX, fLogoY, fLogoSize, fLogoSize, undefined, 'FAST'); }
      catch(e) {}
    }
    doc.setFont('helvetica','italic'); doc.setFontSize(7);
    doc.setTextColor(150,150,150);
    doc.text(`Generado automaticamente - Simulacro ${subTitle}`, 10+pw/2, 272, {align:'center'});
  }

  function getJsPDF() {
    if (window.jspdf && window.jspdf.jsPDF) return window.jspdf.jsPDF;
    throw new Error('jsPDF no ha cargado. Recarga la pagina e intenta de nuevo.');
  }

  // ── Export al namespace global ──
  window.JMRPdf = {
    renderStudentPage,
    calcPercentiles,
    _loadLogo,
    getJsPDF,
    SUBJECTS,
    gradoDeCurso,
    // Configuración desde la app que use el módulo
    setLogoChoice,        // 'auto' | 'sipagre' | 'ietagro'
    resolveLogoChoice,    // (simulacroLabel) → 'sipagre' | 'ietagro'
    setGradoResolver,     // (fn) — la admin registra su gradoDeCurso con catálogo
    // Logos cargados (para encabezados/pies de otros reportes)
    headerLogo: _headerLogoForSim,   // (simulacroLabel) → dataURL
    footerLogo: () => _logoDataUrl,  // logo JMR pequeño del pie
  };
})(window);
