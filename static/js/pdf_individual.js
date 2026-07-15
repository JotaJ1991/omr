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
    const c = String(curso || '').trim();
    if (!c) return '';
    if (_gradoResolver) {
      try {
        const g = _gradoResolver(c);
        if (g) return String(g);
      } catch (e) { /* caer al default */ }
    }
    if (/^\d{4}/.test(c)) return c.slice(0, 2);   // 1101 → 11
    if (/^\d{3}/.test(c)) return c.slice(0, 1);   // 601  → 6
    const m = c.match(/^(\d{1,2})/);              // 11A  → 11
    return m ? m[1] : '';
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
      msg: 'Con esfuerzo, disciplina, perseverancia y constancia puedes ser potencial BECARIO' };
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

  function renderStudentPage(doc, student, pcts, addPage, simulacroLabel) {
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
    const medal   = _medalFor(general);

    // Medidor un poco más compacto (110→92) para dar espacio al mensaje
    // motivacional sin desbordar la página.
    const gaugeCanvas = drawGaugeCanvas(general, maxGeneral, 'PUNTAJE GENERAL', medal.e);
    const gW = 92;
    const gH = gW * (gaugeCanvas.height/gaugeCanvas.width);
    doc.addImage(gaugeCanvas.toDataURL('image/png'), 'PNG', 10, y, gW, gH, undefined, 'FAST');

    const donutCanvas = drawDonutCanvas(pctGen, 'PERCENTIL', '#6366f1');
    const dSize = Math.min(gH, pw - gW - 8);
    const dx = 10 + gW + 8 + (pw - gW - 8 - dSize) / 2;
    doc.addImage(donutCanvas.toDataURL('image/png'), 'PNG', dx, y + (gH - dSize)/2, dSize, dSize, undefined, 'FAST');

    doc.setFont('helvetica','normal'); doc.setFontSize(7.5);
    doc.setTextColor(100, 116, 139);
    doc.text('Posicion respecto a los evaluados', dx + dSize/2, y + (gH + dSize)/2 + 2.5, {align:'center'});

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
    y += 13;

    // Vértices del radar: iconos de asignatura (bandera dibujada para Inglés)
    const labels = SUBJECTS.map(s => SUBJECT_ICONS[s.key] || '');
    const values = SUBJECTS.map(s => Number(student[s.key])||0);
    const radarCanvas = drawRadarCanvas(labels, values, 100);
    const rW = 75, rH = 75;
    doc.addImage(radarCanvas.toDataURL('image/png'), 'PNG', 10, y + 2, rW, rH, undefined, 'FAST');

    const lx = 10 + rW + 5;
    const lw = pw - rW - 7;
    let ly = y;
    const rowH = (rH + 2) / SUBJECTS.length;
    const colNameW   = lw * 0.30;
    const colScoreW  = lw * 0.16;
    const colBarW    = lw - colNameW - colScoreW - 3;

    doc.setFont('helvetica','bold'); doc.setFontSize(7);
    doc.setTextColor(99,102,241);
    doc.text('PRUEBA',     lx + 1,                                  ly - 1);
    doc.text('PUNTAJE',    lx + colNameW + colScoreW/2,             ly - 1, {align:'center'});
    doc.text('PERCENTIL',  lx + colNameW + colScoreW + 2 + colBarW/2, ly - 1, {align:'center'});

    const shortNames = {
      'Matematicas':           'Matemáticas',
      'Lectura Critica':       'Lectura Crítica',
      'Sociales y Ciudadanas': 'Sociales',
      'Ciencias Naturales':    'Naturales',
      'Ingles':                'Inglés',
    };

    // "Tu mejor prueba": la asignatura con mayor puntaje se resalta con ⭐
    let bestIdx = -1, bestScore = -1;
    SUBJECTS.forEach((s, i) => {
      const v = Number(student[s.key]) || 0;
      if (v > bestScore) { bestScore = v; bestIdx = i; }
    });
    if (bestScore <= 0) bestIdx = -1;

    SUBJECTS.forEach((subj, idx) => {
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

      const barCanvas = drawPercentileBarCanvas(pct);
      const barW = colBarW;
      const barH = barW * (barCanvas.height/barCanvas.width);
      const barX = lx + colNameW + colScoreW + 2;
      const barY = ly + (rowH - barH)/2;
      doc.addImage(barCanvas.toDataURL('image/png'), 'PNG', barX, barY, barW, barH, undefined, 'FAST');

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
