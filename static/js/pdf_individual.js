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
  function gradoDeCurso(curso) {
    const c = String(curso || '').trim();
    if (!c) return '';
    const m = c.match(/^(\d{1,2})/);
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
    if (_logoDataUrl === null)        _logoDataUrl        = await _loadImageAsDataUrl('/static/logo.png');
    if (_sipagreLogoDataUrl === null) _sipagreLogoDataUrl = await _loadImageAsDataUrl('/static/sipagre_logo.png');
    return _logoDataUrl;
  }

  function drawGaugeCanvas(value, max, label) {
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
    const W = 700, H = 110;
    const c = _newCanvas(W, H);
    const ctx = c.getContext('2d');
    const x = 30, y = 60, w = W - 60, h = 16;
    const grad = ctx.createLinearGradient(x, 0, x + w, 0);
    grad.addColorStop(0,    '#ef4444');
    grad.addColorStop(0.4,  '#f59e0b');
    grad.addColorStop(0.7,  '#eab308');
    grad.addColorStop(1,    '#22c55e');
    ctx.fillStyle = grad;
    ctx.beginPath();
    if (ctx.roundRect) ctx.roundRect(x, y, w, h, 8);
    else ctx.rect(x, y, w, h);
    ctx.fill();
    ctx.strokeStyle = 'rgba(0,0,0,0.2)';
    ctx.lineWidth = 1;
    for (let i = 0; i <= 10; i++) {
      const tx = x + (w * i/10);
      ctx.beginPath();
      ctx.moveTo(tx, y);
      ctx.lineTo(tx, y + h);
      ctx.stroke();
    }
    ctx.fillStyle = '#64748b';
    ctx.font = 'bold 18px Inter, Arial';
    ctx.textAlign = 'left';
    ctx.fillText('0', x - 22, y + h + 20);
    ctx.textAlign = 'right';
    ctx.fillText('100', x + w + 26, y + h + 20);
    const px = x + (w * Math.max(0, Math.min(100, percent)) / 100);
    ctx.fillStyle = '#1e293b';
    ctx.beginPath();
    ctx.moveTo(px - 9, y - 6);
    ctx.lineTo(px + 9, y - 6);
    ctx.lineTo(px, y + 6);
    ctx.closePath();
    ctx.fill();
    ctx.fillStyle = '#0f172a';
    ctx.font = 'bold 42px Inter, Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'alphabetic';
    ctx.fillText(String(percent), px, y - 14);
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
    ctx.fillStyle = '#1e293b';
    ctx.font = 'bold 22px Inter, Arial';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    for (let i = 0; i < N; i++) {
      const a = -Math.PI/2 + 2*Math.PI*i/N;
      const lblR = R + 50;
      const x = cx + lblR*Math.cos(a);
      const y = cy + lblR*Math.sin(a);
      ctx.fillText(labels[i], x, y);
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
    if (_sipagreLogoDataUrl) {
      const logoSize = 46;
      const logoX = 10 + pw - logoSize + 2;
      const logoY = 10 + (titleH - logoSize) / 2;
      try { doc.addImage(_sipagreLogoDataUrl, 'PNG', logoX, logoY, logoSize, logoSize); }
      catch(e) {}
    }
    doc.setFont('helvetica','bold'); doc.setFontSize(15);
    doc.setTextColor(255,255,255);
    doc.text('REPORTE DE RESULTADOS', 20, 23);
    doc.setFont('helvetica','normal'); doc.setFontSize(10);
    doc.setTextColor(220, 235, 255);
    doc.text(`Simulacro ${subTitle}  ·  Reporte individual`, 20, 30);

    // ── DATOS ESTUDIANTE ──
    let y = 47;
    const cardH = 22;
    doc.setFillColor(245, 247, 252);
    doc.roundedRect(10, y, pw, cardH, 2.5, 2.5, 'F');
    doc.setFillColor(99, 102, 241);
    doc.roundedRect(10, y, 2.2, cardH, 1.1, 1.1, 'F');
    doc.setFont('helvetica','bold'); doc.setFontSize(7);
    doc.setTextColor(99, 102, 241);
    doc.text('ESTUDIANTE', 16, y + 5);
    doc.setFont('helvetica','bold'); doc.setFontSize(13);
    doc.setTextColor(15, 23, 42);
    doc.text(String(student.name||'').toUpperCase(), 16, y + 11);
    doc.setFont('helvetica','bold'); doc.setFontSize(8);
    const idText = `ID:  ${String(student.id||'')}`;
    const idW = doc.getTextWidth(idText) + 6;
    doc.setFillColor(99, 102, 241);
    doc.roundedRect(16, y + 14, idW, 5.5, 1.2, 1.2, 'F');
    doc.setTextColor(255, 255, 255);
    doc.text(idText, 16 + idW/2, y + 17.8, {align:'center'});

    if (student.curso) {
      const cText = `CURSO:  ${String(student.curso)}`;
      const cW = doc.getTextWidth(cText) + 6;
      const cX = 16 + idW + 3;
      doc.setFillColor(6, 182, 212);
      doc.roundedRect(cX, y + 14, cW, 5.5, 1.2, 1.2, 'F');
      doc.setTextColor(255, 255, 255);
      doc.text(cText, cX + cW/2, y + 17.8, {align:'center'});
    }

    y += cardH + 4;

    // ── RESULTADOS GLOBALES ──
    _drawSectionHeader(doc, 10, y, pw, 'RESULTADOS GLOBALES');
    y += 13;

    const general = Number(student.general)||0;
    const pctGen  = (pcts||{}).general||0;

    const gaugeCanvas = drawGaugeCanvas(general, maxGeneral, 'PUNTAJE GENERAL');
    const gW = 110;
    const gH = gW * (gaugeCanvas.height/gaugeCanvas.width);
    doc.addImage(gaugeCanvas.toDataURL('image/png'), 'PNG', 10, y, gW, gH);

    const donutCanvas = drawDonutCanvas(pctGen, 'PERCENTIL', '#6366f1');
    const dSize = Math.min(gH, pw - gW - 8);
    const dx = 10 + gW + 8 + (pw - gW - 8 - dSize) / 2;
    doc.addImage(donutCanvas.toDataURL('image/png'), 'PNG', dx, y + (gH - dSize)/2, dSize, dSize);

    doc.setFont('helvetica','normal'); doc.setFontSize(7.5);
    doc.setTextColor(100, 116, 139);
    doc.text('Posicion respecto a los evaluados', dx + dSize/2, y + (gH + dSize)/2 + 2.5, {align:'center'});

    y += gH + 4;

    // ── RESULTADOS POR PRUEBA ──
    _drawSectionHeader(doc, 10, y, pw, 'RESULTADOS POR PRUEBA');
    y += 13;

    const labels = SUBJECTS.map(s => s.name
      .replace('Sociales y Ciudadanas','Sociales')
      .replace('Ciencias Naturales','Naturales')
      .replace('Lectura Critica','Lect. Crítica')
      .replace('Matematicas','Matem.'));
    const values = SUBJECTS.map(s => Number(student[s.key])||0);
    const radarCanvas = drawRadarCanvas(labels, values, 100);
    const rW = 75, rH = 75;
    doc.addImage(radarCanvas.toDataURL('image/png'), 'PNG', 10, y + 2, rW, rH);

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

    SUBJECTS.forEach((subj, idx) => {
      const score = Number(student[subj.key])||0;
      const pct   = (pcts||{})[subj.key]||0;
      const displayName = shortNames[subj.name] || subj.name;

      doc.setFillColor(idx % 2 === 0 ? 248 : 241,
                       idx % 2 === 0 ? 250 : 245,
                       idx % 2 === 0 ? 252 : 249);
      doc.roundedRect(lx, ly + 0.5, lw, rowH - 1, 1.5, 1.5, 'F');

      doc.setFont('helvetica','bold'); doc.setFontSize(8.5);
      doc.setTextColor(30, 41, 59);
      const maxNameW = colNameW - 4;
      let nameSize = 8.5;
      while (doc.getTextWidth(displayName) > maxNameW && nameSize > 6.5) {
        nameSize -= 0.5;
        doc.setFontSize(nameSize);
      }
      doc.text(displayName, lx + 2, ly + rowH/2 + 1.2);

      const badgeW = colScoreW - 1, badgeH = rowH - 3;
      const badgeX = lx + colNameW + 0.5;
      _drawScoreBadge(doc, badgeX, ly + 1.5, badgeW, badgeH, score, 'PUNTAJE');

      const barCanvas = drawPercentileBarCanvas(pct);
      const barW = colBarW;
      const barH = barW * (barCanvas.height/barCanvas.width);
      const barX = lx + colNameW + colScoreW + 2;
      const barY = ly + (rowH - barH)/2;
      doc.addImage(barCanvas.toDataURL('image/png'), 'PNG', barX, barY, barW, barH);

      ly += rowH;
    });

    if (_logoDataUrl) {
      const fLogoSize = 12;
      const fLogoX = 10 + pw/2 - fLogoSize/2;
      const fLogoY = 256;
      try { doc.addImage(_logoDataUrl, 'PNG', fLogoX, fLogoY, fLogoSize, fLogoSize); }
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
  };
})(window);
