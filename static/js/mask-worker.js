/*
  Mask rendering worker using OffscreenCanvas.
  Receives messages:
    - {type:'init', canvas: OffscreenCanvas}
    - {type:'resize', size:{w,h}}
    - {type:'overlay', masks:[ [ [nx,ny], ... ], ... ], ids:[...], rect:{x,y,w,h}}
*/

let canvas = null;
let ctx = null;
let size = { w: 0, h: 0 };

function colorForInstance(instId) {
  const hue = (Number(instId) * 57) % 360;
  return `hsla(${hue}, 65%, 70%, 0.22)`;
}

function drawOverlay(masks, ids, rect, popupCounters) {
  if (!ctx) return;
  const cw = size.w || (canvas && canvas.width) || 0;
  const ch = size.h || (canvas && canvas.height) || 0;
  ctx.clearRect(0, 0, cw, ch);
  if (!masks || !masks.length) return;
  const r = rect || { x: 0, y: 0, w: cw, h: ch };

  for (let idx = 0; idx < masks.length; idx++) {
    const poly = masks[idx];
    if (!Array.isArray(poly) || poly.length < 3) continue;
    const instId = (ids && ids[idx]) ? ids[idx] : (idx + 1);
    const fill = colorForInstance(instId);
    ctx.beginPath();
    const plen = poly.length;
    const step = Math.max(1, Math.floor(plen / 200));
    for (let i = 0; i < plen; i += step) {
      const nx = poly[i][0], ny = poly[i][1];
      const x = r.x + nx * r.w;
      const y = r.y + ny * r.h;
      if (i === 0) ctx.moveTo(x, y); else ctx.lineTo(x, y);
    }
    ctx.closePath();
    ctx.save();
    ctx.fillStyle = fill;
    ctx.globalAlpha = 1.0;
    ctx.fill();
    ctx.restore();

    // Badge
    let minx = Infinity, miny = Infinity;
    for (let i = 0; i < plen; i += step) {
      const nx = poly[i][0], ny = poly[i][1];
      const x = r.x + nx * r.w;
      const y = r.y + ny * r.h;
      if (x < minx) minx = x;
      if (y < miny) miny = y;
    }
    const label = String(instId);
    const padX = 6, radius = 6;
    ctx.font = '600 13px system-ui, Segoe UI, Arial';
    const tw = ctx.measureText(label).width;
    const bw = Math.ceil(tw + padX * 2);
    const bh = 20;
    let bx = Math.max(r.x, Math.min(minx - 8, r.x + r.w - bw));
    let by = Math.max(r.y, Math.min(miny - 8, r.y + r.h - bh));
    ctx.save();
    ctx.beginPath();
    ctx.moveTo(bx + radius, by);
    ctx.lineTo(bx + bw - radius, by);
    ctx.quadraticCurveTo(bx + bw, by, bx + bw, by + radius);
    ctx.lineTo(bx + bw, by + bh - radius);
    ctx.quadraticCurveTo(bx + bw, by + bh, bx + bw - radius, by + bh);
    ctx.lineTo(bx + radius, by + bh);
    ctx.quadraticCurveTo(bx, by + bh, bx, by + bh - radius);
    ctx.lineTo(bx, by + radius);
    ctx.quadraticCurveTo(bx, by, bx + radius, by);
    ctx.fillStyle = 'rgba(30,50,80,0.85)';
    ctx.fill();
    ctx.fillStyle = '#fff';
    ctx.textAlign = 'center';
    ctx.textBaseline = 'middle';
    ctx.fillText(label, bx + bw / 2, by + bh / 2);
    ctx.restore();
  }
  
  // Отрисовка всплывающих счетчиков
  if (popupCounters && popupCounters.length > 0) {
    const now = performance.now();
    ctx.save();
    popupCounters.forEach(popup => {
      const elapsed = now - popup.startTime;
      const progress = Math.min(1, elapsed / popup.duration);
      const alpha = 1 - progress;
      const ease = progress * (2 - progress);
      
      const x = popup.x * r.w;
      const y = popup.y * r.h - ease * popup.riseDistance;
      const radius = 16;
      
      ctx.globalAlpha = Math.max(0, alpha);
      
      // Круг фона
      ctx.beginPath();
      ctx.arc(x, y, radius, 0, Math.PI * 2);
      ctx.fillStyle = popup.color;
      ctx.fill();
      
      // Обводка
      ctx.lineWidth = 2;
      ctx.strokeStyle = 'rgba(0,0,0,0.3)';
      ctx.stroke();
      
      // Текст
      ctx.font = '700 14px system-ui, Segoe UI, Arial';
      ctx.textAlign = 'center';
      ctx.textBaseline = 'middle';
      ctx.fillStyle = '#fff';
      ctx.fillText(popup.text, x, y + 1);
    });
    ctx.restore();
  }
}

onmessage = (e) => {
  const data = e.data || {};
  if (data.type === 'init') {
    canvas = data.canvas;
    ctx = canvas.getContext('2d');
  } else if (data.type === 'resize') {
    const w = Math.max(1, Math.floor(data.size && data.size.w ? data.size.w : 0));
    const h = Math.max(1, Math.floor(data.size && data.size.h ? data.size.h : 0));
    if (canvas) { canvas.width = w; canvas.height = h; }
    size = { w, h };
  } else if (data.type === 'overlay') {
    drawOverlay(data.masks, data.ids, data.rect, data.popupCounters);
  }
};

