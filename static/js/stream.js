// Конфигурация камер: серверный режим получает кадры по WebSocket (тип frame_jpeg)
window.streamConfig = {
  cameras: [
    { id: 'cam1', name: 'Камера 1' }
  ]
};

let activeStream = null;
let ws = null;

function wsUrl() {
  const proto = (location.protocol === 'https:') ? 'wss' : 'ws';
  return `${proto}://${location.host}/ws/count`;
}

function ensureImgEl() {
  return document.getElementById('videoStream') || document.getElementById('videoElement');
}

async function startStream(cameraId) {
  const camera = window.streamConfig.cameras.find(cam => cam.id === cameraId) || { id: cameraId || 'cam1' };
  try {
    if (activeStream) await stopStream(activeStream);
    const img = ensureImgEl();
    // Подключаем WS и сразу запрашиваем один кадр
    ws = new WebSocket(wsUrl());
    ws.onopen = () => {
      try { ws.send(JSON.stringify({ action: 'frame' })); } catch(_) {}
      const statusEl = document.getElementById('streamStatus');
      if (statusEl){ statusEl.textContent = 'WS connected'; statusEl.style.color = '#2c5'; setTimeout(()=>{ statusEl.textContent=''; statusEl.style.color=''; }, 1000); }
    };
    ws.onclose = () => { ws = null; };
    ws.onerror = () => {
      const statusEl = document.getElementById('streamStatus');
      if (statusEl){ statusEl.textContent = 'WS error'; statusEl.style.color = '#a66'; setTimeout(()=>{ statusEl.textContent=''; statusEl.style.color=''; }, 1200); }
    };
    ws.onmessage = (ev) => {
      try {
        const msg = JSON.parse(ev.data);
        if (msg.type === 'frame_jpeg' && msg.data_b64 && img) {
          img.src = `data:${msg.content_type||'image/jpeg'};base64,${msg.data_b64}`;
        } else if (msg.type === 'count_update') {
          const fpsInfo = document.getElementById('fpsInfo');
          if (fpsInfo) fpsInfo.textContent = `Server FPS~${Number(msg.fps||0).toFixed(1)}`;
        }
      } catch(_){}
    };
    activeStream = camera.id;
    return { ok: true };
  } catch (error) {
    console.warn('Ошибка при запуске стрима (WS):', error);
    const statusEl = document.getElementById('modelStatus') || document.getElementById('streamStatus');
    if (statusEl) {
      const old = statusEl.textContent;
      statusEl.textContent = 'Ошибка при запуске стрима';
      statusEl.style.color = '#e14a4a';
      setTimeout(() => { statusEl.textContent = old || ''; statusEl.style.color = ''; }, 5000);
    }
    return { ok: false };
  }
}

async function stopStream(cameraId) {
  try {
    if (ws) { try { ws.close(); } catch(_) {} ws = null; }
    const img = ensureImgEl();
    if (img) { img.dataset.prev = img.src; img.removeAttribute('src'); }
    activeStream = null;
    return { ok: true };
  } catch (error) {
    console.error('Error stopping stream:', error);
    return { ok: false };
  }
}

document.addEventListener('DOMContentLoaded', () => {
  if (window.streamConfig.cameras.length > 0) {
    startStream(window.streamConfig.cameras[0].id);
  }
});

window.addEventListener('beforeunload', () => {
  if (activeStream) {
    stopStream(activeStream);
  }
});
