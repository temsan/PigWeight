// Конфигурация камер
window.streamConfig = {
  cameras: [
    {
      id: 'cam1',
      name: 'Камера 1',
      rtspUrl: 'rtsp://admin:Qwerty.123@10.15.6.24/1/1'  // Замените на реальные данные камеры
    }
  ],
  apiUrl: 'http://localhost:8000'
};

let activeStream = null;
let hlsPlayer = null;

async function startStream(cameraId) {
  const camera = window.streamConfig.cameras.find(cam => cam.id === cameraId);
  if (!camera) return;

  try {
    // Останавливаем предыдущий стрим если есть
    if (activeStream) {
      await stopStream(activeStream);
    }

    // Запрашиваем новый стрим
    const response = await fetch(`${window.streamConfig.apiUrl}/api/stream/start`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      },
      body: JSON.stringify({
        camera_id: camera.id,
        rtsp_url: camera.rtspUrl
      })
    });

    const data = await response.json();
    if (!data.stream_url) {
      throw new Error('Failed to start stream');
    }

    // Инициализируем HLS плеер
    const video = document.getElementById('videoElement');
    if (Hls.isSupported()) {
      hlsPlayer = new Hls({
        liveDurationInfinity: true,
        enableWorker: true,
        lowLatencyMode: true
      });
      hlsPlayer.loadSource(`http://localhost:5000${data.stream_url}`);
      hlsPlayer.attachMedia(video);
      hlsPlayer.on(Hls.Events.MANIFEST_PARSED, () => {
        video.play();
      });
    } else if (video.canPlayType('application/vnd.apple.mpegurl')) {
      // Для Safari, который поддерживает HLS нативно
      video.src = `http://localhost:5000${data.stream_url}`;
      video.addEventListener('loadedmetadata', () => {
        video.play();
      });
    }

    activeStream = camera.id;
  } catch (error) {
    console.warn('Ошибка при запуске стрима:', error);
    // Тихий режим: без alert, можно обновить статус-элемент на странице, если он есть
    const statusEl = document.getElementById('modelStatus') || document.getElementById('streamStatus');
    if (statusEl) {
      const old = statusEl.textContent;
      statusEl.textContent = 'Ошибка при запуске стрима';
      statusEl.style.color = '#e14a4a';
      setTimeout(() => {
        statusEl.textContent = old || '';
        statusEl.style.color = '';
      }, 5000);
    }
  }
}

async function stopStream(cameraId) {
  if (!cameraId) return;

  try {
    // Останавливаем HLS плеер
    if (hlsPlayer) {
      hlsPlayer.destroy();
      hlsPlayer = null;
    }

    // Останавливаем стрим на сервере
    await fetch(`${window.streamConfig.apiUrl}/api/stream/stop/${cameraId}`, {
      method: 'POST',
      headers: {
        'Content-Type': 'application/json'
      }
    });

    activeStream = null;
  } catch (error) {
    console.error('Error stopping stream:', error);
  }
}

// Инициализация при загрузке страницы
document.addEventListener('DOMContentLoaded', () => {
  // Автоматически запускаем первую камеру
  if (window.streamConfig.cameras.length > 0) {
    startStream(window.streamConfig.cameras[0].id);
  }
});

// Очистка при закрытии страницы
window.addEventListener('beforeunload', () => {
  if (activeStream) {
    stopStream(activeStream);
  }
});
