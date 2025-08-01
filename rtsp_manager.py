import os
import subprocess
import threading
import atexit

class RTSPManager:
    def __init__(self):
        self.streams = {}
        self.stream_dir = "stream"
        if not os.path.exists(self.stream_dir):
            os.makedirs(self.stream_dir)
        atexit.register(self.cleanup)

    def start_stream(self, camera_id, rtsp_url):
        """Запускает проксирование RTSP потока через HLS"""
        if camera_id in self.streams and self.streams[camera_id].poll() is None:
            print(f"Stream {camera_id} already running")
            return

        output_path = os.path.join(self.stream_dir, f"stream_{camera_id}.m3u8")
        
        # Настройки ffmpeg для проксирования RTSP в HLS
        command = [
            'ffmpeg',
            '-i', rtsp_url,
            '-c:v', 'libx264',  # используем H.264 кодек
            '-preset', 'ultrafast',  # минимальная задержка
            '-tune', 'zerolatency',  # оптимизация для стриминга
            '-r', '25',  # 25 кадров в секунду
            '-f', 'hls',  # формат HLS
            '-hls_time', '2',  # длина сегмента
            '-hls_list_size', '3',  # количество сегментов в плейлисте
            '-hls_flags', 'delete_segments+append_list',  # автоудаление старых сегментов
            output_path
        ]
        
        try:
            # Запускаем ffmpeg процесс
            process = subprocess.Popen(
                command,
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                universal_newlines=True
            )
            self.streams[camera_id] = process
            print(f"Started streaming for camera {camera_id}")
            
            # Запускаем отдельный поток для мониторинга вывода ffmpeg
            threading.Thread(
                target=self._monitor_stream,
                args=(camera_id, process),
                daemon=True
            ).start()
            
            return f"/stream/stream_{camera_id}.m3u8"
        except Exception as e:
            print(f"Error starting stream for camera {camera_id}: {str(e)}")
            return None

    def stop_stream(self, camera_id):
        """Останавливает проксирование потока"""
        if camera_id in self.streams:
            process = self.streams[camera_id]
            if process.poll() is None:  # процесс все еще работает
                process.terminate()
                try:
                    process.wait(timeout=5)  # ждем завершения процесса
                except subprocess.TimeoutExpired:
                    process.kill()  # принудительно завершаем, если не завершился
            del self.streams[camera_id]
            print(f"Stopped streaming for camera {camera_id}")

    def _monitor_stream(self, camera_id, process):
        """Мониторит вывод ffmpeg процесса"""
        while True:
            line = process.stderr.readline()
            if not line and process.poll() is not None:
                break
            if line:
                print(f"FFMPEG [{camera_id}]: {line.strip()}")

    def cleanup(self):
        """Очищает все запущенные процессы при завершении работы"""
        for camera_id in list(self.streams.keys()):
            self.stop_stream(camera_id)
