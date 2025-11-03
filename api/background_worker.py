"""
Фоновый worker для асинхронной обработки видео
"""

import asyncio
import logging
from pathlib import Path
from typing import Dict, List, Optional, Callable
from dataclasses import dataclass, asdict
from datetime import datetime
import json
import time

logger = logging.getLogger(__name__)


@dataclass
class ProcessingTask:
    """Задача обработки видео"""
    task_id: str
    video_path: str
    status: str = "pending"  # pending, processing, completed, failed
    progress: float = 0.0
    created_at: float = None
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    error: Optional[str] = None
    result: Optional[Dict] = None
    
    def __post_init__(self):
        if self.created_at is None:
            self.created_at = time.time()
    
    def to_dict(self) -> Dict:
        return asdict(self)


class VideoProcessingQueue:
    """Управляет очередью обработки видео"""
    
    def __init__(self, max_workers: int = 1, persist_dir: str = "records/queue"):
        self.max_workers = max_workers
        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        
        self.queue: asyncio.Queue = asyncio.Queue()
        self.tasks: Dict[str, ProcessingTask] = {}
        self.active_tasks: List[str] = []
        self.callbacks: Dict[str, List[Callable]] = {
            'on_task_start': [],
            'on_task_progress': [],
            'on_task_complete': [],
            'on_task_error': []
        }
        
        # Загружаем состояние из диска
        self._load_state()
    
    def _load_state(self):
        """Загружает состояние очереди из диска"""
        state_file = self.persist_dir / "queue_state.json"
        if state_file.exists():
            try:
                with open(state_file, 'r') as f:
                    data = json.load(f)
                    for task_data in data.get('tasks', []):
                        task = ProcessingTask(**task_data)
                        # Возвращаем незавершённые задачи в очередь
                        if task.status in ['pending', 'processing']:
                            self.tasks[task.task_id] = task
                            asyncio.create_task(self.queue.put(task.task_id))
                logger.info(f"✅ Загруженo {len(self.tasks)} задач из очереди")
            except Exception as e:
                logger.error(f"❌ Ошибка загрузки состояния очереди: {e}")
    
    def _save_state(self):
        """Сохраняет состояние очереди на диск"""
        try:
            state_file = self.persist_dir / "queue_state.json"
            data = {
                'timestamp': datetime.now().isoformat(),
                'tasks': [task.to_dict() for task in self.tasks.values()]
            }
            with open(state_file, 'w') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            logger.error(f"❌ Ошибка сохранения состояния очереди: {e}")
    
    async def add_task(self, task_id: str, video_path: str) -> ProcessingTask:
        """Добавляет задачу в очередь"""
        task = ProcessingTask(task_id=task_id, video_path=video_path)
        self.tasks[task_id] = task
        await self.queue.put(task_id)
        self._save_state()
        logger.info(f"➕ Задача {task_id} добавлена в очередь")
        return task
    
    async def process_queue(self, processor_func: Callable):
        """Обрабатывает очередь с заданной функцией"""
        while True:
            try:
                task_id = await asyncio.wait_for(self.queue.get(), timeout=1.0)
                task = self.tasks[task_id]
                
                # Запускаем обработку
                task.status = "processing"
                task.started_at = time.time()
                self.active_tasks.append(task_id)
                self._save_state()
                self._emit('on_task_start', task)
                
                logger.info(f"🔄 Обработка: {task.video_path}")
                
                try:
                    # Вызываем функцию обработки с callback для прогресса
                    async def progress_callback(progress: float):
                        task.progress = progress
                        self._emit('on_task_progress', task)
                        self._save_state()
                    
                    result = await processor_func(task.video_path, progress_callback)
                    
                    # Успешное завершение
                    task.status = "completed"
                    task.completed_at = time.time()
                    task.result = result
                    self._emit('on_task_complete', task)
                    logger.info(f"✅ Задача {task_id} завершена")
                    
                except Exception as e:
                    # Ошибка обработки
                    task.status = "failed"
                    task.completed_at = time.time()
                    task.error = str(e)
                    self._emit('on_task_error', task)
                    logger.error(f"❌ Ошибка в задаче {task_id}: {e}")
                
                finally:
                    if task_id in self.active_tasks:
                        self.active_tasks.remove(task_id)
                    self._save_state()
                    self.queue.task_done()
                    
            except asyncio.TimeoutError:
                # Нет задач в очереди, ждём
                await asyncio.sleep(0.5)
            except Exception as e:
                logger.error(f"❌ Ошибка обработки очереди: {e}")
                await asyncio.sleep(1)
    
    def subscribe(self, event: str, callback: Callable):
        """Подписывается на событие"""
        if event in self.callbacks:
            self.callbacks[event].append(callback)
    
    def _emit(self, event: str, task: ProcessingTask):
        """Отправляет событие подписчикам"""
        if event in self.callbacks:
            for callback in self.callbacks[event]:
                try:
                    asyncio.create_task(callback(task)) if asyncio.iscoroutinefunction(callback) else callback(task)
                except Exception as e:
                    logger.error(f"❌ Ошибка в callback {event}: {e}")
    
    def get_task(self, task_id: str) -> Optional[ProcessingTask]:
        """Получает информацию о задаче"""
        return self.tasks.get(task_id)
    
    def get_all_tasks(self) -> List[ProcessingTask]:
        """Получает все задачи"""
        return list(self.tasks.values())
    
    def get_active_tasks(self) -> List[ProcessingTask]:
        """Получает активные задачи"""
        return [self.tasks[tid] for tid in self.active_tasks if tid in self.tasks]
    
    def get_stats(self) -> Dict:
        """Получает статистику очереди"""
        tasks = list(self.tasks.values())
        completed = [t for t in tasks if t.status == 'completed']
        failed = [t for t in tasks if t.status == 'failed']
        
        return {
            'total_tasks': len(tasks),
            'pending': sum(1 for t in tasks if t.status == 'pending'),
            'processing': len(self.active_tasks),
            'completed': len(completed),
            'failed': len(failed),
            'success_rate': len(completed) / max(1, len(completed) + len(failed))
        }


# Глобальная очередь (инициализируется в app.py)
_processing_queue: Optional[VideoProcessingQueue] = None


def get_processing_queue() -> VideoProcessingQueue:
    """Получает глобальную очередь обработки"""
    global _processing_queue
    if _processing_queue is None:
        _processing_queue = VideoProcessingQueue()
    return _processing_queue
