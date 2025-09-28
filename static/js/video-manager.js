/**
 * Менеджер видео
 * Управляет воспроизведением, стримами, overlay
 */

import { EventEmitter } from './utils/event-emitter.js';

export class VideoManager extends EventEmitter {
    constructor() {
        super();
        
        // Элементы видео
        this.videoStream = null;
        this.webrtcVideo = null;
        this.overlayCanvas = null;
        this.overlayContext = null;
        
        // Состояние
        this.activeStreamId = null;
        this.isVideoPlaying = false;
        this.hasFirstFrame = false;
        
        // Overlay данные
        this.lastMasks = null;
        this.lastIds = null;
        this.overlayPending = null;
        this.overlayLastDraw = 0;
        this.overlayMaxFps = 12;
        this.overlayBlockUntil = 0;
        
        // Worker для overlay
        this.maskWorker = null;
        this.offscreenCanvas = null;
        this.overlayRaf = null;
    }
    
    async init() {
        console.log('🎥 Инициализация Video Manager...');
        
        // Кэшируем элементы
        this.cacheElements();
        
        // Инициализируем overlay
        this.initializeOverlay();
        
        // Настраиваем обработчики
        this.setupEventHandlers();
        
        // Запускаем overlay луп
        this.startOverlayLoop();
        
        console.log('✅ Video Manager инициализирован');
    }
    
    cacheElements() {
        this.videoStream = document.getElementById('videoStream');
        this.webrtcVideo = document.getElementById('webrtcVideo');
        this.overlayCanvas = document.getElementById('overlayCanvas');
        
        if (this.overlayCanvas) {
            this.overlayContext = this.overlayCanvas.getContext('2d');
        }
    }
    
    initializeOverlay() {
        if (!this.overlayCanvas) return;
        
        try {
            // Создаем OffscreenCanvas для worker
            if (typeof OffscreenCanvas !== 'undefined') {
                this.offscreenCanvas = this.overlayCanvas.transferControlToOffscreen();
                this.maskWorker = new Worker('/static/js/mask-worker.js');
                
                this.maskWorker.postMessage({
                    type: 'init',
                    canvas: this.offscreenCanvas
                }, [this.offscreenCanvas]);
                
                console.log('✅ Overlay worker инициализирован');
            } else {
                console.log('⚠️ OffscreenCanvas не поддерживается, используем основной поток');
            }
        } catch (error) {
            console.warn('⚠️ Не удалось инициализировать overlay worker:', error);
        }
    }
    
    setupEventHandlers() {
        // Обработчики изменения размера
        window.addEventListener('resize', () => {
            this.updateOverlaySize();
        });
        
        // Обработчики видео элементов
        if (this.videoStream) {
            this.videoStream.addEventListener('load', () => {
                this.hasFirstFrame = true;
                this.updateOverlaySize();
                this.emit('first_frame');
            });
        }
        
        if (this.webrtcVideo) {
            this.webrtcVideo.addEventListener('loadedmetadata', () => {
                this.hasFirstFrame = true;
                this.updateOverlaySize();
                this.emit('first_frame');
            });
        }
    }
    
    updateOverlaySize() {
        if (!this.overlayCanvas) return;
        
        const rect = this.overlayCanvas.getBoundingClientRect();
        const width = Math.floor(rect.width);
        const height = Math.floor(rect.height);
        
        if (this.overlayCanvas.width !== width || this.overlayCanvas.height !== height) {
            this.overlayCanvas.width = width;
            this.overlayCanvas.height = height;
            
            // Обновляем размер в worker
            if (this.maskWorker) {
                this.maskWorker.postMessage({
                    type: 'resize',
                    size: { w: width, h: height }
                });
            }
        }
    }
    
    scheduleOverlay(masks, ids) {
        if (performance.now() < this.overlayBlockUntil) return;
        
        // Сохраняем последние маски для перерисовки
        this.lastMasks = masks;
        this.lastIds = ids;
        
        this.overlayPending = { masks, ids };
        
        if (performance.now() - this.overlayLastDraw > 1000 / this.overlayMaxFps) {
            this.drawOverlay();
        }
    }
    
    drawOverlay() {
        if (!this.overlayPending) return;
        
        const { masks, ids } = this.overlayPending;
        this.overlayPending = null;
        this.overlayLastDraw = performance.now();
        
        if (this.maskWorker) {
            // Используем worker
            const rect = this.overlayCanvas.getBoundingClientRect();
            this.maskWorker.postMessage({
                type: 'overlay',
                masks: masks,
                ids: ids,
                popupCounters: this.popupCounters || [],
                rect: { x: 0, y: 0, w: rect.width, h: rect.height }
            });
        } else {
            // Fallback на основной поток
            this.drawOverlayDirect(masks, ids);
        }
    }
    
    drawOverlayDirect(masks, ids) {
        if (!this.overlayContext) return;
        
        const canvas = this.overlayCanvas;
        const ctx = this.overlayContext;
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Отрисовка масок свиней
        if (Array.isArray(masks) && masks.length > 0) {
            masks.forEach((mask, idx) => {
                if (!Array.isArray(mask) || mask.length < 3) return;
                
                const instId = (ids && ids[idx]) ? ids[idx] : (idx + 1);
                const color = this.getColorForInstance(instId);
                
                // Рисуем маску
                ctx.beginPath();
                mask.forEach((point, i) => {
                    const x = point[0] * canvas.width;
                    const y = point[1] * canvas.height;
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                });
                ctx.closePath();
                
                ctx.fillStyle = color;
                ctx.fill();
                
                // Рисуем label
                this.drawInstanceLabel(ctx, mask, instId, canvas.width, canvas.height);
            });
        }
        
        // Отрисовка всплывающих счетчиков
        this.drawPopupCounters(ctx, canvas.width, canvas.height);
    }
    
    getColorForInstance(instId) {
        const hue = (Number(instId) * 57) % 360;
        return `hsla(${hue}, 65%, 70%, 0.22)`;
    }
    
    drawInstanceLabel(ctx, mask, instId, canvasWidth, canvasHeight) {
        // Находим минимальные координаты для размещения label
        let minX = Infinity, minY = Infinity;
        mask.forEach(point => {
            const x = point[0] * canvasWidth;
            const y = point[1] * canvasHeight;
            if (x < minX) minX = x;
            if (y < minY) minY = y;
        });
        
        const label = String(instId);
        const padding = 6;
        const radius = 6;
        
        ctx.font = '600 13px system-ui, Segoe UI, Arial';
        const textWidth = ctx.measureText(label).width;
        const badgeWidth = Math.ceil(textWidth + padding * 2);
        const badgeHeight = 20;
        
        const badgeX = Math.max(0, Math.min(minX - 8, canvasWidth - badgeWidth));
        const badgeY = Math.max(0, Math.min(minY - 8, canvasHeight - badgeHeight));
        
        // Рисуем фон badge
        ctx.save();
        ctx.beginPath();
        ctx.roundRect(badgeX, badgeY, badgeWidth, badgeHeight, radius);
        ctx.fillStyle = 'rgba(30,50,80,0.85)';
        ctx.fill();
        
        // Рисуем текст
        ctx.fillStyle = '#fff';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.fillText(label, badgeX + badgeWidth / 2, badgeY + badgeHeight / 2);
        ctx.restore();
    }
    
    drawPopupCounters(ctx, canvasWidth, canvasHeight) {
        if (!this.popupCounters || this.popupCounters.length === 0) return;
        
        const now = performance.now();
        
        ctx.save();
        this.popupCounters.forEach(popup => {
            const elapsed = now - popup.startTime;
            const progress = Math.min(1, elapsed / popup.duration);
            const alpha = 1 - progress;
            const ease = progress * (2 - progress); // Плавная анимация
            
            const x = popup.x * canvasWidth;
            const y = popup.y * canvasHeight - ease * popup.riseDistance;
            const radius = 16;
            
            // Применяем прозрачность
            ctx.globalAlpha = Math.max(0, alpha);
            
            // Рисуем круг фона
            ctx.beginPath();
            ctx.arc(x, y, radius, 0, Math.PI * 2);
            ctx.fillStyle = popup.color;
            ctx.fill();
            
            // Рисуем обводку
            ctx.lineWidth = 2;
            ctx.strokeStyle = 'rgba(0,0,0,0.3)';
            ctx.stroke();
            
            // Рисуем текст
            ctx.font = '700 14px system-ui, Segoe UI, Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#fff';
            ctx.fillText(popup.text, x, y + 1);
        });
        ctx.restore();
    }
    
    handleControl(action, data) {
        switch (action) {
            case 'play':
                this.play();
                break;
            case 'pause':
                this.pause();
                break;
            case 'seek':
                this.seek(data.time);
                break;
            case 'switch_stream':
                this.switchStream(data.streamId);
                break;
            default:
                console.warn('Неизвестное действие видео:', action);
        }
    }
    
    play() {
        this.isVideoPlaying = true;
        this.emit('play');
    }
    
    pause() {
        this.isVideoPlaying = false;
        this.emit('pause');
    }
    
    seek(time) {
        this.emit('seek', time);
    }
    
    switchStream(streamId) {
        if (this.activeStreamId === streamId) return;
        
        this.activeStreamId = streamId;
        this.hasFirstFrame = false;
        this.emit('stream_change', streamId);
        this.emit('stream_start', streamId);
    }
    
    clearOverlay() {
        if (this.overlayContext) {
            this.overlayContext.clearRect(0, 0, this.overlayCanvas.width, this.overlayCanvas.height);
        }
        
        if (this.maskWorker) {
            this.maskWorker.postMessage({
                type: 'overlay',
                masks: [],
                ids: [],
                rect: { x: 0, y: 0, w: 0, h: 0 }
            });
        }
    }
    
    startOverlayLoop() {
        // Инициализируем основной луп overlay рендеринга
        if (!this.overlayRaf) {
            console.log('🎆 Запуск overlay лупа');
            this.overlayTick();
        }
    }
    
    overlayTick() {
        const now = performance.now();
        const minDt = 1000 / this.overlayMaxFps;
        
        // Обрабатываем очередь overlay, если есть отложенные задачи
        if (this.overlayPending && (now - this.overlayLastDraw >= minDt)) {
            this.drawOverlay();
        }
        // Перерисовываем последние маски, если нет отложенных задач
        else if (!this.overlayPending && this.lastMasks && (now - this.overlayLastDraw >= minDt)) {
            this.drawOverlayDirect(this.lastMasks, this.lastIds);
            this.overlayLastDraw = now;
        }
        
        // Обрабатываем всплывающие счетчики
        this.updatePopupCounters(now);
        
        this.overlayRaf = requestAnimationFrame(() => this.overlayTick());
    }
    
    handleCrossings(crossings) {
        // Обрабатываем пересечения линий для создания всплывающих счетчиков
        if (!Array.isArray(crossings) || crossings.length === 0) return;
        
        const now = performance.now();
        
        // Инициализируем массив всплывающих счетчиков
        if (!this.popupCounters) {
            this.popupCounters = [];
        }
        
        // Отслеживаем последний timestamp для избежания дубликатов
        const prevTs = this.lastPopupTimestamp || 0;
        let maxTs = prevTs;
        
        crossings.forEach(crossing => {
            const tsMs = Number(crossing.ts || 0) * 1000;
            if (tsMs > prevTs) {
                const side = String(crossing.side || 'left');
                const mode = String(crossing.mode || 'enter');
                const isPositive = (side === 'left' && mode === 'enter') || (side === 'right' && mode === 'exit');
                
                const text = isPositive ? '+1' : '-1';
                const color = isPositive ? '#51cf66' : '#ff6b6b';
                
                this.popupCounters.push({
                    x: Number(crossing.x || 0.5),
                    y: Number(crossing.y || 0.5),
                    text: text,
                    color: color,
                    startTime: now,
                    duration: 1200,
                    riseDistance: 40
                });
                
                console.log(`🎆 Создан всплывающий счетчик: ${text} на (${crossing.x}, ${crossing.y})`);
            }
            if (tsMs > maxTs) maxTs = tsMs;
        });
        
        this.lastPopupTimestamp = maxTs;
    }
    
    updatePopupCounters(now) {
        if (!this.popupCounters || this.popupCounters.length === 0) return;
        
        // Фильтруем активные счетчики (не истекшие)
        this.popupCounters = this.popupCounters.filter(popup => {
            return (now - popup.startTime) < popup.duration;
        });
        
        // Если есть активные счетчики, перерисовываем overlay
        if (this.popupCounters.length > 0) {
            // Принудительно перерисовываем для обновления всплывающих счетчиков
            this.scheduleOverlay(this.lastMasks, this.lastIds);
        }
    }
    
    destroy() {
        if (this.overlayRaf) {
            cancelAnimationFrame(this.overlayRaf);
            this.overlayRaf = null;
        }
        
        if (this.maskWorker) {
            this.maskWorker.terminate();
            this.maskWorker = null;
        }
        
        this.removeAllListeners();
    }
}