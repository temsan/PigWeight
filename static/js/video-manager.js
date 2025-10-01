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
        
        // НЕ берем context здесь - оставляем для модульной системы
    }
    
    initializeOverlay() {
        if (!this.overlayCanvas) return;
        
        // Отключаем OffscreenCanvas worker - используем только основной поток
        // Это проще и надежнее для нашего случая
        if (this.overlayCanvas) {
            this.overlayContext = this.overlayCanvas.getContext('2d');
            console.log('✅ Overlay context инициализирован в основном потоке');
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
        
        // Получаем реальные размеры изображения (с учетом object-fit: contain)
        const rect = this.getRenderedImageRect();
        
        // Отрисовка вертикальных линий
        try {
            const leftX = (window.__lines && typeof window.__lines.left_x === 'number') ? window.__lines.left_x : 0.25;
            const rightX = (window.__lines && typeof window.__lines.right_x === 'number') ? window.__lines.right_x : 0.75;
            const x1 = rect.x + rect.w * leftX;
            const x2 = rect.x + rect.w * rightX;
            
            // Фон линий
            ctx.fillStyle = 'rgba(44,123,229,0.12)';
            ctx.fillRect(x1 - 2, rect.y, 4, rect.h);
            ctx.fillStyle = 'rgba(81,207,102,0.12)';
            ctx.fillRect(x2 - 2, rect.y, 4, rect.h);
            
            // Сами линии
            ctx.lineWidth = 2;
            ctx.strokeStyle = 'rgba(44,123,229,0.9)';
            ctx.beginPath();
            ctx.moveTo(x1, rect.y);
            ctx.lineTo(x1, rect.y + rect.h);
            ctx.stroke();
            
            ctx.strokeStyle = 'rgba(81,207,102,0.9)';
            ctx.beginPath();
            ctx.moveTo(x2, rect.y);
            ctx.lineTo(x2, rect.y + rect.h);
            ctx.stroke();
            
            // Кружочки на линиях
            ctx.fillStyle = 'rgba(44,123,229,0.95)';
            ctx.beginPath();
            ctx.arc(x1, rect.y + 10, 3.5, 0, Math.PI * 2);
            ctx.fill();
            
            ctx.fillStyle = 'rgba(81,207,102,0.95)';
            ctx.beginPath();
            ctx.arc(x2, rect.y + 10, 3.5, 0, Math.PI * 2);
            ctx.fill();
        } catch (e) {
            console.warn('Ошибка отрисовки линий:', e);
        }
        
        // Отрисовка масок свиней
        if (Array.isArray(masks) && masks.length > 0) {
            masks.forEach((mask, idx) => {
                if (!Array.isArray(mask) || mask.length < 3) return;
                
                const instId = (ids && ids[idx]) ? ids[idx] : (idx + 1);
                const color = this.getColorForInstance(instId);
                
                // Рисуем маску
                ctx.beginPath();
                mask.forEach((point, i) => {
                    const x = rect.x + point[0] * rect.w;
                    const y = rect.y + point[1] * rect.h;
                    if (i === 0) {
                        ctx.moveTo(x, y);
                    } else {
                        ctx.lineTo(x, y);
                    }
                });
                ctx.closePath();
                
                ctx.fillStyle = color;
                ctx.globalAlpha = 0.60;
                ctx.fill();
                ctx.globalAlpha = 1.0;
                
                // Обводка маски
                ctx.lineWidth = 1.0;
                ctx.strokeStyle = 'rgba(30,50,80,0.20)';
                ctx.stroke();
                
                // Рисуем label с учетом алиасов
                this.drawInstanceLabel(ctx, mask, instId, rect);
            });
        }
        
        // Отрисовка всплывающих счетчиков
        this.drawPopupCounters(ctx, rect);
    }
    
    getRenderedImageRect() {
        // Возвращает реальную область отображения кадра внутри wrapper с учётом object-fit: contain
        const wrapper = document.getElementById('videoWrapper');
        const cw = wrapper ? wrapper.clientWidth : this.overlayCanvas.width;
        const ch = wrapper ? wrapper.clientHeight : this.overlayCanvas.height;
        
        const webrtcEl = document.getElementById('webrtcVideo');
        const webrtcVisible = webrtcEl && webrtcEl.style.display !== 'none';
        
        let iw = 0, ih = 0;
        if (webrtcVisible) {
            iw = webrtcEl.videoWidth || 0;
            ih = webrtcEl.videoHeight || 0;
        } else if (this.videoStream) {
            iw = this.videoStream.naturalWidth || 0;
            ih = this.videoStream.naturalHeight || 0;
        }
        
        if (!iw || !ih) return { x: 0, y: 0, w: cw, h: ch };
        
        const scale = Math.min(cw / iw, ch / ih);
        const w = Math.round(iw * scale);
        const h = Math.round(ih * scale);
        const x = Math.floor((cw - w) / 2);
        const y = Math.floor((ch - h) / 2);
        
        return { x, y, w, h };
    }
    
    getColorForInstance(instId) {
        const hue = (Number(instId) * 57) % 360;
        return `hsla(${hue}, 65%, 70%, 0.22)`;
    }
    
    drawInstanceLabel(ctx, mask, instId, rect) {
        // Вычисляем центр маски для размещения label
        let sx = 0, sy = 0, pc = 0;
        mask.forEach(([nx, ny]) => {
            sx += nx;
            sy += ny;
            pc++;
        });
        
        if (pc === 0) return;
        
        const cx = rect.x + (sx / pc) * rect.w;
        const cy = rect.y + (sy / pc) * rect.h;
        
        // Получаем алиас для инстанса, если есть
        const idAliases = window.idAliases || {};
        const label = String(idAliases[instId] ?? instId);
        
        // Рисуем label с обводкой для читаемости
        ctx.save();
        ctx.font = '700 14px system-ui, Segoe UI, Arial';
        ctx.textAlign = 'center';
        ctx.textBaseline = 'middle';
        ctx.lineWidth = 3.5;
        ctx.strokeStyle = 'rgba(0,0,0,0.65)';
        ctx.strokeText(label, cx, cy);
        ctx.fillStyle = '#ffffff';
        ctx.fillText(label, cx, cy);
        ctx.restore();
    }
    
    drawPopupCounters(ctx, rect) {
        if (!this.popupCounters || this.popupCounters.length === 0) return;
        
        const now = performance.now();
        
        ctx.save();
        this.popupCounters.forEach(popup => {
            const elapsed = now - popup.startTime;
            const progress = Math.min(1, elapsed / popup.duration);
            const alpha = 1 - progress;
            const ease = progress * (2 - progress); // Плавная анимация
            
            const x = rect.x + popup.x * rect.w;
            const y = rect.y + popup.y * rect.h - ease * popup.riseDistance;
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
        ctx.globalAlpha = 1.0; // Сброс прозрачности
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