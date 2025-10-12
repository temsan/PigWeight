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
        this.overlayEnabled = true;
        
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
        
        // Инициализируем позиции линий по умолчанию
        if (!window.__lines || typeof window.__lines.left_x !== 'number') {
            window.__lines = { left_x: 0.25, right_x: 0.75 };
            console.log('🔧 Инициализированы позиции линий по умолчанию:', window.__lines);
        }
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
        this.maskWorker = null; // Принудительно отключаем worker
        if (this.overlayCanvas) {
            this.overlayContext = this.overlayCanvas.getContext('2d');
            console.log('✅ Overlay context инициализирован в основном потоке');
        }
    }
    
    setupEventHandlers() {
        // Обработчики изменения размера
        window.addEventListener('resize', () => {
            console.log('🔄 Изменение размера окна, обновляем overlay');
            this.updateOverlaySize();
            // Принудительно перерисовываем overlay после изменения размера
            if (this.lastMasks && this.lastIds) {
                setTimeout(() => {
                    console.log('🔄 Принудительная перерисовка overlay после resize');
                    this.drawOverlayDirect(this.lastMasks, this.lastIds);
                }, 100);
            }
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
        
        // Обработчики перетаскивания линий
        this.setupLineDragging();
    }
    
    updateOverlaySize() {
        if (!this.overlayCanvas) return;
        
        const rect = this.overlayCanvas.getBoundingClientRect();
        const width = Math.floor(rect.width);
        const height = Math.floor(rect.height);
        
        if (this.overlayCanvas.width !== width || this.overlayCanvas.height !== height) {
            console.log('🔄 Изменение размера overlay canvas:', {
                old: { w: this.overlayCanvas.width, h: this.overlayCanvas.height },
                new: { w: width, h: height }
            });
            
            this.overlayCanvas.width = width;
            this.overlayCanvas.height = height;
            
            // Принудительно перерисовываем overlay после изменения размера
            if (this.lastMasks && this.lastIds) {
                console.log('🔄 Перерисовка overlay после изменения размера');
                // Небольшая задержка для стабилизации размеров
                setTimeout(() => {
                    this.drawOverlayDirect(this.lastMasks, this.lastIds);
                }, 50);
            }
            
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
        console.log('🎯 scheduleOverlay вызван:', {
            masksCount: masks ? masks.length : 0,
            idsCount: ids ? ids.length : 0,
            overlayBlockUntil: this.overlayBlockUntil,
            currentTime: performance.now(),
            overlayEnabled: this.overlayEnabled
        });
        
        // Проверяем что overlay включен
        if (!this.overlayEnabled) {
            console.log('🚫 scheduleOverlay пропущен - overlay отключен');
            return;
        }
        
        // Принудительно обновляем размеры canvas при получении новых масок
        this.updateOverlaySize();
        
        if (performance.now() < this.overlayBlockUntil) {
            console.log('🚫 scheduleOverlay заблокирован до:', this.overlayBlockUntil);
            return;
        }
        
        // Сохраняем последние маски для перерисовки
        this.lastMasks = masks;
        this.lastIds = ids;
        
        this.overlayPending = { masks, ids };
        
        const timeSinceLastDraw = performance.now() - this.overlayLastDraw;
        const minInterval = 1000 / this.overlayMaxFps;
        
        console.log('⏰ scheduleOverlay:', {
            timeSinceLastDraw: timeSinceLastDraw.toFixed(1),
            minInterval: minInterval.toFixed(1),
            shouldDraw: timeSinceLastDraw > minInterval
        });
        
        if (timeSinceLastDraw > minInterval) {
            console.log('🎨 Вызываю drawOverlay()');
            this.drawOverlay();
        } else {
            console.log('⏳ drawOverlay отложен, слишком рано');
        }
    }
    
    drawOverlay() {
        if (!this.overlayPending) {
            console.log('🚫 drawOverlay: нет отложенных задач');
            return;
        }
        
        const { masks, ids } = this.overlayPending;
        this.overlayPending = null;
        this.overlayLastDraw = performance.now();
        
        console.log('🎨 drawOverlay:', {
            masksCount: masks ? masks.length : 0,
            idsCount: ids ? ids.length : 0,
            hasWorker: !!this.maskWorker
        });
        
        if (this.maskWorker) {
            // Используем worker
            console.log('👷 Используем worker для отрисовки');
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
            console.log('🎯 Используем drawOverlayDirect (основной поток)');
            this.drawOverlayDirect(masks, ids);
        }
    }
    
    drawOverlayDirect(masks, ids) {
        if (!this.overlayContext) return;
        
        const canvas = this.overlayCanvas;
        const ctx = this.overlayContext;
        
        // Принудительно обновляем размеры canvas перед отрисовкой
        this.updateOverlaySize();
        
        ctx.clearRect(0, 0, canvas.width, canvas.height);
        
        // Получаем реальные размеры изображения (с учетом object-fit: contain)
        const rect = this.getRenderedImageRect();
        
        // Проверяем что rect валидный
        if (!rect || rect.w <= 0 || rect.h <= 0) {
            console.warn('⚠️ Неверные размеры rect:', rect);
            return;
        }
        
        // Отладочные логи
        console.log('🎨 drawOverlayDirect:', {
            masksCount: masks ? masks.length : 0,
            idsCount: ids ? ids.length : 0,
            rect: rect,
            canvasSize: { width: canvas.width, height: canvas.height },
            canvasRect: canvas.getBoundingClientRect(),
            videoElement: {
                webrtc: document.getElementById('webrtcVideo')?.getBoundingClientRect(),
                stream: this.videoStream?.getBoundingClientRect()
            }
        });
        
        // Отрисовка вертикальных линий
        try {
            const leftX = (window.__lines && typeof window.__lines.left_x === 'number') ? window.__lines.left_x : 0.25;
            const rightX = (window.__lines && typeof window.__lines.right_x === 'number') ? window.__lines.right_x : 0.75;
            const x1 = rect.x + rect.w * leftX;
            const x2 = rect.x + rect.w * rightX;
            
            console.log('📏 Рисую вертикальные линии:', {
                leftX, rightX,
                x1, x2,
                rect: rect,
                lines: window.__lines
            });
            
            // Фон линий (более заметный)
            ctx.fillStyle = 'rgba(44,123,229,0.15)';
            ctx.fillRect(x1 - 3, rect.y, 6, rect.h);
            ctx.fillStyle = 'rgba(81,207,102,0.15)';
            ctx.fillRect(x2 - 3, rect.y, 6, rect.h);
            
            // Сами линии (толще и ярче)
            ctx.lineWidth = 3;
            ctx.strokeStyle = 'rgba(44,123,229,0.95)';
            ctx.beginPath();
            ctx.moveTo(x1, rect.y);
            ctx.lineTo(x1, rect.y + rect.h);
            ctx.stroke();
            
            ctx.strokeStyle = 'rgba(81,207,102,0.95)';
            ctx.beginPath();
            ctx.moveTo(x2, rect.y);
            ctx.lineTo(x2, rect.y + rect.h);
            ctx.stroke();
            
            // Индикаторы направления (стрелки)
            const arrowSize = 8;
            const arrowY = rect.y + 20;
            
            // Стрелка для левой линии (вправо)
            ctx.fillStyle = 'rgba(44,123,229,0.95)';
            ctx.beginPath();
            ctx.moveTo(x1, arrowY);
            ctx.lineTo(x1 + arrowSize, arrowY - arrowSize/2);
            ctx.lineTo(x1 + arrowSize, arrowY + arrowSize/2);
            ctx.closePath();
            ctx.fill();
            
            // Стрелка для правой линии (влево)
            ctx.fillStyle = 'rgba(81,207,102,0.95)';
            ctx.beginPath();
            ctx.moveTo(x2, arrowY);
            ctx.lineTo(x2 - arrowSize, arrowY - arrowSize/2);
            ctx.lineTo(x2 - arrowSize, arrowY + arrowSize/2);
            ctx.closePath();
            ctx.fill();
            
            // Кружочки на линиях (больше и ярче)
            ctx.fillStyle = 'rgba(44,123,229,0.95)';
            ctx.beginPath();
            ctx.arc(x1, rect.y + 10, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            ctx.fillStyle = 'rgba(81,207,102,0.95)';
            ctx.beginPath();
            ctx.arc(x2, rect.y + 10, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Подписи линий
            ctx.font = 'bold 12px system-ui, Segoe UI, Arial';
            ctx.fillStyle = 'rgba(44,123,229,0.95)';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Вход', x1, rect.y + 35);
            
            ctx.fillStyle = 'rgba(81,207,102,0.95)';
            ctx.fillText('Выход', x2, rect.y + 35);
        } catch (e) {
            console.warn('Ошибка отрисовки линий:', e);
        }
        
        // Отрисовка масок свиней
        if (Array.isArray(masks) && masks.length > 0) {
            console.log('🎭 Начинаю отрисовку масок:', masks.length);
            masks.forEach((mask, idx) => {
                if (!Array.isArray(mask) || mask.length < 3) {
                    console.warn('⚠️ Пропускаю маску', idx, 'неверный формат:', mask);
                    return;
                }
                
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
                
                console.log(`🎨 Рисую маску ${idx}:`, {
                    instId,
                    color,
                    points: mask.length,
                    firstPoint: mask[0],
                    rect: rect,
                    maskBounds: {
                        minX: Math.min(...mask.map(p => p[0])),
                        maxX: Math.max(...mask.map(p => p[0])),
                        minY: Math.min(...mask.map(p => p[1])),
                        maxY: Math.max(...mask.map(p => p[1]))
                    }
                });
                
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
        if (!wrapper) {
            console.warn('⚠️ videoWrapper не найден, используем размеры canvas');
            return { x: 0, y: 0, w: this.overlayCanvas.width, h: this.overlayCanvas.height };
        }
        
        const cw = wrapper.clientWidth;
        const ch = wrapper.clientHeight;
        
        const webrtcEl = document.getElementById('webrtcVideo');
        const webrtcVisible = webrtcEl && webrtcEl.style.display !== 'none';
        
        console.log('🔍 Поиск видео элемента:', {
            hasWebrtcEl: !!webrtcEl,
            webrtcDisplay: webrtcEl ? webrtcEl.style.display : 'N/A',
            webrtcVisible: webrtcVisible,
            hasVideoStream: !!this.videoStream,
            videoStreamSrc: this.videoStream ? this.videoStream.src : 'N/A'
        });
        
        let videoEl = null;
        if (webrtcVisible) {
            videoEl = webrtcEl;
            console.log('📹 Используем WebRTC элемент');
        } else if (this.videoStream) {
            videoEl = this.videoStream;
            console.log('📹 Используем videoStream элемент');
        }
        
        if (!videoEl) {
            console.warn('⚠️ Видео элемент не найден, используем размеры wrapper');
            return { x: 0, y: 0, w: cw, h: ch };
        }
        
        // Получаем реальные размеры видео
        const iw = videoEl.videoWidth || videoEl.naturalWidth || 0;
        const ih = videoEl.videoHeight || videoEl.naturalHeight || 0;
        
        console.log('📏 Размеры видео элемента:', {
            videoWidth: videoEl.videoWidth,
            videoHeight: videoEl.videoHeight,
            naturalWidth: videoEl.naturalWidth,
            naturalHeight: videoEl.naturalHeight,
            finalWidth: iw,
            finalHeight: ih
        });
        
        if (!iw || !ih) {
            console.warn('⚠️ Размеры видео не определены, используем размеры wrapper');
            return { x: 0, y: 0, w: cw, h: ch };
        }
        
        // Вычисляем масштаб для object-fit: contain
        const scale = Math.min(cw / iw, ch / ih);
        const w = Math.round(iw * scale);
        const h = Math.round(ih * scale);
        const x = Math.floor((cw - w) / 2);
        const y = Math.floor((ch - h) / 2);
        
        console.log('📐 getRenderedImageRect:', {
            wrapper: { w: cw, h: ch },
            video: { w: iw, h: ih },
            rendered: { x, y, w, h, scale: scale.toFixed(3) }
        });
        
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
        // Проверяем как window.idAliases, так и глобальную переменную idAliases
        const idAliases = window.idAliases || (typeof idAliases !== 'undefined' ? idAliases : {});
        const label = String(idAliases[instId] ?? instId);
        
        // Отладочная информация для алиасов
        if (idAliases[instId]) {
            console.log(`🏷️ Отображаем алиас для ID ${instId}: "${label}"`);
        }
        
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
        this.popupCounters.forEach((popup, index) => {
            const elapsed = now - popup.startTime;
            const progress = Math.min(1, elapsed / popup.duration);
            const alpha = 1 - progress;
            const ease = progress * (2 - progress); // Плавная анимация
            
            const x = rect.x + popup.x * rect.w;
            const y = rect.y + popup.y * rect.h - ease * popup.riseDistance;
            
            // Применяем прозрачность
            ctx.globalAlpha = Math.max(0, alpha);
            
            // Рисуем кружок пересечения на линии (стационарный)
            ctx.beginPath();
            ctx.arc(x, y, 8, 0, Math.PI * 2);
            ctx.fillStyle = popup.color;
            ctx.fill();
            
            // Обводка кружка
            ctx.lineWidth = 2;
            ctx.strokeStyle = '#ffffff';
            ctx.stroke();
            
            // Всплывающий счетчик (поднимается вверх)
            const popupY = y - 25 - ease * 20;
            const radius = 18;
            
            // Тень для всплывающего счетчика
            ctx.shadowColor = 'rgba(0,0,0,0.3)';
            ctx.shadowBlur = 4;
            ctx.shadowOffsetY = 2;
            
            // Рисуем всплывающий круг
            ctx.beginPath();
            ctx.arc(x, popupY, radius, 0, Math.PI * 2);
            ctx.fillStyle = popup.color;
            ctx.fill();
            
            // Обводка всплывающего круга
            ctx.lineWidth = 3;
            ctx.strokeStyle = '#ffffff';
            ctx.stroke();
            
            // Сбрасываем тень
            ctx.shadowColor = 'transparent';
            ctx.shadowBlur = 0;
            ctx.shadowOffsetY = 0;
            
            // Рисуем текст
            ctx.font = 'bold 16px system-ui, Segoe UI, Arial';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'middle';
            ctx.fillStyle = '#ffffff';
            ctx.fillText(popup.text, x, popupY);
            
            // Добавляем стрелку, указывающую на точку пересечения
            if (progress < 0.8) { // Показываем стрелку только в начале анимации
                ctx.beginPath();
                ctx.moveTo(x, popupY + radius);
                ctx.lineTo(x - 4, y + 8);
                ctx.lineTo(x + 4, y + 8);
                ctx.closePath();
                ctx.fillStyle = popup.color;
                ctx.fill();
            }
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
    
    drawLinesOnly() {
        if (!this.overlayContext) return;
        
        const canvas = this.overlayCanvas;
        const ctx = this.overlayContext;
        
        // Очищаем только область линий
        const rect = this.getRenderedImageRect();
        if (!rect || rect.w <= 0 || rect.h <= 0) return;
        
        // Отрисовка вертикальных линий
        try {
            const leftX = (window.__lines && typeof window.__lines.left_x === 'number') ? window.__lines.left_x : 0.25;
            const rightX = (window.__lines && typeof window.__lines.right_x === 'number') ? window.__lines.right_x : 0.75;
            const x1 = rect.x + rect.w * leftX;
            const x2 = rect.x + rect.w * rightX;
            
            // Фон линий (более заметный)
            ctx.fillStyle = 'rgba(44,123,229,0.15)';
            ctx.fillRect(x1 - 3, rect.y, 6, rect.h);
            ctx.fillStyle = 'rgba(81,207,102,0.15)';
            ctx.fillRect(x2 - 3, rect.y, 6, rect.h);
            
            // Сами линии (толще и ярче)
            ctx.lineWidth = 3;
            ctx.strokeStyle = 'rgba(44,123,229,0.95)';
            ctx.beginPath();
            ctx.moveTo(x1, rect.y);
            ctx.lineTo(x1, rect.y + rect.h);
            ctx.stroke();
            
            ctx.strokeStyle = 'rgba(81,207,102,0.95)';
            ctx.beginPath();
            ctx.moveTo(x2, rect.y);
            ctx.lineTo(x2, rect.y + rect.h);
            ctx.stroke();
            
            // Индикаторы направления (стрелки)
            const arrowSize = 8;
            const arrowY = rect.y + 20;
            
            // Стрелка для левой линии (вправо)
            ctx.fillStyle = 'rgba(44,123,229,0.95)';
            ctx.beginPath();
            ctx.moveTo(x1, arrowY);
            ctx.lineTo(x1 + arrowSize, arrowY - arrowSize/2);
            ctx.lineTo(x1 + arrowSize, arrowY + arrowSize/2);
            ctx.closePath();
            ctx.fill();
            
            // Стрелка для правой линии (влево)
            ctx.fillStyle = 'rgba(81,207,102,0.95)';
            ctx.beginPath();
            ctx.moveTo(x2, arrowY);
            ctx.lineTo(x2 - arrowSize, arrowY - arrowSize/2);
            ctx.lineTo(x2 - arrowSize, arrowY + arrowSize/2);
            ctx.closePath();
            ctx.fill();
            
            // Кружочки на линиях (больше и ярче)
            ctx.fillStyle = 'rgba(44,123,229,0.95)';
            ctx.beginPath();
            ctx.arc(x1, rect.y + 10, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            ctx.fillStyle = 'rgba(81,207,102,0.95)';
            ctx.beginPath();
            ctx.arc(x2, rect.y + 10, 5, 0, Math.PI * 2);
            ctx.fill();
            ctx.strokeStyle = '#ffffff';
            ctx.lineWidth = 2;
            ctx.stroke();
            
            // Подписи линий
            ctx.font = 'bold 12px system-ui, Segoe UI, Arial';
            ctx.fillStyle = 'rgba(44,123,229,0.95)';
            ctx.textAlign = 'center';
            ctx.textBaseline = 'top';
            ctx.fillText('Вход', x1, rect.y + 35);
            
            ctx.fillStyle = 'rgba(81,207,102,0.95)';
            ctx.fillText('Выход', x2, rect.y + 35);
            
        } catch (error) {
            console.warn('⚠️ Ошибка отрисовки линий:', error);
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
        // Проверяем что overlay включен
        if (!this.overlayEnabled) {
            console.log('🚫 overlayTick пропущен - overlay отключен');
            return;
        }
        
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
        
        // Продолжаем луп только если overlay включен
        if (this.overlayEnabled) {
            this.overlayRaf = requestAnimationFrame(() => this.overlayTick());
        } else {
            this.overlayRaf = null;
        }
    }
    
    handleCrossings(crossings, linePositions = null) {
        // Обрабатываем пересечения линий для создания всплывающих счетчиков
        if (!Array.isArray(crossings) || crossings.length === 0) return;
        
        const now = performance.now();
        
        // Инициализируем массив всплывающих счетчиков
        if (!this.popupCounters) {
            this.popupCounters = [];
        }
        
        // Получаем актуальные позиции линий
        const lines = linePositions || { left_x: 0.25, right_x: 0.75 };
        const leftLineX = Number(lines.left_x || 0.25);
        const rightLineX = Number(lines.right_x || 0.75);
        
        // Отслеживаем последний timestamp для избежания дубликатов
        const prevTs = this.lastPopupTimestamp || 0;
        let maxTs = prevTs;
        
        crossings.forEach(crossing => {
            const tsMs = Number(crossing.ts || 0) * 1000;
            if (tsMs > prevTs) {
                const side = String(crossing.side || 'left');
                const mode = String(crossing.mode || 'enter');
                
                // Правильная логика определения знака:
                // Вход слева (свинья идет вправо) = +1
                // Выход слева (свинья идет влево) = -1  
                // Вход справа (свинья идет влево) = -1
                // Выход справа (свинья идет вправо) = +1
                const isPositive = (side === 'left' && mode === 'enter') || (side === 'right' && mode === 'exit');
                
                const text = isPositive ? '+1' : '-1';
                const color = isPositive ? '#51cf66' : '#ff6b6b';
                
                // Используем точные координаты пересечения без коррекции
                const x = Number(crossing.x || 0.5);
                const y = Number(crossing.y || 0.5);
                
                console.log(`🎯 Пересечение: side=${side}, x=${x.toFixed(3)} (должно быть ${side === 'left' ? '0.25' : '0.75'})`);
                
                this.popupCounters.push({
                    x: x,
                    y: y,
                    text: text,
                    color: color,
                    startTime: now,
                    duration: 1200,
                    riseDistance: 40
                });
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
    
    stopOverlay() {
        console.log('🛑 Остановка overlay лупа');
        
        if (this.overlayRaf) {
            cancelAnimationFrame(this.overlayRaf);
            this.overlayRaf = null;
        }
        
        // Очищаем overlay
        this.clearOverlay();
        
        // Сбрасываем состояние
        this.lastMasks = null;
        this.lastIds = null;
        this.overlayPending = null;
        this.overlayEnabled = false;
    }
    
    startOverlay() {
        console.log('▶️ Запуск overlay лупа');
        
        this.overlayEnabled = true;
        this.startOverlayLoop();
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
    
    setupLineDragging() {
        console.log('🔧 setupLineDragging вызван:', {
            hasOverlayCanvas: !!this.overlayCanvas,
            lineDragInitialized: this.lineDragInitialized,
            overlayEnabled: this.overlayEnabled
        });
        
        if (!this.overlayCanvas || this.lineDragInitialized) {
            console.warn('⚠️ setupLineDragging пропущен:', {
                hasOverlayCanvas: !!this.overlayCanvas,
                lineDragInitialized: this.lineDragInitialized
            });
            return;
        }
        
        this.lineDragInitialized = true;
        this.isDraggingLines = false;
        this.draggingLine = null; // 'left' | 'right'
        
        // Сохраняем ссылки на обработчики для правильного удаления
        this.globalMouseUpHandler = null;
        this.globalMouseMoveHandler = null;
        this.globalTouchEndHandler = null;
        this.globalTouchMoveHandler = null;
        
        const onPos = (e) => {
            const r = this.overlayCanvas.getBoundingClientRect();
            const clientX = e.touches ? e.touches[0].clientX : e.clientX;
            const x = clientX - r.left;
            const rect = this.getRenderedImageRect();
            return Math.max(0, Math.min(1, (x - rect.x) / rect.w));
        };
        
        const onDown = (e) => {
            console.log('🖱️ onDown вызван:', {
                overlayEnabled: this.overlayEnabled,
                eventType: e.type,
                clientX: e.clientX || (e.touches ? e.touches[0].clientX : 'N/A')
            });
            
            if (!this.overlayEnabled) {
                console.warn('⚠️ overlayEnabled = false, пропускаем onDown');
                return;
            }
            
            const nx = onPos(e);
            const lx = (window.__lines && typeof window.__lines.left_x === 'number') ? window.__lines.left_x : 0.25;
            const rx = (window.__lines && typeof window.__lines.right_x === 'number') ? window.__lines.right_x : 0.75;
            
            const distL = Math.abs(nx - lx);
            const distR = Math.abs(nx - rx);
            
            console.log('🎯 Проверка расстояний:', {
                nx: nx.toFixed(3),
                lx: lx.toFixed(3),
                rx: rx.toFixed(3),
                distL: distL.toFixed(3),
                distR: distR.toFixed(3),
                threshold: 0.02
            });
            
            if (distL < 0.02 || distR < 0.02) {
                this.draggingLine = (distL < distR) ? 'left' : 'right';
                this.isDraggingLines = true;
                this.overlayCanvas.style.cursor = 'ew-resize';
                e.preventDefault();
                console.log('🎯 Начато перетаскивание линии:', this.draggingLine);
                
                // Добавляем глобальные обработчики для корректного завершения перетаскивания
                document.addEventListener('mouseup', this.globalMouseUpHandler);
                document.addEventListener('touchend', this.globalTouchEndHandler);
                document.addEventListener('mousemove', this.globalMouseMoveHandler);
                document.addEventListener('touchmove', this.globalTouchMoveHandler);
            } else {
                console.log('❌ Слишком далеко от линий');
            }
        };
        
        const onMove = (e) => {
            const nx = onPos(e);
            let lx = (window.__lines && typeof window.__lines.left_x === 'number') ? window.__lines.left_x : 0.25;
            let rx = (window.__lines && typeof window.__lines.right_x === 'number') ? window.__lines.right_x : 0.75;
            
            const nearLine = (Math.abs(nx - lx) < 0.02) || (Math.abs(nx - rx) < 0.02);
            
            if (!this.draggingLine) {
                // Подсказка курсора при наведении
                this.overlayCanvas.style.cursor = nearLine ? 'ew-resize' : 'crosshair';
                return;
            }
            
            // Перетаскивание
            if (this.draggingLine === 'left') lx = nx; else rx = nx;
            
            // Предотвращаем пересечение линий
            if (lx > rx) { const t = lx; lx = rx; rx = t; }
            
            const minGap = 0.05;
            if ((rx - lx) < minGap) {
                const mid = (lx + rx) / 2;
                lx = Math.max(0, mid - minGap/2);
                rx = Math.min(1, mid + minGap/2);
            }
            
            window.__lines = { left_x: lx, right_x: rx };
            this.overlayCanvas.style.cursor = 'ew-resize';
            
            // Перерисовываем только линии, не маски
            this.drawLinesOnly();
            
            e.preventDefault();
        };
        
        // Сохраняем ссылку на onMove для глобального обработчика
        this.globalMouseMoveHandler = onMove;
        
        const onUp = async (e) => {
            if (!this.draggingLine) return;
            
            console.log('🎯 Завершено перетаскивание линии:', this.draggingLine);
            
            try {
                const { left_x, right_x } = window.__lines || {};
                const response = await fetch('/api/lines', {
                    method: 'POST',
                    headers: { 'Content-Type': 'application/json' },
                    body: JSON.stringify({ left_x, right_x })
                });
                
                if (response.ok) {
                    console.log('✅ Позиции линий сохранены:', { left_x, right_x });
                } else {
                    console.warn('⚠️ Ошибка сохранения позиций линий');
                }
            } catch (error) {
                console.warn('⚠️ Ошибка отправки позиций линий:', error);
            }
            
            this.draggingLine = null;
            this.overlayCanvas.style.cursor = 'crosshair';
            this.isDraggingLines = false;
            
            // Удаляем глобальные обработчики
            if (this.globalMouseUpHandler) {
                document.removeEventListener('mouseup', this.globalMouseUpHandler);
                this.globalMouseUpHandler = null;
            }
            if (this.globalTouchEndHandler) {
                document.removeEventListener('touchend', this.globalTouchEndHandler);
                this.globalTouchEndHandler = null;
            }
            if (this.globalMouseMoveHandler) {
                document.removeEventListener('mousemove', this.globalMouseMoveHandler);
                this.globalMouseMoveHandler = null;
            }
            if (this.globalTouchMoveHandler) {
                document.removeEventListener('touchmove', this.globalTouchMoveHandler);
                this.globalTouchMoveHandler = null;
            }
        };
        
        // Сохраняем ссылку на onUp для глобального обработчика
        this.globalMouseUpHandler = onUp;
        this.globalTouchEndHandler = onUp;
        this.globalTouchMoveHandler = onMove;
        
        // Добавляем только обработчики начала перетаскивания
        this.overlayCanvas.addEventListener('mousedown', onDown);
        this.overlayCanvas.addEventListener('touchstart', onDown, { passive: false });
        
        // Обработчик для показа курсора при наведении (только если не перетаскиваем)
        this.overlayCanvas.addEventListener('mousemove', (e) => {
            if (!this.draggingLine) {
                const nx = onPos(e);
                const lx = (window.__lines && typeof window.__lines.left_x === 'number') ? window.__lines.left_x : 0.25;
                const rx = (window.__lines && typeof window.__lines.right_x === 'number') ? window.__lines.right_x : 0.75;
                const nearLine = (Math.abs(nx - lx) < 0.02) || (Math.abs(nx - rx) < 0.02);
                this.overlayCanvas.style.cursor = nearLine ? 'ew-resize' : 'crosshair';
            }
        });
        
        // Обработчик для сброса курсора при покидании canvas (только если не перетаскиваем)
        this.overlayCanvas.addEventListener('mouseleave', () => {
            if (!this.draggingLine) {
                this.overlayCanvas.style.cursor = 'crosshair';
            }
        });
        
        console.log('✅ Обработчики перетаскивания линий настроены');
    }
}