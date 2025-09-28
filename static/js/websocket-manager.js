/**
 * Менеджер WebSocket соединений
 * Управляет подключениями, обработкой сообщений, переподключениями
 */

import { EventEmitter } from './utils/event-emitter.js';

export class WebSocketManager extends EventEmitter {
    constructor() {
        super();
        
        this.ws = null;
        this.currentStreamId = null;
        this.reconnectAttempts = 0;
        this.maxReconnectAttempts = 5;
        this.reconnectDelay = 1000; // ms
        this.isConnecting = false;
        
        // Throttling для обработки сообщений
        this.lastMessageTime = 0;
        this.messageThrottle = 50; // ms
    }
    
    async init() {
        console.log('🔌 Инициализация WebSocket Manager...');
        
        // Настраиваем обработчики переподключения
        this.setupReconnectHandlers();
        
        console.log('✅ WebSocket Manager инициализирован');
    }
    
    setupReconnectHandlers() {
        // Переподключение при потере соединения с интернетом
        window.addEventListener('online', () => {
            console.log('🌐 Соединение восстановлено, переподключаемся...');
            if (this.currentStreamId) {
                this.connect(this.currentStreamId);
            }
        });
        
        // Обработка потери соединения
        window.addEventListener('offline', () => {
            console.log('📡 Соединение потеряно');
            this.emit('connection_lost');
        });
    }
    
    async connect(streamId) {
        if (this.isConnecting) {
            console.log('⏳ Уже подключаемся...');
            return;
        }
        
        if (this.ws && this.ws.readyState === WebSocket.OPEN && this.currentStreamId === streamId) {
            console.log('✅ Уже подключены к', streamId);
            return;
        }
        
        this.isConnecting = true;
        this.currentStreamId = streamId;
        
        try {
            // Закрываем существующее соединение
            if (this.ws) {
                this.ws.close();
            }
            
            console.log(`🔌 Подключаемся к WebSocket для стрима: ${streamId}`);
            
            const protocol = location.protocol === 'https:' ? 'wss' : 'ws';
            const wsUrl = `${protocol}://${location.host}/ws/count?id=${streamId}`;
            
            this.ws = new WebSocket(wsUrl);
            
            this.ws.onopen = () => {
                console.log(`✅ WebSocket подключен к ${streamId}`);
                this.reconnectAttempts = 0;
                this.isConnecting = false;
                this.emit('connected', streamId);
            };
            
            this.ws.onmessage = (event) => {
                this.handleMessage(event);
            };
            
            this.ws.onclose = (event) => {
                console.log(`🔌 WebSocket отключен от ${streamId}:`, event.code, event.reason);
                this.isConnecting = false;
                this.emit('disconnected', streamId);
                
                // Автоматическое переподключение
                if (event.code !== 1000 && this.reconnectAttempts < this.maxReconnectAttempts) {
                    this.scheduleReconnect();
                }
            };
            
            this.ws.onerror = (error) => {
                console.error(`❌ Ошибка WebSocket для ${streamId}:`, error);
                this.isConnecting = false;
                this.emit('error', error);
            };
            
        } catch (error) {
            console.error('❌ Ошибка создания WebSocket:', error);
            this.isConnecting = false;
            this.emit('error', error);
        }
    }
    
    handleMessage(event) {
        const now = performance.now();
        
        // Throttling сообщений для производительности
        if (now - this.lastMessageTime < this.messageThrottle) {
            return;
        }
        this.lastMessageTime = now;
        
        try {
            const data = JSON.parse(event.data);
            
            // Логируем получение масок для отладки
            if (data.debug && data.debug.masks && data.debug.masks.length > 0) {
                console.log(`🎭 Получены маски: ${data.debug.masks.length} шт.`);
            }
            
            if (data.type === 'count_update') {
                this.emit('count_update', data);
            } else {
                this.emit('message', data);
            }
            
        } catch (error) {
            console.error('❌ Ошибка парсинга WebSocket сообщения:', error);
        }
    }
    
    scheduleReconnect() {
        this.reconnectAttempts++;
        const delay = this.reconnectDelay * Math.pow(2, this.reconnectAttempts - 1); // Exponential backoff
        
        console.log(`🔄 Переподключение через ${delay}ms (попытка ${this.reconnectAttempts}/${this.maxReconnectAttempts})`);
        
        setTimeout(() => {
            if (this.currentStreamId) {
                this.connect(this.currentStreamId);
            }
        }, delay);
    }
    
    switchStream(streamId) {
        if (this.currentStreamId === streamId) {
            return;
        }
        
        console.log(`🔄 Переключение стрима: ${this.currentStreamId} -> ${streamId}`);
        this.connect(streamId);
    }
    
    send(data) {
        if (this.ws && this.ws.readyState === WebSocket.OPEN) {
            this.ws.send(JSON.stringify(data));
        } else {
            console.warn('⚠️ WebSocket не подключен, сообщение не отправлено:', data);
        }
    }
    
    disconnect() {
        if (this.ws) {
            this.ws.close(1000, 'Отключение по запросу пользователя');
            this.ws = null;
        }
        this.currentStreamId = null;
        this.reconnectAttempts = 0;
    }
    
    getConnectionState() {
        if (!this.ws) return 'disconnected';
        
        switch (this.ws.readyState) {
            case WebSocket.CONNECTING: return 'connecting';
            case WebSocket.OPEN: return 'connected';
            case WebSocket.CLOSING: return 'closing';
            case WebSocket.CLOSED: return 'disconnected';
            default: return 'unknown';
        }
    }
    
    destroy() {
        this.disconnect();
        this.removeAllListeners();
    }
}