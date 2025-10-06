/**
 * Главный модуль приложения PigWeight
 * Инициализация и координация всех компонентов
 */

import { UIManager } from './ui-manager.js';
import { VideoManager } from './video-manager.js?v=3';
import { WebSocketManager } from './websocket-manager.js';
import { ChartManager } from './chart-manager.js';
import { JournalManager } from './journal-manager.js';

class PigWeightApp {
    constructor() {
        this.ui = null;
        this.video = null;
        this.websocket = null;
        this.chart = null;
        this.journal = null;
        
        this.initialized = false;
    }

    async init() {
        if (this.initialized) return;
        
        console.log('🚀 Инициализация PigWeight App...');
        
        try {
            // Инициализируем компоненты в правильном порядке
            this.ui = new UIManager();
            this.video = new VideoManager();
            this.websocket = new WebSocketManager();
            this.chart = new ChartManager();
            this.journal = new JournalManager();
            
            // Инициализируем каждый компонент
            await this.ui.init();
            await this.video.init();
            await this.websocket.init();
            await this.chart.init();
            await this.journal.init();
            
            // Связываем компоненты
            this.setupComponentInteractions();
            
            this.initialized = true;
            console.log('✅ PigWeight App инициализирован');
            
        } catch (error) {
            console.error('❌ Ошибка инициализации PigWeight App:', error);
            throw error;
        }
    }
    
    setupComponentInteractions() {
        // WebSocket -> UI обновления
        this.websocket.on('count_update', (data) => {
            this.ui.updateCounters(data);
            this.chart.updateChart(data);
            
            // Передаем данные масок в VideoManager для overlay
            if (data.debug && (data.debug.masks || data.debug.ids)) {
                this.video.scheduleOverlay(data.debug.masks, data.debug.ids);
            }
        });
        
        // UI -> Video управление
        this.ui.on('video_control', (action, data) => {
            this.video.handleControl(action, data);
        });
        
        // UI -> Video передача пересечений для всплывающих счетчиков
        this.ui.on('crossings_update', (crossings) => {
            this.video.handleCrossings(crossings);
        });
        
        // Video -> WebSocket стрим
        this.video.on('stream_change', (streamId) => {
            this.websocket.switchStream(streamId);
        });
        
        // WebSocket подключение при запуске стрима
        this.video.on('stream_start', (streamId) => {
            console.log(`🔔 Подключаем WebSocket к стриму: ${streamId}`);
            this.websocket.connect(streamId);
        });
        
        // UI -> Journal операции
        this.ui.on('journal_action', (action, data) => {
            this.journal.handleAction(action, data);
        });
    }
    
    // Публичные методы для глобального доступа
    switchPanel(panelName) {
        return this.ui.switchPanel(panelName);
    }
    
    updateCounters(data) {
        return this.ui.updateCounters(data);
    }
    
    // Cleanup при выгрузке страницы
    destroy() {
        if (this.websocket) this.websocket.destroy();
        if (this.video) this.video.destroy();
        if (this.chart) this.chart.destroy();
        if (this.journal) this.journal.destroy();
        if (this.ui) this.ui.destroy();
    }
}

// Создаем глобальный экземпляр приложения
window.pigWeightApp = new PigWeightApp();

// Автоинициализация при загрузке DOM
document.addEventListener('DOMContentLoaded', async () => {
    try {
        await window.pigWeightApp.init();
    } catch (error) {
        console.error('Не удалось инициализировать приложение:', error);
    }
});

// Cleanup при выгрузке
window.addEventListener('beforeunload', () => {
    if (window.pigWeightApp) {
        window.pigWeightApp.destroy();
    }
});

// Экспортируем для использования в HTML (для обратной совместимости)
window.switchPanel = (panelName) => {
    if (window.pigWeightApp && window.pigWeightApp.ui) {
        return window.pigWeightApp.ui.switchPanel(panelName);
    }
};

export default PigWeightApp;