/**
 * Менеджер событий для системы PigWeight
 * Управляет отображением и взаимодействием с журналом событий
 */

class EventsManager {
    constructor() {
        this.currentStream = null;
        this.events = [];
        this.eventTypes = {
            'line_crossing': { name: 'Пересечение линии', icon: '🚶', color: '#4CAF50' },
            'peak_count': { name: 'Пик количества', icon: '📈', color: '#FF9800' },
            'activity_spike': { name: 'Всплеск активности', icon: '⚡', color: '#F44336' }
        };
        
        this.refreshInterval = null;
        this.autoRefresh = true;
        this.refreshRate = 5000; // 5 секунд
        
        this.initializeUI();
        this.bindEvents();
    }
    
    initializeUI() {
        // Создаем контейнер для событий если его нет
        if (!document.getElementById('events-container')) {
            this.createEventsContainer();
        }
        
        this.updateUI();
    }
    
    createEventsContainer() {
        const container = document.createElement('div');
        container.id = 'events-container';
        container.className = 'events-container';
        container.innerHTML = `
            <div class="events-header">
                <h3>📋 Журнал событий</h3>
                <div class="events-controls">
                    <select id="event-type-filter" class="form-control">
                        <option value="">Все события</option>
                        <option value="line_crossing">Пересечения линий</option>
                        <option value="peak_count">Пики количества</option>
                        <option value="activity_spike">Всплески активности</option>
                    </select>
                    <select id="event-time-filter" class="form-control">
                        <option value="">Все время</option>
                        <option value="1">Последний час</option>
                        <option value="6">Последние 6 часов</option>
                        <option value="24">Последние 24 часа</option>
                    </select>
                    <button id="refresh-events" class="btn btn-primary">🔄 Обновить</button>
                    <button id="export-events" class="btn btn-secondary">📥 Экспорт</button>
                </div>
            </div>
            <div class="events-stats" id="events-stats"></div>
            <div class="events-list" id="events-list">
                <div class="loading">Загрузка событий...</div>
            </div>
        `;
        
        // Добавляем в основной контейнер или создаем новую панель
        const mainContainer = document.querySelector('.container-fluid') || document.body;
        mainContainer.appendChild(container);
    }
    
    bindEvents() {
        // Фильтры
        document.addEventListener('change', (e) => {
            if (e.target.id === 'event-type-filter' || e.target.id === 'event-time-filter') {
                this.loadEvents();
            }
        });
        
        // Кнопки
        document.addEventListener('click', (e) => {
            if (e.target.id === 'refresh-events') {
                this.loadEvents();
            } else if (e.target.id === 'export-events') {
                this.exportEvents();
            } else if (e.target.classList.contains('event-frame-btn')) {
                this.showEventFrame(e.target.dataset.eventId);
            }
        });
        
        // Автообновление
        if (this.autoRefresh) {
            this.startAutoRefresh();
        }
    }
    
    setCurrentStream(streamId) {
        if (this.currentStream !== streamId) {
            this.currentStream = streamId;
            this.loadEvents();
            this.loadStats();
        }
    }
    
    async loadEvents() {
        if (!this.currentStream) {
            const eventsContainer = document.getElementById('eventsList');
            if (eventsContainer) {
                eventsContainer.innerHTML = '<div class="loading-message">Выберите поток для просмотра событий</div>';
            }
            return;
        }
        
        try {
            const typeFilter = document.getElementById('eventTypeFilter')?.value || '';
            const timeFilter = document.getElementById('eventTimeFilter')?.value || '';
            
            let url = `/api/events/${this.currentStream}?limit=100`;
            if (typeFilter) url += `&event_type=${typeFilter}`;
            if (timeFilter) url += `&since_hours=${timeFilter}`;
            
            const response = await fetch(url);
            const data = await response.json();
            
            if (data.success) {
                this.events = data.events;
                this.renderEvents();
            } else {
                throw new Error(data.message || 'Ошибка загрузки событий');
            }
        } catch (error) {
            console.error('Ошибка загрузки событий:', error);
            this.showMessage('Ошибка загрузки событий: ' + error.message, 'error');
        }
    }
    
    async loadStats() {
        if (!this.currentStream) return;
        
        try {
            const response = await fetch(`/api/events/${this.currentStream}/stats`);
            const data = await response.json();
            
            if (data.success) {
                this.renderStats(data.statistics);
            }
        } catch (error) {
            console.error('Ошибка загрузки статистики:', error);
        }
    }
    
    renderStats(stats) {
        // Обновляем отдельные элементы статистики
        const totalEventsEl = document.getElementById('totalEvents');
        const peakCountEl = document.getElementById('peakCount');
        const lineCrossingsEl = document.getElementById('lineCrossings');
        const activitySpikesEl = document.getElementById('activitySpikes');
        
        if (totalEventsEl) totalEventsEl.textContent = stats.total_events || 0;
        if (peakCountEl) peakCountEl.textContent = stats.peak_count || 0;
        
        // Обновляем счетчики по типам событий
        const eventTypes = stats.event_types || {};
        if (lineCrossingsEl) lineCrossingsEl.textContent = eventTypes.line_crossing || 0;
        if (activitySpikesEl) activitySpikesEl.textContent = eventTypes.activity_spike || 0;
    }
    
    renderEvents() {
        const eventsContainer = document.getElementById('eventsList');
        if (!eventsContainer) return;
        
        if (this.events.length === 0) {
            eventsContainer.innerHTML = '<div class="loading-message">Событий не найдено</div>';
            return;
        }
        
        const eventsHtml = this.events.map(event => this.renderEvent(event)).join('');
        eventsContainer.innerHTML = eventsHtml;
    }
    
    renderEvent(event) {
        const typeInfo = this.eventTypes[event.event_type] || { 
            name: event.event_type, 
            icon: '📝', 
            color: '#666' 
        };
        
        const datetime = new Date(event.datetime).toLocaleString('ru-RU');
        const hasFrame = event.frame_path !== null;
        
        return `
            <div class="event-item" style="border-left: 4px solid ${typeInfo.color}">
                <div class="event-header">
                    <div class="event-type">
                        <span class="event-icon">${typeInfo.icon}</span>
                        <span class="event-name">${typeInfo.name}</span>
                    </div>
                    <div class="event-time">${datetime}</div>
                </div>
                <div class="event-details">
                    <div class="event-metrics">
                        <span class="metric">
                            <strong>Количество:</strong> ${event.pig_count}
                        </span>
                        <span class="metric">
                            <strong>Уверенность:</strong> ${(event.confidence * 100).toFixed(1)}%
                        </span>
                    </div>
                    ${hasFrame ? `
                        <button class="event-frame-btn btn btn-sm btn-outline-primary" 
                                data-event-id="${event.event_id}">
                            🖼️ Показать кадр
                        </button>
                    ` : ''}
                </div>
                ${event.metadata ? `
                    <div class="event-metadata">
                        <details>
                            <summary>Дополнительная информация</summary>
                            <pre>${JSON.stringify(event.metadata, null, 2)}</pre>
                        </details>
                    </div>
                ` : ''}
            </div>
        `;
    }
    
    async showEventFrame(eventId) {
        if (!this.currentStream || !eventId) return;
        
        try {
            const imageUrl = `/api/events/${this.currentStream}/frame/${eventId}`;
            
            // Создаем модальное окно для отображения кадра
            const modal = document.createElement('div');
            modal.className = 'event-frame-modal';
            modal.innerHTML = `
                <div class="modal-backdrop" onclick="this.parentElement.remove()"></div>
                <div class="modal-content">
                    <div class="modal-header">
                        <h4>Кадр события ${eventId}</h4>
                        <button class="modal-close" onclick="this.closest('.event-frame-modal').remove()">×</button>
                    </div>
                    <div class="modal-body">
                        <img src="${imageUrl}" alt="Event frame" class="event-frame-image" />
                    </div>
                </div>
            `;
            
            document.body.appendChild(modal);
            
        } catch (error) {
            console.error('Ошибка отображения кадра:', error);
            this.showMessage('Ошибка загрузки кадра: ' + error.message, 'error');
        }
    }
    
    async exportEvents() {
        if (!this.currentStream) {
            this.showMessage('Выберите поток для экспорта', 'warning');
            return;
        }
        
        try {
            const typeFilter = document.getElementById('eventTypeFilter')?.value || '';
            const timeFilter = document.getElementById('eventTimeFilter')?.value || '';
            
            let url = `/api/events/${this.currentStream}/export?format=csv`;
            if (typeFilter) url += `&event_type=${typeFilter}`;
            if (timeFilter) url += `&since_hours=${timeFilter}`;
            
            // Создаем скрытую ссылку для скачивания
            const link = document.createElement('a');
            link.href = url;
            link.download = `${this.currentStream}_events.csv`;
            document.body.appendChild(link);
            link.click();
            document.body.removeChild(link);
            
            this.showMessage('Экспорт начат', 'success');
            
        } catch (error) {
            console.error('Ошибка экспорта:', error);
            this.showMessage('Ошибка экспорта: ' + error.message, 'error');
        }
    }
    
    showMessage(message, type = 'info') {
        // Уведомления отключены - только в консоль
        console.log(`[${type.toUpperCase()}] ${message}`);
        return;
    }
    
    startAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
        }
        
        this.refreshInterval = setInterval(() => {
            if (this.currentStream && this.autoRefresh) {
                this.loadEvents();
                this.loadStats();
            }
        }, this.refreshRate);
    }
    
    stopAutoRefresh() {
        if (this.refreshInterval) {
            clearInterval(this.refreshInterval);
            this.refreshInterval = null;
        }
    }
    
    setAutoRefresh(enabled) {
        this.autoRefresh = enabled;
        if (enabled) {
            this.startAutoRefresh();
        } else {
            this.stopAutoRefresh();
        }
    }
    
    updateUI() {
        // Обновляем интерфейс при изменении состояния
        const container = document.getElementById('events-container');
        if (container) {
            container.style.display = this.currentStream ? 'block' : 'none';
        }
    }
    
    destroy() {
        this.stopAutoRefresh();
        const container = document.getElementById('events-container');
        if (container && container.parentElement) {
            container.parentElement.removeChild(container);
        }
    }
}

// Глобальный экземпляр менеджера событий
window.eventsManager = new EventsManager();