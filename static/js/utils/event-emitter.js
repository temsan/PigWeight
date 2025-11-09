/**
 * Простая реализация EventEmitter для модульной архитектуры
 */

export class EventEmitter {
    constructor() {
        this.events = {};
    }
    
    on(event, listener) {
        if (!this.events[event]) {
            this.events[event] = [];
        }
        this.events[event].push(listener);
        return this;
    }
    
    once(event, listener) {
        const onceWrapper = (...args) => {
            listener.apply(this, args);
            this.off(event, onceWrapper);
        };
        return this.on(event, onceWrapper);
    }
    
    off(event, listenerToRemove) {
        if (!this.events[event]) return this;
        
        this.events[event] = this.events[event].filter(
            listener => listener !== listenerToRemove
        );
        
        return this;
    }
    
    emit(event, ...args) {
        if (!this.events[event]) return false;
        
        this.events[event].forEach(listener => {
            try {
                listener.apply(this, args);
            } catch (error) {
                // Логируем только критичные ошибки (не связанные с Chart.js или временными проблемами)
                const errorMessage = error?.message || String(error);
                if (!errorMessage.includes('filter') && 
                    !errorMessage.includes('isPluginEnabled') &&
                    !errorMessage.includes('Chart')) {
                    console.error(`Ошибка в обработчике события '${event}':`, error);
                } else {
                    // Тихо логируем известные проблемы с Chart.js
                    console.debug(`Предупреждение в обработчике события '${event}':`, errorMessage);
                }
            }
        });
        
        return true;
    }
    
    removeAllListeners(event) {
        if (event) {
            delete this.events[event];
        } else {
            this.events = {};
        }
        return this;
    }
    
    listenerCount(event) {
        return this.events[event] ? this.events[event].length : 0;
    }
    
    eventNames() {
        return Object.keys(this.events);
    }
}