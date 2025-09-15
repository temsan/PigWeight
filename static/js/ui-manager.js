/**
 * Менеджер пользовательского интерфейса
 * Управляет панелями, счетчиками, анимациями
 */

import { EventEmitter } from './utils/event-emitter.js';

export class UIManager extends EventEmitter {
    constructor() {
        super();
        
        // Элементы интерфейса
        this.elements = {};
        
        // Состояние счетчиков для throttling
        this.lastCounterUpdate = 0;
        this.lastLeftIn = 0;
        this.lastRightIn = 0;
        this.lastTotalCrossings = 0;
        
        // Throttling для UI обновлений
        this.uiUpdateThrottle = 100; // ms
    }
    
    async init() {
        console.log('🎨 Инициализация UI Manager...');
        
        // Кэшируем элементы DOM
        this.cacheElements();
        
        // Настраиваем обработчики событий
        this.setupEventHandlers();
        
        // Инициализируем состояние
        this.initializeState();
        
        console.log('✅ UI Manager инициализирован');
    }
    
    cacheElements() {
        // Счетчики
        this.elements.leftIn = document.getElementById('leftIn');
        this.elements.rightIn = document.getElementById('rightIn');
        this.elements.totalCrossings = document.getElementById('totalCrossings');
        
        // Акт взвешивания
        this.elements.actPeak = document.getElementById('actPeak');
        this.elements.actDur = document.getElementById('actDur');
        
        // Панели
        this.elements.panels = document.querySelectorAll('.panel-content');
        this.elements.tabs = document.querySelectorAll('.journal-tab-btn');
        
        // Статус
        this.elements.streamStatus = document.getElementById('streamStatus');
        
        // Формы
        this.elements.manualCount = document.getElementById('manualCount');
        this.elements.manualTotalWeight = document.getElementById('manualTotalWeight');
        this.elements.saveManualBtn = document.getElementById('saveManualBtn');
    }
    
    setupEventHandlers() {
        // Обработчики форм
        if (this.elements.saveManualBtn) {
            this.elements.saveManualBtn.addEventListener('click', () => {
                this.handleSaveManual();
            });
        }
        
        // Обработчики вкладок (делегирование событий)
        document.addEventListener('click', (e) => {
            if (e.target.classList.contains('journal-tab-btn')) {
                const panelName = this.extractPanelName(e.target);
                if (panelName) {
                    this.switchPanel(panelName);
                }
            }
        });
    }
    
    initializeState() {
        // Устанавливаем начальные значения
        this.updateCounterElement(this.elements.leftIn, 0);
        this.updateCounterElement(this.elements.rightIn, 0);
        this.updateCounterElement(this.elements.totalCrossings, 0);
    }
    
    // Переключение панелей
    switchPanel(panelName) {
        console.log(`🔄 Переключение на панель: ${panelName}`);
        
        // Скрываем все панели
        this.elements.panels.forEach(panel => {
            panel.style.display = 'none';
        });
        
        // Убираем активный класс со всех вкладок
        this.elements.tabs.forEach(tab => {
            tab.classList.remove('active');
        });
        
        // Показываем нужную панель
        const targetPanel = document.getElementById(`${panelName}Panel`);
        if (targetPanel) {
            targetPanel.style.display = 'block';
        }
        
        // Активируем нужную вкладку
        const activeTab = document.querySelector(`.journal-tab-btn[onclick*="${panelName}"]`);
        if (activeTab) {
            activeTab.classList.add('active');
        }
        
        // Загружаем данные для панели при первом открытии
        if (panelName === 'logs' && !targetPanel.dataset.loaded) {
            this.emit('journal_action', 'load_logs');
            targetPanel.dataset.loaded = 'true';
        }
        
        this.emit('panel_switched', panelName);
    }
    
    // Обновление счетчиков с throttling
    updateCounters(data) {
        const now = performance.now();
        
        // Throttling для UI обновлений
        if (now - this.lastCounterUpdate < this.uiUpdateThrottle) {
            return;
        }
        this.lastCounterUpdate = now;
        
        if (data.debug && data.debug.flow) {
            const flow = data.debug.flow;
            const leftInCount = Number(flow.left_in ?? 0);
            const rightInCount = Number(flow.right_in ?? 0);
            const totalCrossingsCount = Number(flow.total_crossings ?? 0);
            
            // Обновляем только изменившиеся значения
            if (leftInCount !== this.lastLeftIn) {
                this.updateCounterElement(this.elements.leftIn, leftInCount);
                this.animateCounterChange(this.elements.leftIn, leftInCount, this.lastLeftIn);
                this.lastLeftIn = leftInCount;
            }
            
            if (rightInCount !== this.lastRightIn) {
                this.updateCounterElement(this.elements.rightIn, rightInCount);
                this.animateCounterChange(this.elements.rightIn, rightInCount, this.lastRightIn);
                this.lastRightIn = rightInCount;
            }
            
            if (totalCrossingsCount !== this.lastTotalCrossings) {
                this.updateCounterElement(this.elements.totalCrossings, totalCrossingsCount);
                this.animateCounterChange(this.elements.totalCrossings, totalCrossingsCount, this.lastTotalCrossings);
                this.lastTotalCrossings = totalCrossingsCount;
            }
            
            // Обновляем статус только при изменении
            this.updateStatus(leftInCount, rightInCount, totalCrossingsCount, data.debug.model);
        }
        
        // Обновляем данные акта взвешивания
        if (data.debug && data.debug.act) {
            this.updateActData(data.debug.act);
        }
        
        // Обновляем форму
        if (data.count !== undefined) {
            this.updateManualForm(data.count);
        }
    }
    
    updateCounterElement(element, value) {
        if (element) {
            element.textContent = String(value);
        }
    }
    
    animateCounterChange(element, newValue, lastValue) {
        if (!element) return;
        
        if (newValue > lastValue) {
            // Увеличение - зеленый цвет
            element.style.color = '#51cf66';
            setTimeout(() => {
                element.style.color = '';
            }, 500);
        } else if (newValue < lastValue) {
            // Уменьшение - красный цвет
            element.style.color = '#ff6b6b';
            setTimeout(() => {
                element.style.color = '';
            }, 500);
        }
    }
    
    updateStatus(leftIn, rightIn, total, modelInfo) {
        if (!this.elements.streamStatus) return;
        
        let modelText = '';
        if (modelInfo && modelInfo.name) {
            const device = modelInfo.device ? `, ${modelInfo.device}${modelInfo.half ? '/fp16' : ''}` : '';
            modelText = ` | GPU: ${device || 'cpu'} | Модель: ${modelInfo.name}`;
        }
        
        const newStatus = `Live | Всего пересечений: ${total} | Слева: ${leftIn} | Справа: ${rightIn}${modelText}`;
        
        // Обновляем только если изменилось
        if (this.elements.streamStatus.textContent !== newStatus) {
            this.elements.streamStatus.textContent = newStatus;
        }
    }
    
    updateActData(actData) {
        if (this.elements.actPeak) {
            this.elements.actPeak.textContent = String(Number(actData.peak_concurrent || 0));
        }
        if (this.elements.actDur) {
            this.elements.actDur.textContent = String((Number(actData.duration_sec || 0)).toFixed(1));
        }
    }
    
    updateManualForm(count) {
        if (this.elements.manualCount) {
            this.elements.manualCount.value = count;
        }
        
        // Пересчитываем средний вес если есть функция
        if (typeof window.recalcAvgWeight === 'function') {
            window.recalcAvgWeight();
        }
    }
    
    handleSaveManual() {
        const count = this.elements.manualCount?.value || 0;
        const weight = this.elements.manualTotalWeight?.value || 0;
        
        this.emit('journal_action', 'save_manual', {
            count: Number(count),
            weight: Number(weight)
        });
    }
    
    extractPanelName(tabElement) {
        const onclick = tabElement.getAttribute('onclick');
        if (onclick) {
            const match = onclick.match(/switchPanel\\('([^']+)'\\)/);
            return match ? match[1] : null;
        }
        return null;
    }
    
    // Сброс счетчиков
    resetCounters() {
        this.lastLeftIn = 0;
        this.lastRightIn = 0;
        this.lastTotalCrossings = 0;
        
        this.updateCounterElement(this.elements.leftIn, 0);
        this.updateCounterElement(this.elements.rightIn, 0);
        this.updateCounterElement(this.elements.totalCrossings, 0);
    }
    
    destroy() {
        // Очистка обработчиков событий и ресурсов
        this.removeAllListeners();
        this.elements = {};
    }
}