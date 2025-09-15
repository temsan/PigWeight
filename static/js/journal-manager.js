/**
 * Менеджер журнала взвешиваний
 * Управляет записями, экспортом, сверкой данных
 */

import { EventEmitter } from './utils/event-emitter.js';

export class JournalManager extends EventEmitter {
    constructor() {
        super();
        
        this.records = [];
        this.isLoading = false;
        
        // Элементы форм
        this.elements = {};
    }
    
    async init() {
        console.log('📋 Инициализация Journal Manager...');
        
        // Кэшируем элементы
        this.cacheElements();
        
        // Настраиваем обработчики
        this.setupEventHandlers();
        
        // Инициализируем формы
        this.initializeForms();
        
        console.log('✅ Journal Manager инициализирован');
    }
    
    cacheElements() {
        // Форма взвешивания
        this.elements.weighingDate = document.getElementById('weighingDate');
        this.elements.weighingTime = document.getElementById('weighingTime');
        this.elements.weighingGroup = document.getElementById('weighingGroup');
        this.elements.weighingTotal = document.getElementById('weighingTotal');
        this.elements.weighingWeight = document.getElementById('weighingWeight');
        this.elements.weighingAvgWeight = document.getElementById('weighingAvgWeight');
        this.elements.saveWeighingBtn = document.getElementById('saveWeighingBtn');
        
        // Фильтры журнала
        this.elements.logsDateFrom = document.getElementById('logsDateFrom');
        this.elements.logsDateTo = document.getElementById('logsDateTo');
        this.elements.logsCameraFilter = document.getElementById('logsCameraFilter');
        this.elements.refreshLogsBtn = document.getElementById('refreshLogsBtn');
        this.elements.exportLogsBtn = document.getElementById('exportLogsBtn');
        
        // Статистика
        this.elements.journalTotalActs = document.getElementById('journalTotalActs');
        this.elements.journalTotalCount = document.getElementById('journalTotalCount');
        this.elements.journalTotalWeight = document.getElementById('journalTotalWeight');
        this.elements.journalAvgWeight = document.getElementById('journalAvgWeight');
        
        // Список записей
        this.elements.journalList = document.getElementById('journalList');
        
        // Сверка
        this.elements.verificationUploadArea = document.getElementById('verificationUploadArea');
        this.elements.verificationFileInput = document.getElementById('verificationFileInput');
        this.elements.startVerificationBtn = document.getElementById('startVerificationBtn');
        this.elements.verificationResults = document.getElementById('verificationResults');
        this.elements.verificationMatches = document.getElementById('verificationMatches');
        this.elements.verificationDiffs = document.getElementById('verificationDiffs');
    }
    
    setupEventHandlers() {
        // Форма взвешивания
        if (this.elements.saveWeighingBtn) {
            this.elements.saveWeighingBtn.addEventListener('click', () => {
                this.saveWeighingRecord();
            });
        }
        
        // Автоматический расчет среднего веса
        if (this.elements.weighingTotal && this.elements.weighingWeight) {
            [this.elements.weighingTotal, this.elements.weighingWeight].forEach(el => {
                el.addEventListener('input', () => {
                    this.calculateAverageWeight();
                });
            });
        }
        
        // Фильтры журнала
        if (this.elements.refreshLogsBtn) {
            this.elements.refreshLogsBtn.addEventListener('click', () => {
                this.loadJournalRecords();
            });
        }
        
        if (this.elements.exportLogsBtn) {
            this.elements.exportLogsBtn.addEventListener('click', () => {
                this.exportRecords();
            });
        }
        
        // Сверка файлов
        if (this.elements.verificationUploadArea && this.elements.verificationFileInput) {
            this.elements.verificationUploadArea.addEventListener('click', () => {
                this.elements.verificationFileInput.click();
            });
            
            this.elements.verificationFileInput.addEventListener('change', (e) => {
                this.handleFileUpload(e.target.files[0]);
            });
            
            // Drag & Drop
            this.elements.verificationUploadArea.addEventListener('dragover', (e) => {
                e.preventDefault();
                e.currentTarget.style.backgroundColor = 'rgba(44, 123, 229, 0.1)';
            });
            
            this.elements.verificationUploadArea.addEventListener('dragleave', (e) => {
                e.preventDefault();
                e.currentTarget.style.backgroundColor = '';
            });
            
            this.elements.verificationUploadArea.addEventListener('drop', (e) => {
                e.preventDefault();
                e.currentTarget.style.backgroundColor = '';
                const file = e.dataTransfer.files[0];
                if (file) {
                    this.handleFileUpload(file);
                }
            });
        }
        
        if (this.elements.startVerificationBtn) {
            this.elements.startVerificationBtn.addEventListener('click', () => {
                this.startVerification();
            });
        }
    }
    
    initializeForms() {
        // Устанавливаем текущую дату и время
        const now = new Date();
        if (this.elements.weighingDate) {
            this.elements.weighingDate.value = now.toISOString().split('T')[0];
        }
        if (this.elements.weighingTime) {
            this.elements.weighingTime.value = now.toTimeString().slice(0, 5);
        }
        
        // Устанавливаем диапазон дат для фильтра (последние 7 дней)
        const weekAgo = new Date(now.getTime() - 7 * 24 * 60 * 60 * 1000);
        if (this.elements.logsDateFrom) {
            this.elements.logsDateFrom.value = weekAgo.toISOString().split('T')[0];
        }
        if (this.elements.logsDateTo) {
            this.elements.logsDateTo.value = now.toISOString().split('T')[0];
        }
    }
    
    handleAction(action, data) {
        switch (action) {
            case 'save_manual':
                this.saveManualRecord(data);
                break;
            case 'load_logs':
                this.loadJournalRecords();
                break;
            case 'export_records':
                this.exportRecords();
                break;
            case 'verify_data':
                this.startVerification();
                break;
            default:
                console.warn('Неизвестное действие журнала:', action);
        }
    }
    
    async saveManualRecord(data) {
        try {
            const record = {
                timestamp: new Date().toISOString(),
                count: data.count,
                weight: data.weight,
                average_weight: data.weight / data.count,
                type: 'manual',
                source: 'ui'
            };
            
            const response = await fetch('/api/journal/save', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(record)
            });
            
            if (response.ok) {
                console.log('✅ Ручная запись сохранена');
                this.emit('record_saved', record);
                this.loadJournalRecords(); // Обновляем список
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
            
        } catch (error) {
            console.error('❌ Ошибка сохранения записи:', error);
            this.emit('save_error', error);
        }
    }
    
    async saveWeighingRecord() {
        try {
            const record = {
                date: this.elements.weighingDate?.value,
                time: this.elements.weighingTime?.value,
                group: this.elements.weighingGroup?.value,
                total_count: Number(this.elements.weighingTotal?.value || 0),
                total_weight: Number(this.elements.weighingWeight?.value || 0),
                average_weight: Number(this.elements.weighingAvgWeight?.value || 0),
                type: 'weighing',
                source: 'form'
            };
            
            const response = await fetch('/api/journal/weighing', {
                method: 'POST',
                headers: {
                    'Content-Type': 'application/json'
                },
                body: JSON.stringify(record)
            });
            
            if (response.ok) {
                console.log('✅ Акт взвешивания сохранен');
                this.clearWeighingForm();
                this.emit('weighing_saved', record);
                this.loadJournalRecords();
            } else {
                throw new Error(`HTTP ${response.status}`);
            }
            
        } catch (error) {
            console.error('❌ Ошибка сохранения акта взвешивания:', error);
            this.emit('save_error', error);
        }
    }
    
    calculateAverageWeight() {
        const total = Number(this.elements.weighingTotal?.value || 0);
        const weight = Number(this.elements.weighingWeight?.value || 0);
        
        if (total > 0 && weight > 0) {
            const average = (weight / total).toFixed(2);
            if (this.elements.weighingAvgWeight) {
                this.elements.weighingAvgWeight.value = average;
            }
        }
    }
    
    clearWeighingForm() {
        if (this.elements.weighingGroup) this.elements.weighingGroup.value = '';
        if (this.elements.weighingTotal) this.elements.weighingTotal.value = '';
        if (this.elements.weighingWeight) this.elements.weighingWeight.value = '';
        if (this.elements.weighingAvgWeight) this.elements.weighingAvgWeight.value = '';
        
        // Обновляем дату и время
        const now = new Date();
        if (this.elements.weighingDate) {
            this.elements.weighingDate.value = now.toISOString().split('T')[0];
        }
        if (this.elements.weighingTime) {
            this.elements.weighingTime.value = now.toTimeString().slice(0, 5);
        }
    }
    
    async loadJournalRecords() {
        if (this.isLoading) return;
        
        this.isLoading = true;
        
        try {
            const params = new URLSearchParams();
            if (this.elements.logsDateFrom?.value) {
                params.append('date_from', this.elements.logsDateFrom.value);
            }
            if (this.elements.logsDateTo?.value) {
                params.append('date_to', this.elements.logsDateTo.value);
            }
            if (this.elements.logsCameraFilter?.value) {
                params.append('camera', this.elements.logsCameraFilter.value);
            }
            
            const response = await fetch(`/api/journal/records?${params}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const data = await response.json();
            this.records = data.records || [];
            
            this.updateStatistics(data.statistics);
            this.renderRecordsList();
            
            this.emit('records_loaded', this.records);
            
        } catch (error) {
            console.error('❌ Ошибка загрузки записей:', error);
            this.emit('load_error', error);
        } finally {
            this.isLoading = false;
        }
    }
    
    updateStatistics(stats) {
        if (!stats) return;
        
        if (this.elements.journalTotalActs) {
            this.elements.journalTotalActs.textContent = stats.total_acts || 0;
        }
        if (this.elements.journalTotalCount) {
            this.elements.journalTotalCount.textContent = stats.total_count || 0;
        }
        if (this.elements.journalTotalWeight) {
            this.elements.journalTotalWeight.textContent = (stats.total_weight || 0).toFixed(1);
        }
        if (this.elements.journalAvgWeight) {
            this.elements.journalAvgWeight.textContent = (stats.average_weight || 0).toFixed(2);
        }
    }
    
    renderRecordsList() {
        if (!this.elements.journalList) return;
        
        if (this.records.length === 0) {
            this.elements.journalList.innerHTML = '<div class="journal-loading">Записи не найдены</div>';
            return;
        }
        
        const html = this.records.map(record => `
            <div class="journal-record">
                <div class="journal-record-header">
                    <span class="journal-record-date">${this.formatDate(record.timestamp)}</span>
                    <span class="journal-record-type">${this.getRecordTypeLabel(record.type)}</span>
                </div>
                <div class="journal-record-data">
                    <span>Голов: ${record.count || record.total_count || 0}</span>
                    <span>Вес: ${(record.weight || record.total_weight || 0).toFixed(1)} кг</span>
                    <span>Средний: ${(record.average_weight || 0).toFixed(2)} кг</span>
                </div>
                ${record.group ? `<div class="journal-record-group">Группа: ${record.group}</div>` : ''}
            </div>
        `).join('');
        
        this.elements.journalList.innerHTML = html;
    }
    
    formatDate(timestamp) {
        return new Date(timestamp).toLocaleString('ru-RU');
    }
    
    getRecordTypeLabel(type) {
        const labels = {
            'manual': 'Ручной ввод',
            'weighing': 'Взвешивание',
            'auto': 'Автоматический'
        };
        return labels[type] || type;
    }
    
    async exportRecords() {
        try {
            const params = new URLSearchParams();
            if (this.elements.logsDateFrom?.value) {
                params.append('date_from', this.elements.logsDateFrom.value);
            }
            if (this.elements.logsDateTo?.value) {
                params.append('date_to', this.elements.logsDateTo.value);
            }
            
            const response = await fetch(`/api/journal/export?${params}`);
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const blob = await response.blob();
            const url = window.URL.createObjectURL(blob);
            const a = document.createElement('a');
            a.href = url;
            a.download = `journal_export_${new Date().toISOString().split('T')[0]}.csv`;
            document.body.appendChild(a);
            a.click();
            document.body.removeChild(a);
            window.URL.revokeObjectURL(url);
            
            this.emit('export_completed');
            
        } catch (error) {
            console.error('❌ Ошибка экспорта:', error);
            this.emit('export_error', error);
        }
    }
    
    handleFileUpload(file) {
        if (!file) return;
        
        if (!file.name.match(/\\.(xlsx?|csv)$/i)) {
            alert('Пожалуйста, выберите файл Excel (.xlsx, .xls) или CSV');
            return;
        }
        
        console.log('📎 Файл выбран для сверки:', file.name);
        this.uploadedFile = file;
        
        // Обновляем UI
        const uploadArea = this.elements.verificationUploadArea;
        if (uploadArea) {
            uploadArea.innerHTML = `
                <div style="text-align: center; padding: 24px;">
                    <div style="font-size: 48px; margin-bottom: 12px;">✅</div>
                    <div style="font-size: 16px; font-weight: 600; color: var(--text); margin-bottom: 8px;">
                        Файл загружен: ${file.name}
                    </div>
                    <div style="font-size: 13px; color: var(--muted);">
                        Нажмите "Проверить" для начала сверки
                    </div>
                </div>
            `;
        }
    }
    
    async startVerification() {
        if (!this.uploadedFile) {
            alert('Сначала выберите файл для сверки');
            return;
        }
        
        try {
            const formData = new FormData();
            formData.append('file', this.uploadedFile);
            
            const response = await fetch('/api/journal/verify', {
                method: 'POST',
                body: formData
            });
            
            if (!response.ok) {
                throw new Error(`HTTP ${response.status}`);
            }
            
            const result = await response.json();
            this.displayVerificationResults(result);
            
        } catch (error) {
            console.error('❌ Ошибка сверки:', error);
            this.emit('verification_error', error);
        }
    }
    
    displayVerificationResults(result) {
        if (this.elements.verificationMatches) {
            this.elements.verificationMatches.textContent = result.matches || 0;
        }
        if (this.elements.verificationDiffs) {
            this.elements.verificationDiffs.textContent = result.differences || 0;
        }
        
        if (this.elements.verificationResults) {
            this.elements.verificationResults.style.display = 'block';
            // Здесь можно добавить детальное отображение результатов
        }
        
        this.emit('verification_completed', result);
    }
    
    destroy() {
        this.records = [];
        this.uploadedFile = null;
        this.removeAllListeners();
    }
}