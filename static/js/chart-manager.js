/**
 * Менеджер графиков
 * Управляет Chart.js графиками и их обновлением
 */

import { EventEmitter } from './utils/event-emitter.js';

export class ChartManager extends EventEmitter {
    constructor() {
        super();
        
        this.chart = null;
        this.chartCanvas = null;
        this.countHistory = [];
        this.avgWindow = 10;
        
        // Throttling для обновлений графика
        this.lastChartUpdate = 0;
        this.chartUpdateThrottle = 400; // ms
    }
    
    async init() {
        console.log('📊 Инициализация Chart Manager...');
        
        // Ждем загрузки Chart.js
        await this.waitForChartJS();
        
        // Инициализируем график
        this.initializeChart();
        
        console.log('✅ Chart Manager инициализирован');
    }
    
    async waitForChartJS() {
        return new Promise((resolve) => {
            if (typeof Chart !== 'undefined') {
                resolve();
                return;
            }
            
            // Ждем загрузки Chart.js
            const checkChart = () => {
                if (typeof Chart !== 'undefined') {
                    resolve();
                } else {
                    setTimeout(checkChart, 100);
                }
            };
            checkChart();
        });
    }
    
    initializeChart() {
        this.chartCanvas = document.getElementById('liveCountChart');
        if (!this.chartCanvas) {
            console.warn('⚠️ Canvas для графика не найден');
            return;
        }
        
        const ctx = this.chartCanvas.getContext('2d');
        
        this.chart = new Chart(ctx, {
            type: 'line',
            data: {
                labels: [],
                datasets: [
                    {
                        label: 'Количество',
                        data: [],
                        borderColor: '#2c7be5',
                        backgroundColor: 'rgba(44, 123, 229, 0.1)',
                        borderWidth: 2,
                        fill: true,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 4
                    },
                    {
                        label: 'Среднее',
                        data: [],
                        borderColor: '#51cf66',
                        backgroundColor: 'rgba(81, 207, 102, 0.1)',
                        borderWidth: 2,
                        fill: false,
                        tension: 0.3,
                        pointRadius: 0,
                        pointHoverRadius: 4,
                        borderDash: [5, 5]
                    }
                ]
            },
            options: {
                responsive: true,
                maintainAspectRatio: false,
                animation: {
                    duration: 0 // Отключаем анимации для производительности
                },
                interaction: {
                    intersect: false,
                    mode: 'index'
                },
                plugins: {
                    legend: {
                        display: true,
                        position: 'top',
                        labels: {
                            usePointStyle: true,
                            padding: 15,
                            font: {
                                size: 12
                            }
                        }
                    },
                    tooltip: {
                        backgroundColor: 'rgba(255, 255, 255, 0.95)',
                        titleColor: '#1f3d6b',
                        bodyColor: '#2f5078',
                        borderColor: 'rgba(60, 90, 140, 0.2)',
                        borderWidth: 1,
                        cornerRadius: 8,
                        displayColors: true
                    }
                },
                scales: {
                    x: {
                        display: true,
                        grid: {
                            color: 'rgba(60, 90, 140, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#2f5078',
                            font: {
                                size: 11
                            },
                            maxTicksLimit: 8
                        }
                    },
                    y: {
                        display: true,
                        beginAtZero: true,
                        grid: {
                            color: 'rgba(60, 90, 140, 0.1)',
                            drawBorder: false
                        },
                        ticks: {
                            color: '#2f5078',
                            font: {
                                size: 11
                            },
                            precision: 0
                        }
                    }
                }
            }
        });
        
        console.log('📊 График инициализирован');
    }
    
    updateChart(data) {
        if (!this.chart || !data.count === undefined) return;
        
        const now = performance.now();
        
        // Throttling обновлений графика
        if (now - this.lastChartUpdate < this.chartUpdateThrottle) {
            return;
        }
        this.lastChartUpdate = now;
        
        const count = Number(data.count);
        const timestamp = new Date().toLocaleTimeString();
        
        // Добавляем данные в историю
        this.countHistory.push(count);
        if (this.countHistory.length > 60) {
            this.countHistory.shift();
        }
        
        // Вычисляем скользящее среднее
        const start = Math.max(0, this.countHistory.length - this.avgWindow);
        const slice = this.countHistory.slice(start);
        const average = slice.reduce((a, b) => a + b, 0) / slice.length;
        
        // Обновляем данные графика
        this.chart.data.labels.push(timestamp);
        this.chart.data.datasets[0].data.push(count);
        this.chart.data.datasets[1].data.push(Number(average.toFixed(2)));
        
        // Ограничиваем количество точек на графике
        const maxPoints = 60;
        if (this.chart.data.labels.length > maxPoints) {
            this.chart.data.labels.shift();
            this.chart.data.datasets.forEach(dataset => {
                dataset.data.shift();
            });
        }
        
        // Обновляем график без анимации для производительности
        this.chart.update('none');
        
        this.emit('chart_updated', { count, average });
    }
    
    clearChart() {
        if (!this.chart) return;
        
        this.chart.data.labels = [];
        this.chart.data.datasets.forEach(dataset => {
            dataset.data = [];
        });
        
        this.countHistory = [];
        this.chart.update('none');
        
        this.emit('chart_cleared');
    }
    
    setAverageWindow(window) {
        this.avgWindow = Math.max(1, Math.min(60, Number(window)));
        this.emit('average_window_changed', this.avgWindow);
    }
    
    exportChartData() {
        const data = {
            labels: [...this.chart.data.labels],
            counts: [...this.chart.data.datasets[0].data],
            averages: [...this.chart.data.datasets[1].data],
            timestamp: new Date().toISOString()
        };
        
        return data;
    }
    
    importChartData(data) {
        if (!this.chart || !data) return;
        
        this.chart.data.labels = data.labels || [];
        this.chart.data.datasets[0].data = data.counts || [];
        this.chart.data.datasets[1].data = data.averages || [];
        
        this.countHistory = [...(data.counts || [])];
        
        this.chart.update('none');
        this.emit('chart_imported', data);
    }
    
    destroy() {
        if (this.chart) {
            this.chart.destroy();
            this.chart = null;
        }
        
        this.countHistory = [];
        this.removeAllListeners();
    }
}