"""
Swagger/OpenAPI документация для PigWeight API
"""

from fastapi.openapi.utils import get_openapi


def get_pig_weight_openapi():
    """Возвращает OpenAPI схему"""
    return {
        "openapi": "3.0.0",
        "info": {
            "title": "PigWeight API",
            "version": "1.1.0",
            "description": "API для системы видеоаналитики взвешивания свиней",
            "contact": {
                "name": "Kiro AI",
                "url": "https://kiro.dev"
            }
        },
        "servers": [
            {
                "url": "http://localhost:8000",
                "description": "Local development server"
            }
        ],
        "tags": [
            {
                "name": "Debug",
                "description": "Диагностика и отладка"
            },
            {
                "name": "Processing",
                "description": "Управление обработкой видео"
            },
            {
                "name": "Cameras",
                "description": "Управление камерами"
            },
            {
                "name": "Records",
                "description": "Получение записей актов взвешивания"
            }
        ],
        "paths": {
            "/debug/health": {
                "get": {
                    "tags": ["Debug"],
                    "summary": "Проверка здоровья сервера",
                    "description": "Возвращает статус сервера и всех компонентов",
                    "responses": {
                        "200": {
                            "description": "OK",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "status": {"type": "string"},
                                            "timestamp": {"type": "string"},
                                            "server": {"type": "object"},
                                            "components": {"type": "object"},
                                            "rtsp_diagnostics": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/debug/rtsp": {
                "get": {
                    "tags": ["Debug"],
                    "summary": "Диагностика RTSP",
                    "description": "Подробная информация о состоянии RTSP подключений",
                    "responses": {
                        "200": {
                            "description": "OK",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "rtsp_status": {"type": "string"},
                                            "diagnostics": {"type": "object"},
                                            "summary": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/debug/infer_status": {
                "get": {
                    "tags": ["Debug"],
                    "summary": "Статус инференса",
                    "description": "Информация о производительности и нагрузке системы",
                    "responses": {
                        "200": {
                            "description": "OK",
                            "content": {
                                "application/json": {
                                    "schema": {
                                        "type": "object",
                                        "properties": {
                                            "inference_status": {"type": "string"},
                                            "process_info": {"type": "object"},
                                            "system_info": {"type": "object"}
                                        }
                                    }
                                }
                            }
                        }
                    }
                }
            },
            "/debug/test_rtsp/{camera_id}": {
                "post": {
                    "tags": ["Debug"],
                    "summary": "Тест RTSP камеры",
                    "description": "Проверяет подключение к конкретной RTSP камере",
                    "parameters": [
                        {
                            "name": "camera_id",
                            "in": "path",
                            "required": True,
                            "schema": {"type": "string"},
                            "description": "ID камеры для тестирования"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "Подключение успешно"
                        },
                        "504": {
                            "description": "Таймаут"
                        },
                        "500": {
                            "description": "Ошибка подключения"
                        }
                    }
                }
            },
            "/api/processing/queue/add": {
                "post": {
                    "tags": ["Processing"],
                    "summary": "Добавить видео в очередь",
                    "description": "Добавляет видеофайл для обработки в фоновую очередь",
                    "requestBody": {
                        "required": True,
                        "content": {
                            "application/json": {
                                "schema": {
                                    "type": "object",
                                    "properties": {
                                        "video_path": {"type": "string"}
                                    }
                                }
                            }
                        }
                    },
                    "responses": {
                        "200": {
                            "description": "Видео добавлено в очередь"
                        }
                    }
                }
            },
            "/api/processing/queue/tasks": {
                "get": {
                    "tags": ["Processing"],
                    "summary": "Список всех задач",
                    "description": "Получает список всех задач обработки видео",
                    "parameters": [
                        {
                            "name": "status",
                            "in": "query",
                            "schema": {"type": "string"},
                            "description": "Фильтр по статусу: pending, processing, completed, failed"
                        }
                    ],
                    "responses": {
                        "200": {
                            "description": "OK"
                        }
                    }
                }
            },
            "/api/processing/queue/stats": {
                "get": {
                    "tags": ["Processing"],
                    "summary": "Статистика очереди",
                    "description": "Получает статистику по обработке видео",
                    "responses": {
                        "200": {
                            "description": "OK"
                        }
                    }
                }
            },
            "/api/cameras": {
                "get": {
                    "tags": ["Cameras"],
                    "summary": "Список доступных камер",
                    "description": "Возвращает список всех доступных RTSP камер из конфигурации",
                    "responses": {
                        "200": {
                            "description": "OK"
                        }
                    }
                }
            },
            "/api/records": {
                "get": {
                    "tags": ["Records"],
                    "summary": "Список всех записей актов",
                    "description": "Получает список всех записей о актах взвешивания",
                    "responses": {
                        "200": {
                            "description": "OK"
                        }
                    }
                }
            }
        }
    }


def setup_swagger(app):
    """Настраивает Swagger документацию в приложении FastAPI"""
    app.openapi_schema = get_pig_weight_openapi()
    return app
