# Changelog

## [Unreleased] - 2025-10-18

### Cleaned Up
- Удалены 40+ устаревших документов из корня проекта
- Удалены временные файлы (tash list, tatus, temp_ui_commit.html)
- Удалены дубликаты конфигов (config.env.example)
- Удалены устаревшие тестовые файлы из корня
- Удалены папки: archive/, stream/, supabase/, pig_tracking/, .benchmarks/, .pytest_cache/, .qoder/
- Очищены все __pycache__ директории

### Organized
- Перенесены TODO.md и TROUBLESHOOTING.md в docs/
- Создан docs/README.md с навигацией по документации
- Обновлена структура проекта в README.md

### Current Structure
```
PigWeight/
├── api/              # API endpoints
├── core/             # Core system
├── services/         # Services (inference, model adapter)
├── static/           # Frontend assets
├── scripts/          # Utility scripts
├── tests/            # Tests
├── docs/             # Documentation
├── main.py           # Entry point
├── requirements.txt  # Dependencies
└── README.md         # Main documentation
```

## Previous Versions

См. git history для предыдущих изменений.
