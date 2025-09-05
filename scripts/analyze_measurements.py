#!/usr/bin/env python3
"""
Анализ файла замеров для понимания структуры данных
"""
import pandas as pd
import json
from datetime import datetime
import sys
import os

def analyze_measurements_file(filepath):
    """Анализ Excel файла с замерами"""
    try:
        # Читаем Excel файл
        df = pd.read_excel(filepath, engine='openpyxl')
        
        print(f"📊 Анализ файла: {filepath}")
        print(f"📏 Размер данных: {df.shape[0]} строк, {df.shape[1]} столбцов")
        print(f"📋 Названия столбцов: {df.columns.tolist()}")
        print()
        
        # Анализируем структуру данных
        print("🔍 Первые 15 строк:")
        print(df.head(15).to_string())
        print()
        
        # Ищем даты в данных
        print("📅 Поиск дат в данных:")
        for i, col in enumerate(df.columns):
            if isinstance(col, datetime):
                print(f"  Столбец {i}: {col} (дата)")
            elif 'дата' in str(col).lower() or 'date' in str(col).lower():
                print(f"  Столбец {i}: {col} (возможно дата)")
            else:
                print(f"  Столбец {i}: {col}")
        print()
        
        # Анализируем типы данных
        print("📊 Типы данных по столбцам:")
        for i, (col, dtype) in enumerate(zip(df.columns, df.dtypes)):
            print(f"  {i}: {col} -> {dtype}")
        print()
        
        # Ищем числовые столбцы (возможно веса)
        numeric_cols = df.select_dtypes(include=['int64', 'float64']).columns
        print(f"🔢 Числовые столбцы: {numeric_cols.tolist()}")
        
        # Статистика по числовым столбцам
        if len(numeric_cols) > 0:
            print("\n📈 Статистика по числовым данным:")
            print(df[numeric_cols].describe())
        
        # Ищем уникальные значения в текстовых столбцах
        text_cols = df.select_dtypes(include=['object']).columns
        print(f"\n📝 Текстовые столбцы: {text_cols.tolist()}")
        
        for col in text_cols:
            unique_vals = df[col].dropna().unique()
            if len(unique_vals) < 20:  # Показываем только если не слишком много значений
                print(f"  {col}: {unique_vals.tolist()}")
            else:
                print(f"  {col}: {len(unique_vals)} уникальных значений")
        
        # Пытаемся определить структуру данных
        print("\n🎯 Предполагаемая структура данных:")
        
        # Ищем столбцы с весами (числовые значения в разумных пределах для свиней)
        weight_candidates = []
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                min_val, max_val = values.min(), values.max()
                if 10 <= min_val <= 1000 and 50 <= max_val <= 2000:  # Разумные пределы веса свиней
                    weight_candidates.append((col, min_val, max_val, len(values)))
        
        if weight_candidates:
            print("  Возможные столбцы с весами:")
            for col, min_val, max_val, count in weight_candidates:
                print(f"    {col}: {min_val:.1f} - {max_val:.1f} кг ({count} записей)")
        
        # Ищем столбцы с количеством
        count_candidates = []
        for col in numeric_cols:
            values = df[col].dropna()
            if len(values) > 0:
                min_val, max_val = values.min(), values.max()
                if 1 <= min_val <= 500 and 1 <= max_val <= 500 and all(v == int(v) for v in values[:10]):
                    count_candidates.append((col, min_val, max_val, len(values)))
        
        if count_candidates:
            print("  Возможные столбцы с количеством:")
            for col, min_val, max_val, count in count_candidates:
                print(f"    {col}: {min_val:.0f} - {max_val:.0f} шт ({count} записей)")
        
        # Сохраняем результат анализа
        analysis_result = {
            'file_path': filepath,
            'shape': list(df.shape),
            'columns': [str(col) for col in df.columns],
            'column_types': {str(col): str(dtype) for col, dtype in zip(df.columns, df.dtypes)},
            'numeric_columns': [str(col) for col in numeric_cols],
            'text_columns': [str(col) for col in text_cols],
            'weight_candidates': [(str(col), float(min_val), float(max_val), int(count)) for col, min_val, max_val, count in weight_candidates],
            'count_candidates': [(str(col), int(min_val), int(max_val), int(count)) for col, min_val, max_val, count in count_candidates],
            'sample_data': [
                {str(k): (str(v) if pd.isna(v) else v) for k, v in row.items()}
                for row in df.head(10).to_dict('records')
            ]
        }
        
        # Сохраняем в JSON файл
        output_file = 'measurements_analysis.json'
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_result, f, ensure_ascii=False, indent=2, default=str)
        
        print(f"\n💾 Результат анализа сохранен в {output_file}")
        return analysis_result
        
    except Exception as e:
        print(f"❌ Ошибка при анализе файла: {e}")
        return None

if __name__ == "__main__":
    # Ищем файл замеров
    measurements_file = "docs/Замеры 20.07 по 03.09.xlsx"
    
    if not os.path.exists(measurements_file):
        print(f"❌ Файл не найден: {measurements_file}")
        sys.exit(1)
    
    result = analyze_measurements_file(measurements_file)
    if result:
        print("✅ Анализ завершен успешно!")
    else:
        print("❌ Анализ завершился с ошибкой")
        sys.exit(1)
