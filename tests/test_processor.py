import pytest
import numpy as np
from unittest.mock import patch, MagicMock
import asyncio

# Добавляем корень проекта в sys.path для корректного импорта
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from core.processor import get_processor, reset_processors, FrameResult

# pytest-asyncio будет использовать это как основной event loop для тестов
@pytest.fixture(scope="session")
def event_loop():
    try:
        loop = asyncio.get_running_loop()
    except RuntimeError:
        loop = asyncio.new_event_loop()
    yield loop
    loop.close()

@pytest.fixture(autouse=True)
async def cleanup_processors():
    """Сбрасывает все процессоры после каждого теста."""
    yield
    await reset_processors()

@pytest.mark.asyncio
async def test_processor_creation():
    """Тест, что процессор создается и удаляется корректно."""
    processor = await get_processor("test_stream")
    assert processor is not None
    assert processor.stream_id == "test_stream"
    
    processor_same = await get_processor("test_stream")
    assert processor is processor_same

    await reset_processors()
    processor_new = await get_processor("test_stream")
    assert processor is not processor_new

# --- Тестирование маппинга координат ---

def get_mock_model_adapter(mock_inference_result):
    """Создает мок ModelAdapter с заданным результатом инференса."""
    mock_adapter = MagicMock()
    
    # Мок для инференса
    def infer_side_effect(frames):
        # Возвращаем результат для каждого кадра в батче
        return [mock_inference_result] * len(frames)
        
    mock_adapter.infer.side_effect = infer_side_effect
    
    # Мок для бэкенда, чтобы процессор считался активным
    mock_adapter.backend = "mock"
    
    return mock_adapter

@pytest.mark.asyncio
async def test_mask_mapping_with_batcher():
    """
    Проверяет корректность маппинга масок при использовании DynamicBatcher.
    """
    # 1. Входные данные
    original_h, original_w = 1080, 1920
    target_size = 960
    stream_id = "batch_mapping_test"
    original_frame = np.zeros((original_h, original_w, 3), dtype=np.uint8)

    # 2. Ожидаемый результат от модели (координаты в пространстве 960x960)
    padding_height = int(target_size * 0.075)
    content_height = target_size - 2 * padding_height
    x_start, y_start = (target_size - 100) // 2, padding_height + (content_height - 100) // 2
    mock_poly = [
        (x_start, y_start), (x_start + 100, y_start),
        (x_start + 100, y_start + 100), (x_start, y_start + 100),
    ]
    mock_inference_result = {
        "detections": 1, "confidence": 0.9, "masks": [mock_poly],
        "bboxes": [], "centroids": []
    }

    # 3. Ручной расчет ожидаемых координат
    crop_size = original_h
    crop_start_x = (original_w - crop_size) // 2
    crop_start_y = 0
    expected_mapped_poly = []
    for x_proc, y_proc in mock_poly:
        y_in_content = y_proc - padding_height
        x_in_content = x_proc
        y_in_resized_square = y_in_content * (target_size / content_height)
        x_in_resized_square = x_in_content
        y_in_cropped = y_in_resized_square * (crop_size / target_size)
        x_in_cropped = x_in_resized_square * (crop_size / target_size)
        y_orig = y_in_cropped + crop_start_y
        x_orig = x_in_cropped + crop_start_x
        expected_mapped_poly.append((x_orig, y_orig))

    # 4. Запуск теста с моком
    mock_adapter_instance = get_mock_model_adapter(mock_inference_result)
    
    with patch('core.processor.ModelAdapter', return_value=mock_adapter_instance):
        processor = await get_processor(stream_id)
        
        # Запускаем обработку асинхронно
        result = await processor.process_frame_async(original_frame)

    # 5. Проверка результатов
    assert result is not None
    assert result.detections == 1
    assert len(result.masks) == 1
    
    mapped_poly_result = result.masks[0]
    assert len(mapped_poly_result) == len(expected_mapped_poly)
    
    for i, (x_res, y_res) in enumerate(mapped_poly_result):
        x_exp, y_exp = expected_mapped_poly[i]
        assert abs(x_res - x_exp) < 1.0
        assert abs(y_res - y_exp) < 1.0

    print("\n✅ Тест маппинга масок с батчером успешно пройден!")