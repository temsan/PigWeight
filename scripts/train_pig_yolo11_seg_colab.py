#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Colab-скрипт обучения pig-only YOLO11 сегментационной модели.

Ожидания:
- Вы предварительно загрузите архив датасета в Colab по пути: /content/pig.v1i.yolov11.zip
  (из Roboflow экспорт YOLOv11/YOLOv8 Segmentation).
- Скрипт распакует архив в /content/datasets/pig.v1i.yolov11
- Установит нужные пакеты (ultralytics + CUDA-совместимый torch через index PyTorch для Colab)
- Запустит обучение YOLO11-seg
- Скопирует best.pt в /content/pig_yolo11-seg.pt
- (Опционально) Сохранит на Google Drive, если смонтирован.

Быстрый запуск в Colab:
  !python /content/scripts/train_pig_yolo11_seg_colab.py \
      --zip /content/pig.v1i.yolov11.zip \
      --epochs 300 --batch 8 --imgsz 640 --device 0

Параметры по умолчанию:
- zip: /content/pig.v1i.yolov11.zip
- model: yolo11n-seg.pt (скачается из ultralytics автоматически, если не доступна локально)
- epochs: 300 (рекомендуется потолок для маленького датасета) — при необходимости добавьте patience в команду yolo вручную
- batch: 8
- imgsz: 640
- device: 0 (GPU)
"""

import os
import sys
import zipfile
import argparse
import subprocess
from pathlib import Path
import shutil


def run(cmd, check=True):
    print("[CMD]", " ".join(cmd), flush=True)
    process = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True)
    while True:
        line = process.stdout.readline()
        if not line and process.poll() is not None:
            break
        if line:
            print(line.strip(), flush=True)
    rc = process.poll()
    if check and rc != 0:
        raise SystemExit(rc)
    return rc


def ensure_colab_torch():
    """
    На Colab обычно уже есть совместимый CUDA torch.
    Если нужно — можно переустановить:
      pip install --index-url https://download.pytorch.org/whl/cu124 torch torchvision torchaudio
    Здесь просто проверим, что torch видит CUDA.
    """
    try:
        import torch  # noqa: F401
        import torch  # type: ignore
        print("[INFO] torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "CUDA:", getattr(torch.version, "cuda", None))
        if not torch.cuda.is_available():
            print("[WARN] CUDA недоступна. Если вы уверены, что вам нужен GPU, проверьте, что в Colab выбрано Runtime -> Change runtime type -> T4/A100 GPU.")
    except Exception as e:
        print("[WARN] Не удалось импортировать torch:", e)
        print("[INFO] Устанавливаю CUDA-сборку PyTorch...")
        run([sys.executable, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cu124", "torch", "torchvision", "torchaudio"], check=True)
        import torch  # type: ignore
        print("[INFO] torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "CUDA:", getattr(torch.version, "cuda", None))


def ensure_ultralytics():
    run([sys.executable, "-m", "pip", "install", "-U", "ultralytics==8.3.70"], check=True)
    # Иногда numpy>=2 конфликтует с другими пакетами; если потребуется, можно закрепить:
    # run([sys.executable, "-m", "pip", "install", "numpy<2"], check=False)
    import ultralytics  # noqa: F401
    print("[INFO] ultralytics installed OK")


def unzip_dataset(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Распаковка датасета {zip_path} -> {out_dir}", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)
    # Пытаемся обнаружить корень с data.yaml (Roboflow обычно: models/pig.v1i.yolov11/... но у нас кладём в /content/datasets).
    # Здесь архив ожидается в формате:
    #   /content/pig.v1i.yolov11.zip с папками train/valid/test + data.yaml в корне распаковки
    # После распаковки out_dir может содержать папку верхнего уровня — найдём data.yaml.
    root = out_dir
    candidates = list(out_dir.rglob("data.yaml"))
    if candidates:
        # Выберем ближайший к корню
        root = candidates[0].parent
    print("[INFO] Dataset root:", root)
    return root


def write_data_yaml_if_needed(root: Path) -> Path:
    """
    Если в распакованном датасете уже есть корректный data.yaml — используем его.
    Если нет — создадим минимальный под YOLO сегментацию.
    """
    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        print("[INFO] Используем существующий data.yaml:", data_yaml)
        return data_yaml

    # Создадим минимальный
    print("[INFO] Создаю data.yaml:", data_yaml)
    content = "\n".join([
        f"path: {root.as_posix()}",
        "train: train/images",
        "val: valid/images",
        "test: test/images",
        "names: [pig]",
        "task: segment",
        ""
    ])
    data_yaml.write_text(content, encoding="utf-8")
    return data_yaml


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--zip", type=str, default="/content/pig.v1i.yolov11.zip", help="Путь к ZIP датасета в Colab")
    ap.add_argument("--model", type=str, default="yolo11n-seg.pt", help="Базовая модель (локальный файл или имя модели Ultralytics)")
    ap.add_argument("--epochs", type=int, default=300, help="Количество эпох (верхняя граница)")
    ap.add_argument("--batch", type=int, default=8, help="Batch size")
    ap.add_argument("--imgsz", type=int, default=640, help="Размер входного изображения")
    ap.add_argument("--device", type=str, default="0", help="GPU id ('0') или 'cpu' (на Colab обычно 0)")
    ap.add_argument("--out", type=str, default="/content/pig_yolo11-seg.pt", help="Куда скопировать итоговый best.pt")
    ap.add_argument("--save_to_drive", action="store_true", help="Скопировать чекпойнт на Google Drive, если смонтирован")
    args = ap.parse_args()

    zip_path = Path(args.zip)
    if not zip_path.exists():
        print(f"[ERROR] ZIP датасет не найден: {zip_path}", file=sys.stderr)
        sys.exit(2)

    # Убедимся в наличии torch/ultralytics
    ensure_colab_torch()
    ensure_ultralytics()

    datasets_root = Path("/content/datasets")
    ds_root = unzip_dataset(zip_path, datasets_root / "pig.v1i.yolov11")
    data_yaml = ds_root / "data.yaml"
    data_yaml = write_data_yaml_if_needed(ds_root)

    # Собираем команду обучения
    # При желании можно добавить раннюю остановку: patience=30
    train_cmd = [
        "yolo", "train",
        f"model={args.model}",
        f"data={str(data_yaml)}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"device={args.device}",
        "patience=30",  # раскомментируйте при необходимости
    ]
    print("[INFO] Запускаю обучение:", " ".join(train_cmd), flush=True)
    code = run(train_cmd, check=False)
    if code != 0:
        print(f"[ERROR] yolo train завершился с кодом {code}", file=sys.stderr)
        sys.exit(code)

    # Поиск best.pt
    seg_dir = Path("/content/runs/segment")
    runs = sorted(seg_dir.glob("train*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print("[ERROR] Не найдены каталоги /content/runs/segment/train*", file=sys.stderr)
        sys.exit(3)

    best = runs[0] / "weights" / "best.pt"
    if not best.exists():
        print(f"[ERROR] best.pt не найден: {best}", file=sys.stderr)
        sys.exit(3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_path)
    print(f"[INFO] Чекпойнт скопирован в: {out_path}", flush=True)

    # Копирование на Google Drive, если смонтирован и флаг указан
    if args.save_to_drive:
        drive_root = Path("/content/drive/MyDrive")
        if drive_root.exists():
            dst = drive_root / Path(out_path.name)
            shutil.copy2(out_path, dst)
            print(f"[INFO] Чекпойнт также сохранён в Google Drive: {dst}", flush=True)
        else:
            print("[WARN] Google Drive не смонтирован. Пропускаю сохранение.", flush=True)

    print("[DONE] Обучение в Colab завершено.")


if __name__ == "__main__":
    main()