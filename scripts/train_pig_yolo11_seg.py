#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Скрипт обучения pig-only YOLO11 сегментационной модели на датасете Roboflow.

По умолчанию:
- data: models/pig.v1i.yolov11/data.yaml (как у вас распаковано)
- базовая модель: models/yolo11n-seg.pt
- device=0, epochs=100, imgsz=640, batch=8
- По завершении копирует runs/segment/train*/weights/best.pt в models/pig_yolo11-seg.pt

Запуск:
  python scripts/train_pig_yolo11_seg.py
  # либо с параметрами:
  python scripts/train_pig_yolo11_seg.py --data models/pig.v1i.yolov11/data.yaml --model models/yolo11n-seg.pt --epochs 300 --batch 8 --imgsz 640 --device cpu

Примечания:
- Требует установленный пакет 'ultralytics' (будет установлен автоматически, если отсутствует).
- Рабочая директория — корень проекта.
"""

import argparse
import subprocess
import sys
import shutil
from pathlib import Path
import pkgutil


def ensure_pkg(name: str, pip_name: str = None) -> None:
    """Устанавливает пакет через pip, если не найден."""
    if pkgutil.find_loader(name) is None:
        pn = pip_name or name
        print(f"[INFO] Installing {pn} ...", flush=True)
        subprocess.check_call([sys.executable, "-m", "pip", "install", pn])


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Train pig-only YOLO11 segmentation model")
    p.add_argument("--data", type=str, default="models/pig.v1i.yolov11/data.yaml", help="Путь к data.yaml")
    p.add_argument("--model", type=str, default="models/yolo11n-seg.pt", help="Базовая модель")
    p.add_argument("--epochs", type=int, default=300, help="Количество эпох")
    p.add_argument("--batch", type=int, default=8, help="Batch size")
    p.add_argument("--imgsz", type=int, default=640, help="Размер входного изображения")
    p.add_argument("--device", type=str, default="cpu", help="GPU id ('0') или 'cpu'")
    p.add_argument("--out", type=str, default="models/pig_yolo11-seg.pt", help="Куда скопировать итоговый best.pt")
    return p.parse_args()


def main():
    args = parse_args()

    project_root = Path(".").resolve()
    data_path = project_root / args.data
    model_path = project_root / args.model
    out_path = project_root / args.out

    if not data_path.exists():
        print(f"[ERROR] data.yaml не найден: {data_path}", file=sys.stderr)
        sys.exit(2)
    if not model_path.exists():
        print(f"[ERROR] базовая модель не найдена: {model_path}", file=sys.stderr)
        sys.exit(2)

    # Гарантируем ultralytics
    ensure_pkg("ultralytics", "ultralytics==8.3.70")

    # Команда обучения
    # Авто-детект: если запросили GPU '0', но CUDA недоступна — переключаемся на cpu
    wanted_device = str(args.device).strip()
    if wanted_device != "cpu":
        try:
            import torch  # type: ignore
            if not torch.cuda.is_available():
                print("[WARN] CUDA недоступна, переключаюсь на device=cpu", flush=True)
                wanted_device = "cpu"
        except Exception:
            print("[WARN] Не удалось проверить CUDA, переключаюсь на device=cpu", flush=True)
            wanted_device = "cpu"

    cmd = [
        "yolo",
        "train",
        f"model={str(model_path)}",
        f"data={str(data_path)}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"device={wanted_device}",
    ]
    print("[INFO] Running:", " ".join(cmd), flush=True)
    ret = subprocess.call(cmd)
    if ret != 0:
        print(f"[ERROR] yolo train завершился с кодом {ret}", file=sys.stderr)
        sys.exit(ret)

    # Поиск последнего best.pt
    seg_dir = project_root / "runs" / "segment"
    if not seg_dir.exists():
        print(f"[ERROR] Директория с результатами не найдена: {seg_dir}", file=sys.stderr)
        sys.exit(3)

    runs = sorted(seg_dir.glob("train*"), key=lambda p: p.stat().st_mtime, reverse=True)
    if not runs:
        print("[ERROR] Не найдены каталоги runs/segment/train*", file=sys.stderr)
        sys.exit(3)

    best = runs[0] / "weights" / "best.pt"
    if not best.exists():
        print(f"[ERROR] best.pt не найден: {best}", file=sys.stderr)
        sys.exit(3)

    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_path)
    print(f"[INFO] Чекпойнт скопирован в: {out_path}", flush=True)
    print("[DONE] Обучение завершено успешно.")


if __name__ == "__main__":
    main()