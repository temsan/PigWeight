#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Google Colab-скрипт дообучения (fine-tune) YOLO11 сегментационной модели на вашем датасете
с визуализацией ключевых метрик в конце.

Ожидания:
- В Colab у вас есть архив датасета с экспортом Roboflow YOLOv11/YOLOv8 Segmentation.
  Например, "/content/models/pig 2.v2i.yolov11.zip" (обратите внимание на пробелы в имени).
- У вас есть стартовые веса сегментационной модели, например "/content/models/pig_yolo11-seg.pt".
- Скрипт распакует датасет в /content/datasets/<имя>, запустит обучение с указанными весами
  и по завершении создаст графики метрик наподобие Roboflow (лоссы, PR/Recall/mAP и т.п.).

Быстрый запуск в Colab (пример):
  !python /content/PigWeight/scripts/finetune_pig_yolo11_seg_colab.py \
      --zip "/content/PigWeight/models/pig 2.v2i.yolov11.zip" \
      --base "/content/PigWeight/models/pig_yolo11-seg.pt" \
      --epochs 300 --batch 8 --imgsz 640 --device 0

Параметры по умолчанию:
- zip: /content/models/pig 2.v2i.yolov11.zip
- base: /content/models/pig_yolo11-seg.pt (ваши текущие веса)
- epochs: 300
- batch: 8
- imgsz: 640
- device: 0 (GPU)
- out: /content/pig_yolo11-seg-finetuned.pt
"""

from __future__ import annotations

import os
import sys
import zipfile
import argparse
import subprocess
from pathlib import Path
import shutil


def run(cmd, check: bool = True) -> int:
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


def ensure_pkg_installation():
    """Гарантируем наличие torch (CUDA), ultralytics, pandas и matplotlib в Colab."""
    # Torch (обычно уже установлен в Colab)
    try:
        import torch  # noqa: F401
        import torch  # type: ignore
        print("[INFO] torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "CUDA:", getattr(torch.version, "cuda", None))
        if not torch.cuda.is_available():
            print("[WARN] CUDA недоступна. В Colab проверьте Runtime -> Change runtime type -> GPU.")
    except Exception as e:
        print("[WARN] Не удалось импортировать torch:", e)
        print("[INFO] Устанавливаю CUDA-сборку PyTorch...")
        run([sys.executable, "-m", "pip", "install", "--index-url", "https://download.pytorch.org/whl/cu124", "torch", "torchvision", "torchaudio"], check=True)
        import torch  # type: ignore
        print("[INFO] torch:", torch.__version__, "CUDA avail:", torch.cuda.is_available(), "CUDA:", getattr(torch.version, "cuda", None))

    # Ultralytics
    run([sys.executable, "-m", "pip", "install", "-U", "ultralytics==8.3.70"], check=True)
    import ultralytics  # noqa: F401
    print("[INFO] ultralytics installed OK")

    # Pandas + Matplotlib для графиков
    run([sys.executable, "-m", "pip", "install", "-U", "pandas", "matplotlib"], check=True)
    import pandas as pd  # noqa: F401
    import matplotlib  # noqa: F401
    print("[INFO] pandas/matplotlib installed OK")


def unzip_dataset(zip_path: Path, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"[INFO] Распаковка датасета {zip_path} -> {out_dir}", flush=True)
    with zipfile.ZipFile(zip_path, 'r') as zf:
        zf.extractall(out_dir)

    # Найдём корень, где лежит data.yaml
    root = out_dir
    candidates = list(out_dir.rglob("data.yaml"))
    if candidates:
        root = candidates[0].parent
    print("[INFO] Dataset root:", root)
    return root


def ensure_data_yaml(root: Path) -> Path:
    """Если нет data.yaml, создадим минимальный под сегментацию."""
    data_yaml = root / "data.yaml"
    if data_yaml.exists():
        print("[INFO] Используем существующий data.yaml:", data_yaml)
        return data_yaml

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


def find_latest_run(seg_root: Path, preferred_name: str | None = None) -> Path:
    """Возвращает каталог последнего прогона сегментации.
    Сначала проверяет каталог по имени (например, name=pig-finetune),
    затем выбирает самый свежий подкаталог из seg_root.
    """
    if preferred_name:
        preferred = seg_root / preferred_name
        if preferred.exists() and preferred.is_dir():
            return preferred

    # Современные версии Ultralytics создают произвольные имена (не только train*).
    # Берём любой подкаталог, отсортированный по времени изменения.
    candidates = [p for p in seg_root.iterdir() if p.is_dir()]
    candidates.sort(key=lambda p: p.stat().st_mtime, reverse=True)
    if not candidates:
        raise FileNotFoundError("Не найдены каталоги прогона в " + str(seg_root))
    return candidates[0]


def try_display_images(image_paths: list[Path]) -> None:
    """Отобразить изображения в Colab, если возможно, иначе просто вывести пути."""
    shown_any = False
    try:
        from IPython.display import display
        from PIL import Image
        for p in image_paths:
            if p.exists():
                try:
                    display(Image.open(p))
                    shown_any = True
                except Exception:
                    print("[IMG]", p.as_posix())
            else:
                print("[MISS]", p.as_posix())
    except Exception:
        # Вне ноутбука просто печатаем пути
        for p in image_paths:
            tag = "[IMG]" if p.exists() else "[MISS]"
            print(tag, p.as_posix())
    if not shown_any:
        print("[INFO] Изображения не были отображены inline (возможно, запуск вне ноутбука). См. пути выше.")


def generate_custom_plots(results_csv: Path, out_dir: Path) -> list[Path]:
    """Строим сводные графики по results.csv (как в Roboflow: лоссы и метрики)."""
    import pandas as pd
    import matplotlib.pyplot as plt

    out_dir.mkdir(parents=True, exist_ok=True)

    if not results_csv.exists():
        print("[WARN] results.csv не найден:", results_csv)
        return []

    df = pd.read_csv(results_csv)
    if 'epoch' not in df.columns:
        print("[WARN] В results.csv нет столбца 'epoch' — пропускаю пользовательские графики")
        return []

    saved: list[Path] = []

    # Группа 1: train*/val* лоссы
    loss_cols = [c for c in df.columns if (c.startswith('train/') or c.startswith('val/')) and not c.startswith('val/best') and not c.startswith('val/box') and not c.startswith('val/cls') and not c.startswith('val/dfl')]
    # Подстрахуемся: если ничего не выбралось, попробуем любые train/ или val/
    if not loss_cols:
        loss_cols = [c for c in df.columns if c.startswith('train/') or c.startswith('val/')]

    if loss_cols:
        plt.figure(figsize=(10, 6))
        for col in loss_cols:
            try:
                plt.plot(df['epoch'], df[col], label=col)
            except Exception:
                pass
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.title('Train/Val losses and related values')
        plt.legend(loc='best', fontsize=8)
        loss_png = out_dir / 'custom_losses.png'
        plt.tight_layout()
        plt.savefig(loss_png.as_posix())
        plt.close()
        saved.append(loss_png)
        print("[PLOT]", loss_png.as_posix())

    # Группа 2: метрики
    metric_cols = [c for c in df.columns if c.startswith('metrics/')]
    if metric_cols:
        plt.figure(figsize=(10, 6))
        for col in metric_cols:
            try:
                plt.plot(df['epoch'], df[col], label=col)
            except Exception:
                pass
        plt.xlabel('epoch')
        plt.ylabel('value')
        plt.title('Metrics (precision/recall/mAP etc.)')
        plt.legend(loc='best', fontsize=8)
        metrics_png = out_dir / 'custom_metrics.png'
        plt.tight_layout()
        plt.savefig(metrics_png.as_posix())
        plt.close()
        saved.append(metrics_png)
        print("[PLOT]", metrics_png.as_posix())

    return saved


def _pick_first_existing(df, candidates: list[str]) -> str | None:
    for c in candidates:
        if c in df.columns:
            return c
    return None


def generate_roboflow_style_dashboard(results_csv: Path, out_png: Path) -> Path | None:
    """Строим единое полотно в стиле Roboflow: верх — mAP50/mAP50-95, снизу — лоссы.
    Возвращает путь к PNG или None, если results.csv отсутствует/пустой.
    """
    import pandas as pd
    import matplotlib.pyplot as plt
    from matplotlib import gridspec

    if not results_csv.exists():
        print("[WARN] results.csv не найден:", results_csv)
        return None

    df = pd.read_csv(results_csv)
    if df.empty or 'epoch' not in df.columns:
        print("[WARN] results.csv пуст или без столбца 'epoch' — пропускаю Roboflow-style график")
        return None

    # Поиск метрик (предпочитаем маски для сегментации, затем боксы)
    map50_col = _pick_first_existing(df, [
        'metrics/mAP50(M)', 'metrics/mAP50(S)', 'metrics/mAP50', 'metrics/mAP50(B)'
    ])
    map5095_col = _pick_first_existing(df, [
        'metrics/mAP50-95(M)', 'metrics/mAP50-95(S)', 'metrics/mAP50-95', 'metrics/mAP50-95(B)'
    ])
    precision_col = _pick_first_existing(df, [
        'metrics/precision(M)', 'metrics/precision', 'metrics/precision(B)'
    ])
    recall_col = _pick_first_existing(df, [
        'metrics/recall(M)', 'metrics/recall', 'metrics/recall(B)'
    ])

    # Поиск лоссов (train/val) с гибкими именами
    def find_loss(prefix: str, keywords: list[str]) -> tuple[str | None, str | None, str]:
        # Возвращает (train_col, val_col, title)
        train_col = None
        val_col = None
        title = prefix
        # Возможные варианты имён
        train_candidates = []
        val_candidates = []
        for kw in keywords:
            # Частые паттерны Ultralytics
            train_candidates += [f"train/{kw}", f"train/{kw}_loss", f"train/loss_{kw}"]
            val_candidates += [f"val/{kw}", f"val/{kw}_loss", f"val/loss_{kw}"]
        # Общие
        train_candidates += ["train/obj_loss", "train/object_loss", "train/box_loss", "train/cls_loss", "train/dfl_loss", "train/seg_loss", "train/mask_loss"]
        val_candidates += ["val/obj_loss", "val/object_loss", "val/box_loss", "val/cls_loss", "val/dfl_loss", "val/seg_loss", "val/mask_loss"]

        train_col = _pick_first_existing(df, train_candidates)
        val_col = _pick_first_existing(df, val_candidates)

        # Красивые заголовки
        lowers = " ".join(keywords).lower()
        if "box" in lowers:
            title = "Box Loss"
        elif "cls" in lowers or "class" in lowers:
            title = "Class Loss"
        elif "obj" in lowers or "object" in lowers:
            title = "Object Loss"
        elif "dfl" in lowers:
            title = "DFL Loss"
        elif "seg" in lowers or "mask" in lowers:
            title = "Seg Loss"
        else:
            title = prefix
        return train_col, val_col, title

    box_tr, box_val, box_title = find_loss("Box Loss", ["box"])
    cls_tr, cls_val, cls_title = find_loss("Class Loss", ["cls", "class"])
    obj_tr, obj_val, obj_title = find_loss("Object Loss", ["obj", "object", "dfl"])  # obj приоритетно, иначе dfl
    seg_tr, seg_val, seg_title = find_loss("Seg Loss", ["seg", "mask"])  # для сегментации

    # Настраиваем сетку: 1x1 сверху, 1x3 снизу (если есть seg — добавим 4-й справа)
    num_bottom = 3 + (1 if (seg_tr or seg_val) else 0)
    plt.figure(figsize=(14, 8))
    gs = gridspec.GridSpec(2, num_bottom, height_ratios=[2.0, 1.4])

    # Верх: mAP(ы)
    ax_top = plt.subplot(gs[0, :])
    if map50_col or map5095_col:
        if map50_col:
            ax_top.plot(df['epoch'], df[map50_col], label=map50_col)
        if map5095_col:
            ax_top.plot(df['epoch'], df[map5095_col], label=map5095_col)
        # Для контекста — Precision/Recall, если есть
        if precision_col:
            ax_top.plot(df['epoch'], df[precision_col], label=precision_col, linestyle='--', alpha=0.6)
        if recall_col:
            ax_top.plot(df['epoch'], df[recall_col], label=recall_col, linestyle='--', alpha=0.6)
        ax_top.set_title('Model Performance')
        ax_top.set_xlabel('epoch')
        ax_top.set_ylabel('score')
        ax_top.legend(loc='best', fontsize=8)
    else:
        ax_top.text(0.5, 0.5, 'No mAP columns found', ha='center', va='center')
        ax_top.set_axis_off()

    # Низ: лоссы
    def plot_loss(ax, train_col: str | None, val_col: str | None, title: str):
        if not (train_col or val_col):
            ax.text(0.5, 0.5, f'No data for {title}', ha='center', va='center')
            ax.set_axis_off()
            return
        if train_col:
            ax.plot(df['epoch'], df[train_col], label=f"train: {train_col}")
        if val_col:
            ax.plot(df['epoch'], df[val_col], label=f"val: {val_col}")
        ax.set_title(title)
        ax.set_xlabel('epoch')
        ax.set_ylabel('loss')
        ax.legend(loc='best', fontsize=8)

    ax_b0 = plt.subplot(gs[1, 0])
    plot_loss(ax_b0, box_tr, box_val, box_title)

    ax_b1 = plt.subplot(gs[1, 1])
    plot_loss(ax_b1, cls_tr, cls_val, cls_title)

    ax_b2 = plt.subplot(gs[1, 2])
    plot_loss(ax_b2, obj_tr, obj_val, obj_title)

    if num_bottom == 4:
        ax_b3 = plt.subplot(gs[1, 3])
        plot_loss(ax_b3, seg_tr, seg_val, seg_title)

    plt.tight_layout()
    out_png.parent.mkdir(parents=True, exist_ok=True)
    plt.savefig(out_png.as_posix())
    plt.close()
    print("[PLOT]", out_png.as_posix())
    return out_png


def print_metrics_summary(results_csv: Path) -> None:
    import pandas as pd
    if not results_csv.exists():
        return
    df = pd.read_csv(results_csv)
    if df.empty:
        return
    last = df.iloc[-1]
    # Выведем несколько ключевых метрик, если есть
    def get_any(keys: list[str]) -> float | None:
        for k in keys:
            if k in df.columns:
                v = last[k]
                try:
                    return float(v)
                except Exception:
                    return None
        return None

    p = get_any(['metrics/precision(M)', 'metrics/precision', 'metrics/precision(B)'])
    r = get_any(['metrics/recall(M)', 'metrics/recall', 'metrics/recall(B)'])
    m50 = get_any(['metrics/mAP50(M)', 'metrics/mAP50', 'metrics/mAP50(B)'])
    m5095 = get_any(['metrics/mAP50-95(M)', 'metrics/mAP50-95', 'metrics/mAP50-95(B)'])

    print("[SUMMARY] Metrics at last epoch:")
    if p is not None:
        print(f"  Precision: {p:.4f}")
    if r is not None:
        print(f"  Recall:    {r:.4f}")
    if m50 is not None:
        print(f"  mAP@50:    {m50:.4f}")
    if m5095 is not None:
        print(f"  mAP@50-95: {m5095:.4f}")

def main():
    ap = argparse.ArgumentParser(description="Fine-tune YOLO11 segmentation model on Colab with metrics plots")
    ap.add_argument("--zip", type=str, default="/content/models/pig 2.v2i.yolov11.zip", help="Путь к ZIP датасета (возможно с пробелами)")
    ap.add_argument("--base", type=str, default="/content/models/pig_yolo11-seg.pt", help="Путь к начальным весам .pt для дообучения")
    ap.add_argument("--epochs", type=int, default=300, help="Количество эпох")
    ap.add_argument("--batch", type=int, default=8, help="Batch size")
    ap.add_argument("--imgsz", type=int, default=640, help="Размер входного изображения")
    ap.add_argument("--device", type=str, default="0", help="GPU id ('0') или 'cpu'")
    ap.add_argument("--name", type=str, default="pig-finetune", help="Имя прогона (runs/segment/<name>)")
    ap.add_argument("--out", type=str, default="/content/pig_yolo11-seg-finetuned.pt", help="Куда скопировать итоговый best.pt")
    ap.add_argument("--save_to_drive", action="store_true", help="Скопировать чекпойнт на Google Drive, если смонтирован")
    # В среде Jupyter/Colab ядро может передавать служебный флаг -f <kernel.json>.
    # Используем parse_known_args(), чтобы игнорировать такие посторонние аргументы.
    args, _ = ap.parse_known_args()

    zip_path = Path(args.zip)
    base_path = Path(args.base)
    if not zip_path.exists():
        print(f"[ERROR] ZIP датасет не найден: {zip_path}", file=sys.stderr)
        sys.exit(2)
    if not base_path.exists():
        print(f"[ERROR] Начальные веса не найдены: {base_path}", file=sys.stderr)
        sys.exit(2)

    ensure_pkg_installation()

    datasets_root = Path("/content/datasets")
    ds_root = unzip_dataset(zip_path, datasets_root / "pig.v2i.yolov11")
    data_yaml = ensure_data_yaml(ds_root)

    # Построение команды обучения
    # Для ранней остановки можно добавить: "patience=30"
    train_cmd = [
        "yolo", "train",
        f"model={str(base_path)}",
        f"data={str(data_yaml)}",
        f"imgsz={args.imgsz}",
        f"epochs={args.epochs}",
        f"batch={args.batch}",
        f"device={args.device}",
        f"name={args.name}",
        "verbose=True",
    ]
    print("[INFO] Запускаю обучение:", " ".join(train_cmd), flush=True)
    code = run(train_cmd, check=False)
    if code != 0:
        print(f"[ERROR] yolo train завершился с кодом {code}", file=sys.stderr)
        sys.exit(code)

    # Поиск best.pt и результатов
    seg_dir = Path("/content/runs/segment")
    try:
        run_dir = find_latest_run(seg_dir, preferred_name=str(args.name))
    except FileNotFoundError as e:
        print("[ERROR]", e, file=sys.stderr)
        sys.exit(3)

    best = run_dir / "weights" / "best.pt"
    if not best.exists():
        print(f"[ERROR] best.pt не найден: {best}", file=sys.stderr)
        sys.exit(3)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(best, out_path)
    print(f"[INFO] Чекпойнт скопирован в: {out_path}", flush=True)

    # Генерация графиков
    results_csv = run_dir / "results.csv"
    custom_plots_dir = run_dir / "custom_plots"
    saved_custom = generate_custom_plots(results_csv, custom_plots_dir)

    # Единое полотно в стиле Roboflow
    dashboard_png = generate_roboflow_style_dashboard(results_csv, run_dir / "dashboard_roboflow_style.png")

    # Попытка показать встроенные картинки Ultralytics + наши
    builtin_imgs = [
        run_dir / "results.png",
        run_dir / "confusion_matrix.png",
        run_dir / "PR_curve.png",
        run_dir / "R_curve.png",
        run_dir / "P_curve.png",
        run_dir / "F1_curve.png",
        run_dir / "val_batch0_pred.jpg",
        run_dir / "val_batch1_pred.jpg",
        run_dir / "val_batch2_pred.jpg",
    ]
    extra_imgs = [dashboard_png] if dashboard_png else []
    to_show = [p for p in builtin_imgs + saved_custom + extra_imgs if p.exists()]
    if to_show:
        print("[INFO] Готовые графики/изображения:")
        for p in to_show:
            print(" -", p.as_posix())
        try_display_images(to_show)
    else:
        print("[WARN] Не удалось найти изображения для отображения. Проверьте каталог:", run_dir.as_posix())

    # Сохранение на Google Drive
    if args.save_to_drive:
        drive_root = Path("/content/drive/MyDrive")
        if drive_root.exists():
            dst = drive_root / Path(out_path.name)
            shutil.copy2(out_path, dst)
            print(f"[INFO] Чекпойнт также сохранён в Google Drive: {dst}", flush=True)
        else:
            print("[WARN] Google Drive не смонтирован. Пропускаю сохранение.", flush=True)

    # Сводка метрик последней эпохи
    print_metrics_summary(results_csv)

    print("[DONE] Дообучение в Colab завершено.")


if __name__ == "__main__":
    main()


