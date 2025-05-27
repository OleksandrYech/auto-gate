#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-YOLO tester (PT / ONNX / TFLite-LiteRT)

• Автоматично обробляє три типи TFLite-виходу:
  ① 4-тензорний   ② raw 41-атрибут   ③ post-sigmoid 7-атрибут.
• 1-D IoU-NMS вздовж X прибирає «стіну нулів».

Використання приклад:

    python3 ocr_test.py --model ocr.pt/onnx/tflite --image image.png --input_size 320 --conf 0.12 --iou_thr 0.35 --runs 5

Залежності: opencv-python, numpy, (tflite_runtime | tensorflow), ultralytics,
onnxruntime, ai_edge_litert.  Python ≥ 3.8.
"""

from __future__ import annotations
import argparse, math, os, sys, time
from typing import List, Sequence, Tuple
import cv2
import numpy as np
from ultralytics import YOLO
import shutil  # Додано для очищення директорій
from pathlib import Path  # Додано для роботи зі шляхами кешу


# ─────────── parse_tflite_out ───────────────────────────────────────────────
def parse_tflite_out(interp, od, thr: float, names: Sequence[str],
                     img_w: int, iou_thr: float):
    """Повертає [(x_center_px, char, conf), …] для LiteRT."""

    def deq(a, d):
        """Деквантизація тензора."""
        if a.dtype == np.float32:
            return a
        qp = d["quantization_parameters"]
        # Переконуємося, що zero_points та scales є скалярами або мають сумісні розміри
        zero_points = qp["zero_points"]
        scales = qp["scales"]
        if isinstance(zero_points, (np.ndarray, list)) and len(zero_points) > 0:
            zero_points = zero_points[0]
        if isinstance(scales, (np.ndarray, list)) and len(scales) > 0:
            scales = scales[0]
        return (a.astype(np.float32) - zero_points) * scales

    def iou1d(a1: Tuple[float, float], a2: Tuple[float, float]) -> float:
        """Розрахунок 1D Intersection over Union."""
        l, r = max(a1[0], a2[0]), min(a1[1], a2[1])
        inter = max(0.0, r - l)
        return inter / ((a1[1] - a1[0]) + (a2[1] - a2[0]) - inter + 1e-6)

    det_raw: List[Tuple[float, float, str, float]] = []  # (x1, x2, char_name, confidence)

    # 1) 4-тензорний вихід (зазвичай boxes, scores, classes, num_detections)
    if len(od) == 4 and od[0]["shape"][-1] == 4:  # Припускаємо, що перший тензор - це boxes [1, N, 4]
        boxes = deq(interp.get_tensor(od[0]["index"]), od[0])[0]  # [N, 4]
        scores = deq(interp.get_tensor(od[1]["index"]), od[1])[0]  # [N]
        classes = deq(interp.get_tensor(od[2]["index"]), od[2])[0]  # [N]
        # Останній тензор може бути кількістю детекцій
        n = int(deq(interp.get_tensor(od[3]["index"]), od[3])[0])
        for j in range(n):
            cf = float(scores[j])
            cid = int(classes[j])
            if cf < thr or cid >= len(names): continue
            x1, _, x2, _ = boxes[j]  # Використовуємо тільки x-координати для 1D OCR
            det_raw.append((x1 * img_w, x2 * img_w, names[cid], cf))  # Масштабуємо, якщо координати відносні (0-1)

    # 2) один тензор на виході
    else:
        raw = deq(interp.get_tensor(od[0]["index"]), od[0])
        if raw.ndim != 3 or raw.shape[0] != 1:  # Очікуємо [1, N, attrs]
            print(f"Неочікувана форма вихідного тензора: {raw.shape}", file=sys.stderr)
            return []
        raw = raw[0]  # [N, attrs]
        attrs = raw.shape[1]  # Кількість атрибутів

        # 2.a) 6-атрибутний вихід (наприклад, x1, y1, x2, y2, conf, class_id)
        if attrs == 6:
            for row in raw:  # x1, y1, x2, y2, cf, cid
                cf = float(row[4])
                cid = int(row[5])
                if cf < thr or cid >= len(names): continue
                x1, x2 = float(row[0]) * img_w, float(row[2]) * img_w  # Масштабуємо
                det_raw.append((x1, x2, names[cid], cf))

        # 2.b) 7-атрибутний (post-sigmoid) або 41-атрибутний (raw, для 36 класів + 5 параметрів)
        elif attrs >= 7:
            sigm = lambda x: 1 / (1 + math.exp(-x))
            for row in raw:
                if attrs == 7:  # [cx, cy, w, h, conf, cls_id, cls_conf] or [cx,w,cy,h,conf,cls_id]
                    # Скрипт припускає: [cx, ?, w, ?, conf, cls_id]
                    cf = float(row[4])
                    cid = int(row[5])
                    if cf < thr or cid >= len(names): continue
                    # Координати cx, w вже можуть бути абсолютними або відносними
                    # Якщо відносні, їх треба множити на img_w
                    # Оригінальний код: cx, w = float(row[0])*img_w, float(row[2])*img_w
                    # Якщо вони вже абсолютні після post-sigmoid, множення не потрібне.
                    # Залишаємо як в оригіналі, припускаючи, що вони відносні або уніфіковані.
                    cx, w = float(row[0]) * img_w, float(row[2]) * img_w
                else:  # raw N-attrib (e.g., 41: cx,cy,w,h,obj_conf, cls_scores...)
                    obj_conf = sigm(float(row[4]))
                    if obj_conf < 1e-6: continue  # Низька впевненість в об'єкті

                    # Класи починаються з 5-го індексу
                    class_scores = row[5: 5 + len(names)]
                    cid = int(np.argmax(class_scores))
                    max_class_score = sigm(float(class_scores[cid]))
                    cf = obj_conf * max_class_score

                    if cf < thr or cid >= len(names): continue
                    # cx, cy, w, h - зазвичай перші 4 значення
                    cx, w = float(row[0]) * img_w, float(row[2]) * img_w
                if w < 2: continue  # Дуже вузький символ
                x1, x2 = cx - w / 2, cx + w / 2
                det_raw.append((x1, x2, names[cid], cf))
        else:
            print(f"Непідтримувана кількість атрибутів ({attrs}) у вихідному тензорі.", file=sys.stderr)
            return []

    if not det_raw:
        return []

    # NMS (Non-Maximum Suppression)
    det_raw.sort(key=lambda d: d[3], reverse=True)  # Сортування за впевненістю (conf)
    keep: List[Tuple[float, float, str, float]] = []
    for cand in det_raw:  # (x1, x2, name, conf)
        # Порівнюємо поточного кандидата з уже відібраними
        if all(iou1d((cand[0], cand[1]), (k[0], k[1])) <= iou_thr for k in keep):
            keep.append(cand)

    # Сортування за центральною x-координатою для читабельного порядку
    keep.sort(key=lambda d: (d[0] + d[1]) / 2)
    return [((k[0] + k[1]) / 2, k[2], k[3]) for k in keep]  # (x_center, char, conf)


# ─────────── решта допоміжних функцій ───────────────────────────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")  # 36 класів


def preprocess(img: np.ndarray, inp_det: List[dict], img_sz: int) -> np.ndarray:
    """Попередня обробка зображення для моделі."""
    # Отримуємо параметри вхідного тензора моделі
    shp, dtype = inp_det[0]["shape"], inp_det[0]["dtype"]
    # Визначаємо, чи канали йдуть першими (NCHW) чи останніми (NHWC)
    nchw = shp[1] == 3 if len(shp) == 4 else False  # NCHW if shp=[B,C,H,W] and C=3

    # Визначаємо висоту та ширину для зміни розміру
    # Якщо shp має 4 виміри (B,C,H,W або B,H,W,C)
    if len(shp) == 4:
        h = shp[2] if nchw else shp[1]
        w = shp[3] if nchw else shp[2]
    # Якщо shp має 3 виміри (наприклад, для деяких моделей без батча H,W,C)
    elif len(shp) == 3:
        # Припускаємо H,W,C або C,H,W - потрібно уточнити або зробити гнучкіше
        # Для простоти, якщо C=3 першим, то C,H,W, інакше H,W,C
        if shp[0] == 3:  # C,H,W
            h, w = shp[1], shp[2]
            nchw = True  # Встановлюємо NCHW вручну, хоча батча немає
        else:  # H,W,C
            h, w = shp[0], shp[1]
            nchw = False
    else:  # Невідома форма, використовуємо img_sz
        h = w = img_sz

    h = h or img_sz  # Якщо h=0 (не визначено з shp), використовуємо img_sz
    w = w or img_sz  # Якщо w=0 (не визначено з shp), використовуємо img_sz

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0  # Нормалізація до [0, 1]

    if nchw:
        img = img.transpose(2, 0, 1)  # HWC to CHW

    img = img[None]  # Додаємо вимір батча: CHW -> BCHW або HWC -> BHWC

    # Квантизація, якщо модель очікує INT8/UINT8
    if dtype in (np.int8, np.uint8):
        qp = inp_det[0]["quantization_parameters"]
        scale = qp["scales"][0] if isinstance(qp["scales"], (list, np.ndarray)) and len(qp["scales"]) > 0 else 1.0
        zero_point = qp["zero_points"][0] if isinstance(qp["zero_points"], (list, np.ndarray)) and len(
            qp["zero_points"]) > 0 else 0

        if scale == 0:  # Уникаємо ділення на нуль, хоча це не повинно траплятися
            print("Warning: Quantization scale is zero. Using 1.0 instead.", file=sys.stderr)
            scale = 1.0

        img = (img / scale + zero_point).astype(dtype)
    return img.astype(dtype)


def order_string(det: List[Tuple[float, str, float]]) -> Tuple[str, float]:
    """Формує рядок з розпізнаних символів та середню впевненість."""
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0
    # det вже відсортований за x-координатою з parse_tflite_out (для TFLite)
    # Для PT/ONNX сортування відбувається у run_pt_or_onnx перед викликом
    chars, confs = zip(*[(c, cf) for _, c, cf in det])
    return "".join(chars), float(np.mean(confs)) if confs else 0.0


def load_names(pt_path: str | None) -> List[str]:
    """Завантажує імена класів з .pt моделі або повертає стандартні."""
    if pt_path and pt_path.endswith(".pt"):
        try:
            model_names = YOLO(pt_path).names
            if model_names and isinstance(model_names, dict):  # YOLO names are dict: {index: name}
                # Сортуємо за індексами, щоб отримати список імен у правильному порядку
                return [model_names[i] for i in sorted(model_names.keys())]
            elif model_names and isinstance(model_names, list):  # Якщо вже список
                return model_names
        except Exception as e:
            print(f"Не вдалося завантажити імена класів з {pt_path}: {e}", file=sys.stderr)
            pass  # Використовуємо стандартні імена
    return CLASS_NAMES


# ─────────── back-ends ──────────────────────────────────────────────────────
def run_pt_or_onnx(path: str, img: np.ndarray, names: List[str],
                   thr: float, runs: int) -> Tuple[str, float, float]:
    """Запускає інференс для PyTorch або ONNX моделей."""
    m = YOLO(path)
    # Прогрів
    _ = m(img, verbose=False, conf=thr, iou=0.5)  # iou для NMS в YOLO, можна налаштувати

    times: List[float] = []
    det_for_order: List[Tuple[float, str, float]] = []  # (x_center, char, conf)

    for i in range(runs):
        t0 = time.perf_counter()
        results = m(img, verbose=False, conf=thr, iou=0.5)  # Використовуємо той самий iou
        times.append(time.perf_counter() - t0)

        if i == runs - 1 and results and results[0].boxes:  # Обробляємо результати останнього запуску
            boxes = results[0].boxes
            for b in boxes:
                cf = float(b.conf.squeeze())
                cid = int(b.cls.squeeze())
                if cf < thr or cid >= len(names):
                    continue
                # b.xywh містить [x_center, y_center, width, height]
                # Нам потрібен x_center для сортування
                x_center = float(b.xywh.squeeze()[0])
                det_for_order.append((x_center, names[cid], cf))

    # Сортуємо детекції за x-координатою перед формуванням рядка
    det_for_order.sort(key=lambda d: d[0])
    plate_str, avg_conf = order_string(det_for_order)
    return plate_str, avg_conf, np.mean(times) if times else 0.0


def run_litert(path: str, img: np.ndarray, names: List[str],
               thr: float, runs: int, img_sz: int, iou_thr: float) -> Tuple[str, float, float]:
    """Запускає інференс для TFLite моделей через LiteRT."""
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        print("Помилка: Бібліотека 'ai_edge_litert' не знайдена. Встановіть її для роботи з TFLite.", file=sys.stderr)
        return "ПОМИЛКА ІМПОРТУ", 0.0, 0.0

    it = Interpreter(model_path=path)
    it.allocate_tensors()
    idet, odet = it.get_input_details(), it.get_output_details()

    inp = preprocess(img, idet, img_sz)

    # Прогрів
    it.set_tensor(idet[0]["index"], inp)
    it.invoke()

    times: List[float] = []
    final_detections: List[Tuple[float, str, float]] = []  # (x_center, char, conf)

    for i in range(runs):
        t0 = time.perf_counter()
        it.set_tensor(idet[0]["index"], inp)
        it.invoke()
        times.append(time.perf_counter() - t0)

        if i == runs - 1:  # Обробляємо результати останнього запуску
            final_detections = parse_tflite_out(it, odet, thr, names, img_w=img_sz,  # Передаємо img_sz як img_w
                                                iou_thr=iou_thr)

    # parse_tflite_out вже сортує детекції, тому додаткове сортування не потрібне
    plate_str, avg_conf = order_string(final_detections)
    return plate_str, avg_conf, np.mean(times) if times else 0.0


# ─────────── Функції очищення кешу ─────────────────────────────────────────
def clear_pycache(directory: str = "."):
    """Рекурсивно видаляє директорії __pycache__ та файли .pyc."""
    count_dirs = 0
    count_files = 0
    for root, dirs, files_in_dir in os.walk(directory):
        if "__pycache__" in dirs:
            pycache_dir = os.path.join(root, "__pycache__")
            print(f"Видалення {pycache_dir}...")
            try:
                shutil.rmtree(pycache_dir)
                count_dirs += 1
            except OSError as e:
                print(f"Помилка видалення {pycache_dir}: {e}", file=sys.stderr)

        for file_item in files_in_dir:
            if file_item.endswith(".pyc"):
                pyc_file = os.path.join(root, file_item)
                print(f"Видалення {pyc_file}...")
                try:
                    os.remove(pyc_file)
                    count_files += 1
                except OSError as e:
                    print(f"Помилка видалення {pyc_file}: {e}", file=sys.stderr)
    print(f"Очищення __pycache__ завершено. Видалено директорій: {count_dirs}, файлів: {count_files}.")


def get_ultralytics_cache_dir() -> Path:
    """Повертає типовий шлях до кешу Ultralytics."""
    # Відповідно до XDG Base Directory Specification
    xdg_cache_home = os.getenv('XDG_CACHE_HOME')
    if xdg_cache_home:
        cache_home = Path(xdg_cache_home)
    else:
        # Стандартні шляхи для різних ОС
        if sys.platform == "win32":
            cache_home = Path.home() / "AppData" / "Local"
        elif sys.platform == "darwin":  # macOS
            cache_home = Path.home() / "Library" / "Caches"
        else:  # Linux та інші Unix-подібні
            cache_home = Path.home() / ".cache"
    return cache_home / "Ultralytics"


# ─────────── CLI та main ────────────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser(description="OCR-YOLO tester (PT / ONNX / TFLite-LiteRT)",
                                formatter_class=argparse.RawTextHelpFormatter)
    p.add_argument("-m", "--model", "--path", dest="model_paths",
                   action="append", required=True,
                   help="Шлях до файлу моделі (.pt, .onnx, .tflite).\nМожна вказати декілька моделей.")
    p.add_argument("--image", required=True, help="Шлях до зображення для розпізнавання.")
    p.add_argument("--input_size", type=int, default=320,
                   help="Розмір входу для моделі (ширина, якщо TFLite, інакше сторона квадрата).")
    p.add_argument("--conf", type=float, default=0.12, help="Поріг впевненості (confidence threshold).")
    p.add_argument("--iou_thr", type=float, default=0.35,
                   help="Поріг IoU для NMS (для TFLite).")
    p.add_argument("--runs", type=int, default=5, help="Кількість запусків для вимірювання середнього часу.")
    p.add_argument("--clear_cache", action="store_true",
                   help="Очистити локальний __pycache__ та показати шлях до кешу Ultralytics.")
    return p.parse_args()


if __name__ == "__main__":
    args = cli()

    if args.clear_cache:
        print("Розпочато очищення кешу...")
        clear_pycache()
        ultralytics_cache_dir = get_ultralytics_cache_dir()
        print(f"\nКеш Ultralytics зазвичай знаходиться тут: {ultralytics_cache_dir}")
        print("Ви можете видалити його вручну, якщо потрібно. Наприклад:")
        if sys.platform == "win32":
            print(f'  rd /s /q "{ultralytics_cache_dir}"')
        else:
            print(f'  rm -rf "{ultralytics_cache_dir}"')
        print("\nРоботу завершено після очищення кешу.")
        sys.exit(0)

    for f_path in args.model_paths + [args.image]:
        if not os.path.exists(f_path):
            sys.exit(f"Файл не знайдено: {f_path}")

    img_orig = cv2.imread(args.image)
    if img_orig is None:
        sys.exit(f"Не вдалося відкрити зображення: {args.image}")

    # Завантажуємо імена класів. Пріоритет .pt моделі, якщо є.
    pt_first_path = next((p for p in args.model_paths if p.endswith(".pt")), None)
    class_names = load_names(pt_first_path)
    if not class_names or len(class_names) == 0:  # Додаткова перевірка
        print("Попередження: не вдалося завантажити імена класів, використовуються стандартні.", file=sys.stderr)
        class_names = CLASS_NAMES

    print(
        f"\nВикористовуються імена класів (всього {len(class_names)}): {''.join(class_names[:10])}...{''.join(class_names[-5:]) if len(class_names) > 15 else ''}")
    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    for model_path in args.model_paths:
        model_basename = os.path.basename(model_path)
        ext = os.path.splitext(model_path)[1].lower()

        plate_text, confidence, avg_time_ms = "", 0.0, 0.0

        try:
            if ext in (".pt", ".onnx"):
                plate_text, confidence, avg_time = run_pt_or_onnx(
                    model_path, img_orig, class_names, args.conf, args.runs
                )
                avg_time_ms = avg_time * 1000
            elif ext == ".tflite":
                plate_text, confidence, avg_time = run_litert(
                    model_path, img_orig, class_names, args.conf, args.runs,
                    img_sz=args.input_size, iou_thr=args.iou_thr
                )
                avg_time_ms = avg_time * 1000
            else:
                print(f"[{model_basename:>20}]  ❌ Формат не підтримується.")
                continue

            print(f"[{model_basename:>20}]  Plate: {plate_text:<15} "
                  f"Avg conf: {confidence:.3f}  Time: {avg_time_ms:.1f} ms")

        except Exception as e:
            print(f"[{model_basename:>20}]  ❌ Помилка обробки: {e}", file=sys.stderr)
            import traceback

            traceback.print_exc()

    print("====================================\n")