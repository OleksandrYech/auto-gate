#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-YOLO tester  —  PT / ONNX / TFLite-INT8

• Перевіряє декілька моделей за один запуск.
• Опція --show-all друкує результат кожного run (деталізація тесту).
• Детермінованість INT8: прапор --no-xnnpack і --threads 1.
• Очищення кешу/GC після кожної моделі через --clear-cache.

Залежності: opencv-python, numpy, ultralytics, onnxruntime,
tflite-runtime (або tensorflow), Python ≥ 3.8.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ────────────────────────── TFLite helpers ──────────────────────────────
def _dequant(arr: np.ndarray, det: dict) -> np.ndarray:
    if arr.dtype == np.float32:
        return arr
    qp = det["quantization_parameters"]
    zp, sc = qp["zero_points"], qp["scales"]
    # Додано перевірку для уникнення помилки, якщо sc або zp порожні або None
    if not isinstance(zp, np.ndarray) or not zp.size:
        zp = 0
    if not isinstance(sc, np.ndarray) or not sc.size:
        sc = 1.0
    return (arr.astype(np.float32) - zp) * sc


def _iou1d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    l, r = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, r - l)
    return inter / ((a[1] - a[0]) + (b[1] - b[0]) - inter + 1e-6)


def parse_tflite_out(interp, out_det, thr: float, names: Sequence[str],
                     in_w: int, iou_thr: float) -> List[Tuple[float, str, float]]:
    det_raw: List[Tuple[float, float, str, float]] = []

    # ① 4-тензорний out
    if len(out_det) == 4 and out_det[0]["shape"][-1] == 4:
        boxes = _dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])[0]
        scores = _dequant(interp.get_tensor(out_det[1]["index"]), out_det[1])[0]
        classes = _dequant(interp.get_tensor(out_det[2]["index"]), out_det[2])[0]
        n_det_tensor = interp.get_tensor(out_det[3]["index"])
        n = int(n_det_tensor[0]) if n_det_tensor.size > 0 else 0

        for j in range(n):
            cf, cid = float(scores[j]), int(classes[j])
            if cf < thr or cid >= len(names):
                continue
            x1, _, x2, _ = boxes[j]
            det_raw.append((x1, x2, names[cid], cf))

    # ②/③ — один тензор ≥6 атрибутів
    else:
        raw = _dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])
        if raw.ndim != 3:
            return []
        attrs = raw.shape[2]
        σ = lambda x: 1 / (1 + math.exp(-x))

        for row in raw[0]:
            # формат 6 attrs (x1 y1 x2 y2 conf cls)
            if attrs == 6:
                x1_val, _, x2_val, _, cf, cid_val = row  # Renamed to avoid conflict with x1, x2 below
                cf, cid = float(cf), int(cid_val)
                if cf < thr or cid >= len(names):
                    continue
                # Для формату з 6 атрибутами, x1 та x2 вже надані
                # Якщо вони нормалізовані, їх потрібно масштабувати. Якщо абсолютні - використовувати як є.
                # Припускаємо, що вони вже в потрібному масштабі або це буде оброблено пізніше.
                # Тут важливо, щоб x1, x2 були визначені для додавання в det_raw.
                # Якщо row містить абсолютні значення:
                # x1, x2 = float(x1_val), float(x2_val)
                # Якщо row містить нормалізовані значення cx, w (приклад, не для attrs==6):
                # Припускаємо, що для attrs==6 x1_val, x2_val це вже координати символів
                x1, x2 = float(row[0]), float(row[2])  # Або float(x1_val), float(x2_val) якщо вони є x-координатами
            else:  # ≥7 атрибутів (YOLO-5/8/11)
                # Для YOLO-подібних моделей, row[4] - objectness score (logit), row[5:] - class scores (logits)
                # num_classes = len(names) -> 36
                # attrs повинен бути 5 + num_classes = 41
                if attrs != (5 + len(names)) and attrs < 41:  # Додамо більш жорстку перевірку для attrs < 41
                    # Якщо це не очікуваний формат YOLO, можемо пропустити або обробити інакше
                    # print(f"Warning: Unexpected number of attributes {attrs} for YOLO-style output. Expected {5+len(names)}.")
                    obj_score_val = float(row[4])  # Припускаємо, що це вже confidence, а не logit
                    obj = obj_score_val  # без сігмоїди
                    if obj < 1e-6:  # використовуємо дуже мале значення для фільтрації
                        continue
                    # Припускаємо, що класи вже визначені або відсутні в цьому форматі
                    # Це є відхиленням від стандартного YOLO, обережно
                    cls_logits = row[5:attrs]  # Взяти всі доступні атрибути як можливі логіти класів
                    if not cls_logits.size:  # Якщо немає логітів класів
                        # print(f"Warning: No class logits for attrs={attrs}")
                        continue
                    cid = int(np.argmax(cls_logits))  # може бути невірним, якщо структура інша

                    # Якщо немає окремих логітів класів, cf може бути просто obj
                    # Або, якщо є один логіт класу після obj_score_val:
                    if len(cls_logits) > cid:
                        # Якщо cls_logits не є логітами, а вже ймовірностями, сігмоїда не потрібна
                        # cf = obj * float(cls_logits[cid]) # Без сігмоїди
                        # Якщо це все ж логіти, але ми не впевнені (attrs < 41)
                        cf = obj * σ(float(cls_logits[cid]))  # Залишаємо сігмоїду з обережністю
                    else:
                        # print(f"Warning: cid {cid} out of bounds for cls_logits length {len(cls_logits)}")
                        continue

                else:  # Це випадок attrs >= 41 (або attrs == 5 + len(names))
                    obj = σ(float(row[4]))  # obj_score logit
                    if obj < 1e-6:  # Поріг для objectness score після сигмоїди
                        continue
                    cls_logits = row[5:5 + len(names)]
                    cid = int(np.argmax(cls_logits))
                    cf = obj * σ(float(cls_logits[cid]))  # conf = obj_score * class_score (обидва після сигмоїди)

                if cf < thr or cid >= len(names):
                    continue

                # cx, w зазвичай нормалізовані для YOLO
                cx, w = float(row[0]) * in_w, float(row[2]) * in_w
                if w < 2:  # мінімальна ширина символу
                    continue
                x1, x2 = cx - w / 2, cx + w / 2

            det_raw.append((x1, x2, names[cid], cf))

    if not det_raw:
        return []

    det_raw.sort(key=lambda d: d[3], reverse=True)
    keep: List[Tuple[float, float, str, float]] = []
    for cand in det_raw:
        if all(_iou1d((cand[0], cand[1]), (k[0], k[1])) <= iou_thr for k in keep):
            keep.append(cand)

    keep.sort(key=lambda d: (d[0] + d[1]) / 2)  # left→right
    return [((k[0] + k[1]) / 2, k[2], k[3]) for k in keep]  # Повертаємо центр символу, символ, впевненість


# ───────────────────────────── misc ─────────────────────────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def preprocess(img: np.ndarray, in_det: List[dict], img_sz: int) -> np.ndarray:
    shp, dtype = in_det[0]["shape"], in_det[0]["dtype"]
    nchw = shp[1] == 3
    h = shp[2] if nchw else shp[1]
    w = shp[3] if nchw else shp[2]

    # Якщо h або w не визначені з форми тензора (може бути -1), використовуємо img_sz
    h = h if h and h > 0 else img_sz
    w = w if w and w > 0 else img_sz

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    if nchw:
        img = img.transpose(2, 0, 1)
    img = img[None]

    if dtype in (np.int8, np.uint8):
        qp = in_det[0]["quantization_parameters"]
        scale = qp["scales"][0] if qp["scales"].size > 0 else 1.0
        zero_point = qp["zero_points"][0] if qp["zero_points"].size > 0 else 0
        if scale == 0: scale = 1.0  # Запобігання діленню на нуль

        img = (img / scale + zero_point)
        if dtype == np.int8:
            img = np.clip(img, -128, 127)  # Кліпування для int8
        elif dtype == np.uint8:
            img = np.clip(img, 0, 255)  # Кліпування для uint8
        img = img.astype(dtype)
    return img.astype(dtype)  # Переконуємось, що тип даних остаточний


def order_string(det: List[Tuple[float, str, float]]) -> Tuple[str, float]:
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0
    det.sort(key=lambda d: d[0])  # d[0] - це x-координата центру символу
    chars, confs = zip(*[(c, cf) for _, c, cf in det])
    return "".join(chars), float(np.mean(confs))


def load_names(pt_path: str | None) -> Sequence[str]:
    if pt_path:
        try:
            model_yolo = YOLO(pt_path)
            if model_yolo.names:
                # model.names повертає словник {id: name}, нам потрібен список імен за індексом
                return [model_yolo.names[i] for i in sorted(model_yolo.names.keys())]
        except Exception as e:
            # print(f"Не вдалося завантажити імена з {pt_path}: {e}")
            pass
    return CLASS_NAMES


# ───────────────────── back-ends with per-run print ─────────────────────
def run_pt_or_onnx(path: str, img: np.ndarray, names: Sequence[str], thr: float,
                   runs: int, show_all: bool) -> Tuple[str, float, float]:
    model = YOLO(path)  # Завантаження моделі ONNX або PT
    _ = model(img, verbose=False, conf=thr)  # warm-up

    times, det_final = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        res = model(img, verbose=False, conf=thr)[0]
        dt = time.perf_counter() - t0
        times.append(dt)

        det_now = []
        if res.boxes:
            for b in res.boxes:
                cf, cid = float(b.conf.squeeze()), int(b.cls.squeeze())
                if cf < thr or cid >= len(names):  # Перевірка cid відносно довжини names
                    continue
                # b.xywh дає [cx, cy, w, h] в абсолютних координатах зображення
                # Нам потрібна x-координата для сортування, наприклад, cx
                x_coord = float(b.xywh.squeeze()[0])
                det_now.append((x_coord, names[cid], cf))

        plate_now, conf_now = order_string(det_now)
        if show_all:
            print(f"      run {i + 1}/{runs}: {plate_now:<15} "
                  f"conf {conf_now:.3f}  {dt * 1000:.1f} ms")
        if i == runs - 1:  # save last run result
            det_final = det_now

    plate_fin, conf_fin = order_string(det_final)
    return plate_fin, conf_fin, float(np.mean(times))


def run_tflite(path: str, img: np.ndarray, names: Sequence[str], thr: float,
               runs: int, img_sz: int, iou_thr: float,
               no_xnnpack: bool, threads: int, show_all: bool) -> Tuple[str, float, float]:
    try:
        # Спробуємо імпортувати з ai_edge_litert, якщо доступно
        from ai_edge_litert.interpreter import Interpreter  # type: ignore
    except ImportError:
        # Якщо ні, використовуємо стандартний tflite_runtime
        from tflite_runtime.interpreter import Interpreter  # type: ignore

    kwargs = {"model_path": path, "num_threads": threads}
    if no_xnnpack:
        kwargs["experimental_delegates"] = []
    else:
        # Якщо XNNPACK дозволено, можна його явно додати, хоча зазвичай він використовується за замовчуванням
        # from tflite_runtime.interpreter import load_delegate
        # kwargs["experimental_delegates"] = [load_delegate('libxnnpack.so')] # Назва бібліотеки може відрізнятися
        pass

    it = Interpreter(**kwargs)
    it.allocate_tensors()

    in_det, out_det = it.get_input_details(), it.get_output_details()
    inp = preprocess(img, in_det, img_sz)

    it.set_tensor(in_det[0]["index"], inp)
    it.invoke()  # warm-up

    times, det_final = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        it.set_tensor(in_det[0]["index"], inp)
        it.invoke()
        dt = time.perf_counter() - t0
        times.append(dt)

        det_now = parse_tflite_out(it, out_det, thr, names,
                                   in_w=img_sz, iou_thr=iou_thr)  # in_w=img_sz для нормалізованих координат
        plate_now, conf_now = order_string(det_now)
        if show_all:
            print(f"      run {i + 1}/{runs}: {plate_now:<15} "
                  f"conf {conf_now:.3f}  {dt * 1000:.1f} ms")
        if i == runs - 1:
            det_final = det_now

    plate_fin, conf_fin = order_string(det_final)
    return plate_fin, conf_fin, float(np.mean(times))


# ────────────────────────────── CLI ─────────────────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("OCR-YOLO tester (PT / ONNX / TFLite-INT8)")
    p.add_argument("-m", "--model", dest="model_paths", action="append",
                   required=True, help="Шляхи до моделей (.pt / .onnx / .tflite)")
    p.add_argument("--image", required=True, help="Тестове зображення")
    p.add_argument("--input_size", type=int, default=320, help="Розмір інпуту (ширина) для TFLite")
    p.add_argument("--conf", type=float, default=0.12, help="Поріг conf")
    p.add_argument("--iou_thr", type=float, default=0.35, help="IoU-NMS поріг для TFLite")
    p.add_argument("--runs", type=int, default=5, help="К-сть запусків для avg")
    p.add_argument("--no-xnnpack", action="store_true",
                   help="Вимкнути XNNPACK delegate для TFLite")
    p.add_argument("--threads", type=int, default=1,  # ЗМІНЕНО: Значення за замовчуванням тепер 1
                   help="Потоки для TFLite (рекомендується 1 для детермінізму з INT8 та --no-xnnpack)")
    p.add_argument("--clear-cache", action="store_true",
                   help="Очищати кеш/GC після кожної моделі")
    p.add_argument("--show-all", action="store_true",
                   help="Показувати розпізнання кожного run")
    args = p.parse_args()

    if args.no_xnnpack and args.threads != 1:
        print("Warning: Для детермінізму з --no-xnnpack рекомендується використовувати --threads 1.")

    return args


# ────────────────────────────── main ────────────────────────────────────
if __name__ == "__main__":
    args = cli()

    for f_path in args.model_paths + [args.image]:
        if not os.path.exists(f_path):
            sys.exit(f"Файл не знайдено: {f_path}")

    img_orig = cv2.imread(args.image)
    if img_orig is None:
        sys.exit(f"Не вдалося відкрити {args.image}")

    # Завантаження імен класів. Спробувати з .pt моделі, якщо надано, інакше стандартні.
    pt_model_path_for_names = next((p for p in args.model_paths if p.endswith(".pt")), None)
    # Якщо .pt не надано, але є інші моделі, load_names використає CLASS_NAMES
    class_names_to_use = load_names(pt_model_path_for_names)
    if not class_names_to_use or len(class_names_to_use) != 36:  # Перевірка, що імена завантажено коректно
        # print(f"Warning: Не вдалося завантажити коректний список імен класів. Використовуються стандартні {len(CLASS_NAMES)} класів.")
        class_names_to_use = CLASS_NAMES

    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    for model_path_arg in args.model_paths:
        model_file_name = os.path.basename(model_path_arg)
        model_ext = os.path.splitext(model_file_name)[1].lower()
        print(f"[{model_file_name:>20}]")  # Збільшено ширину для кращого вирівнювання

        plate_res, conf_res, time_res = "ERROR", 0.0, 0.0

        try:
            if model_ext in (".pt", ".onnx"):
                plate_res, conf_res, time_res = run_pt_or_onnx(
                    model_path_arg, img_orig.copy(), class_names_to_use,
                    args.conf, args.runs, args.show_all
                )

            elif model_ext == ".tflite":
                plate_res, conf_res, time_res = run_tflite(
                    model_path_arg, img_orig.copy(), class_names_to_use,
                    args.conf, args.runs,
                    img_sz=args.input_size,  # Передаємо input_size як img_sz
                    iou_thr=args.iou_thr,
                    no_xnnpack=args.no_xnnpack,
                    threads=args.threads,
                    show_all=args.show_all
                )
            else:
                print("  ❌ Формат не підтримується.")
                continue

            print(f"  ↪  Plate: {plate_res:<15}  Avg conf: {conf_res:.3f}  "
                  f"Avg time: {time_res * 1000:.1f} ms\n")

        except Exception as e:
            print(f"  ❌ Помилка при обробці моделі {model_file_name}: {e}")
            # Можна додати traceback для детальної діагностики, якщо потрібно
            # import traceback
            # traceback.print_exc()
            print("")  # Додатковий порожній рядок для відокремлення помилок

        if args.clear_cache:
            # Спроба звільнити пам'ять від змінних результатів
            del plate_res, conf_res, time_res
            gc.collect()

    print("====================================\n")