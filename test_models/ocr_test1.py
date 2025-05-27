#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-YOLO tester  —  PT / ONNX / TFLite-INT8

• Підтримує три формати виходу TFLite-моделей:
  ① 4-тензорний, ② raw 41-атрибут, ③ post-sigmoid 7-атрибут.
• Одновимірний IoU-NMS уздовж осі X прибирає «стіни нулів».
• Повністю детермінований режим для INT8 (—no-xnnpack, 1 thread).
• Опційне очищення кешу після кожного запуску (—clear-cache).

Використання приклад:

    python3 ocr_test.py \
        --model ocr.pt ocr.onnx ocr_int8.tflite \
        --image test.png --input_size 320 \
        --conf 0.2 --iou_thr 0.35 --runs 10 \
        --no-xnnpack --threads 1 --clear-cache

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

# ───────────────────────────── TFLite utils ────────────────────────────────
def _dequant(arr: np.ndarray, det: dict) -> np.ndarray:
    """Зворотне квантування INT8/UINT8 тензора."""
    if arr.dtype == np.float32:
        return arr
    qp = det["quantization_parameters"]
    zp = qp["zero_points"]
    sc = qp["scales"]
    return (arr.astype(np.float32) - zp) * sc


def _iou1d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    """IoU на прямій."""
    l, r = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, r - l)
    return inter / ((a[1] - a[0]) + (b[1] - b[0]) - inter + 1e-6)


def parse_tflite_out(interp, out_det, thr: float, names: Sequence[str],
                     in_w: int, iou_thr: float) -> List[Tuple[float, str, float]]:
    """Повертає [(x_center_px, char, conf), …] зі всіх доступних форматів."""
    det_raw: List[Tuple[float, float, str, float]] = []

    # ① 4-тензорний вихід — boxes, scores, classes, n
    if len(out_det) == 4 and out_det[0]["shape"][-1] == 4:
        boxes = _dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])[0]
        scores = _dequant(interp.get_tensor(out_det[1]["index"]), out_det[1])[0]
        classes = _dequant(interp.get_tensor(out_det[2]["index"]), out_det[2])[0]
        n = int(interp.get_tensor(out_det[3]["index"])[0])

        for j in range(n):
            cf, cid = float(scores[j]), int(classes[j])
            if cf < thr or cid >= len(names):
                continue
            x1, _, x2, _ = boxes[j]
            det_raw.append((x1, x2, names[cid], cf))

    # ②/③ — один тензор (6, 7 або 41 атрибут)
    else:
        raw = _dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])
        if raw.ndim != 3:
            return []
        attrs = raw.shape[2]
        σ = lambda x: 1 / (1 + math.exp(-x))

        for row in raw[0]:
            if attrs == 6:                       # x1 y1 x2 y2 conf cls
                x1, _, x2, _, cf, cid = row
                cf, cid = float(cf), int(cid)
                if cf < thr or cid >= len(names):
                    continue

            else:                                # ≥7 атрибутів
                obj = σ(float(row[4])) if attrs >= 41 else float(row[4])
                if obj < 1e-6:
                    continue

                cls_logits = row[5:5 + len(names)]
                cid = int(np.argmax(cls_logits))
                cf = obj * (σ(float(cls_logits[cid])) if attrs >= 41 else 1.0)
                if cf < thr or cid >= len(names):
                    continue

                cx, w = float(row[0]) * in_w, float(row[2]) * in_w
                if w < 2:
                    continue
                x1, x2 = cx - w / 2, cx + w / 2

            det_raw.append((x1, x2, names[cid], cf))

    if not det_raw:
        return []

    # 1-D IoU-NMS
    det_raw.sort(key=lambda d: d[3], reverse=True)
    keep: List[Tuple[float, float, str, float]] = []
    for cand in det_raw:
        if all(_iou1d((cand[0], cand[1]), (k[0], k[1])) <= iou_thr for k in keep):
            keep.append(cand)

    keep.sort(key=lambda d: (d[0] + d[1]) / 2)       # ліворуч-→праворуч
    return [((k[0] + k[1]) / 2, k[2], k[3]) for k in keep]


# ────────────────────────────── helpers ────────────────────────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def preprocess(img: np.ndarray, in_det: List[dict], img_sz: int) -> np.ndarray:
    """Підготовка зображення (NCHW/NHWC, float32/int8/uint8)."""
    shp, dtype = in_det[0]["shape"], in_det[0]["dtype"]
    nchw = shp[1] == 3
    h = shp[2] if nchw else shp[1] or img_sz
    w = shp[3] if nchw else shp[2] or img_sz

    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = img.astype(np.float32) / 255.0
    if nchw:
        img = img.transpose(2, 0, 1)
    img = img[None]

    if dtype in (np.int8, np.uint8):
        qp = in_det[0]["quantization_parameters"]
        img = (img / (qp["scales"][0] or 1.0) + qp["zero_points"][0]).astype(dtype)

    return img.astype(dtype)


def order_string(det: List[Tuple[float, str, float]]) -> Tuple[str, float]:
    """Сортує символи ліворуч→праворуч, повертає строку та середню впевненість."""
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0
    det.sort(key=lambda d: d[0])
    chars, confs = zip(*[(c, cf) for _, c, cf in det])
    return "".join(chars), float(np.mean(confs))


def load_names(pt_path: str | None) -> Sequence[str]:
    """Читає список класів із .pt-моделі або повертає стандартний."""
    if pt_path:
        try:
            names = YOLO(pt_path).names
            if names:
                return names
        except Exception:
            pass
    return CLASS_NAMES


# ───────────────────────────── back-ends ───────────────────────────────────
def run_pt_or_onnx(path: str, img: np.ndarray, names: Sequence[str],
                   thr: float, runs: int) -> Tuple[str, float, float]:
    model = YOLO(path)
    _ = model(img, verbose=False, conf=thr)           # прогрів
    times, det = [], []

    for i in range(runs):
        t0 = time.perf_counter()
        res = model(img, verbose=False, conf=thr)[0]
        times.append(time.perf_counter() - t0)
        if i == runs - 1 and res.boxes:
            for b in res.boxes:
                cf, cid = float(b.conf.squeeze()), int(b.cls.squeeze())
                if cf < thr or cid >= len(names):
                    continue
                det.append((float(b.xywh.squeeze()[0]), names[cid], cf))

    return *order_string(det), np.mean(times)


def run_tflite(path: str, img: np.ndarray, names: Sequence[str], thr: float,
               runs: int, img_sz: int, iou_thr: float,
               no_xnnpack: bool, threads: int) -> Tuple[str, float, float]:
    try:
        from ai_edge_litert.interpreter import Interpreter
    except ImportError:
        from tflite_runtime.interpreter import Interpreter  # type: ignore

    kwargs = {"model_path": path, "num_threads": threads}
    if no_xnnpack:
        kwargs["experimental_delegates"] = []              # disable XNNPACK

    it = Interpreter(**kwargs)
    it.allocate_tensors()

    in_det, out_det = it.get_input_details(), it.get_output_details()
    inp = preprocess(img, in_det, img_sz)

    it.set_tensor(in_det[0]["index"], inp)
    it.invoke()                                            # прогрів

    times, det = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        it.set_tensor(in_det[0]["index"], inp)
        it.invoke()
        times.append(time.perf_counter() - t0)
        if i == runs - 1:
            det = parse_tflite_out(it, out_det, thr, names,
                                   in_w=img_sz, iou_thr=iou_thr)

    return *order_string(det), np.mean(times)


# ─────────────────────────────── CLI ───────────────────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("OCR-YOLO tester (PT / ONNX / TFLite-INT8)")
    p.add_argument("-m", "--model", "--path", dest="model_paths",
                   action="append", required=True,
                   help="Шляхи до моделей (.pt / .onnx / .tflite)")
    p.add_argument("--image", required=True, help="Тестове зображення")
    p.add_argument("--input_size", type=int, default=320, help="Розмір інпуту")
    p.add_argument("--conf", type=float, default=0.12, help="Поріг conf")
    p.add_argument("--iou_thr", type=float, default=0.35, help="IoU-NMS поріг")
    p.add_argument("--runs", type=int, default=5, help="Запусків для avg time")
    p.add_argument("--no-xnnpack", action="store_true",
                   help="Вимкнути XNNPACK delegate (детермінованість)")
    p.add_argument("--threads", type=int, default=2,
                   help="К-сть потоків для TFLite")
    p.add_argument("--clear-cache", action="store_true",
                   help="Очищати кеш/GC після кожної моделі")
    return p.parse_args()


# ─────────────────────────────── main ──────────────────────────────────────
if __name__ == "__main__":
    args = cli()

    # перевірка наявності файлів
    for f in args.model_paths + [args.image]:
        if not os.path.exists(f):
            sys.exit(f"Файл не знайдено: {f}")

    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Не вдалося відкрити {args.image}")

    pt_first = next((p for p in args.model_paths if p.endswith(".pt")), None)
    names = load_names(pt_first)

    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    for mp in args.model_paths:
        ext = os.path.splitext(mp)[1].lower()

        if ext in (".pt", ".onnx"):
            plate, conf, t = run_pt_or_onnx(mp, img, names,
                                            args.conf, args.runs)

        elif ext == ".tflite":
            plate, conf, t = run_tflite(mp, img, names,
                                        args.conf, args.runs,
                                        img_sz=args.input_size,
                                        iou_thr=args.iou_thr,
                                        no_xnnpack=args.no_xnnpack,
                                        threads=args.threads)
        else:
            print(f"[{os.path.basename(mp):>12}]  ❌ Формат не підтримується.")
            continue

        print(f"[{os.path.basename(mp):>12}]  Plate: {plate:<15} "
              f"Avg conf: {conf:.3f}  Time: {t*1000:.1f} ms")

        if args.clear_cache:
            del plate, conf, t                      # звільняємо об’єкти
            gc.collect()

    print("====================================\n")
