#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_test_fix.py  –  детермінований тестер OCR-YOLO
==================================================
 • Підтримка  .pt / .onnx / .tflite-INT8.
 • Дублікати прибираються правилом
       |cx_i − cx_j| ≥ center_factor × max(w_i, w_j)
 • --no-xnnpack  →  delegate повністю відключено
 • Робота з моделями, що приймають 3-D (HWC/CHW) чи 4-D (NHWC/NCHW) вхід.
"""

from __future__ import annotations
import argparse, os, sys, math, time, gc
from typing import List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO


# ────────────────────────── CLI ──────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("OCR-YOLO tester (PT / ONNX / TFLite-INT8)")
    p.add_argument("-m", "--model", action="append", required=True,
                   help="Файли моделей .pt / .onnx / .tflite")
    p.add_argument("--image", required=True, help="Тестове зображення")
    p.add_argument("--input_size", type=int, default=320, help="Розмір підгонки")
    p.add_argument("--conf", type=float, default=0.15, help="Поріг conf")
    p.add_argument("--center-factor", type=float, default=0.6,
                   help="Множник відступу центрів (default 0.6)")
    p.add_argument("--max-chars", type=int, default=8,
                   help="Ліміт символів у рядку (0 = без)")
    p.add_argument("--runs", type=int, default=5, help="Запусків для середнього часу")
    p.add_argument("--no-xnnpack", action="store_true",
                   help="Повністю вимкнути XNNPACK-delegate")
    p.add_argument("--threads", type=int, default=1,
                   help="К-сть потоків TFLite (1 = повна відтворюваність)")
    p.add_argument("--show-all", action="store_true",
                   help="Показувати результат кожного run")
    return p.parse_args()


# ───────────────────── загальні утиліти ───────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _dequant(arr: np.ndarray, det: dict) -> np.ndarray:
    """INT8/UINT8 → float32 (залишає float32 без змін)."""
    if arr.dtype == np.float32:
        return arr
    qp = det["quantization_parameters"]
    return (arr.astype(np.float32) - qp["zero_points"]) * qp["scales"]


def center_filter(raw: List[Tuple[float, float, str, float]],
                  factor: float, max_chars: int) -> List[Tuple[float, str, float]]:
    """Застосовує правило “відступу по центру” та обрізає до max_chars."""
    keep: List[Tuple[float, float, str, float]] = []
    for cand in sorted(raw, key=lambda d: d[3], reverse=True):
        cx, w, ch, cf = cand
        if all(abs(cx - k[0]) >= factor * max(w, k[1]) for k in keep):
            keep.append(cand)
            if max_chars and len(keep) == max_chars:
                break
    keep.sort(key=lambda d: d[0])                # ліворуч→праворуч
    return [(cx, ch, cf) for cx, w, ch, cf in keep]


def det_to_str(det: List[Tuple[float, str, float]]) -> Tuple[str, float]:
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0
    chars, confs = zip(*[(c, cf) for _, c, cf in det])
    return "".join(chars), float(np.mean(confs))


def load_names(pt_path: str | None) -> Sequence[str]:
    if pt_path and os.path.splitext(pt_path)[1].lower() == ".pt":
        try:
            names = YOLO(pt_path).names
            return names if names else CLASS_NAMES
        except Exception:
            pass
    return CLASS_NAMES


def preprocess(img: np.ndarray, inp_det: dict, img_sz: int) -> np.ndarray:
    """
    Готує зображення під будь-який формат вхідного тензора:
    • 3-D (HWC/CHW)  → додає batch-розмір
    • 4-D (NHWC/NCHW)
    • динамічні розміри (None / 0) заповнює img_sz
    """
    shape = inp_det["shape"]
    dtype = inp_det["dtype"]
    nd = len(shape)

    # --- визначаємо H, W та порядок каналів ---
    if nd == 4:                                     # [N,C,H,W] або [N,H,W,C]
        if shape[1] == 3:                           # NCHW
            h = shape[2] or img_sz
            w = shape[3] or img_sz
            nchw = True
        else:                                       # NHWC
            h = shape[1] or img_sz
            w = shape[2] or img_sz
            nchw = False
    elif nd == 3:                                   # [C,H,W] або [H,W,C]
        if shape[2] == 3:                           # HWC
            h, w = shape[0] or img_sz, shape[1] or img_sz
            nchw = False
        else:                                       # CHW
            h, w = shape[1] or img_sz, shape[2] or img_sz
            nchw = True
    else:                                           # fallback
        h = w = img_sz
        nchw = False

    # --- сам препроцесс ---
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR).astype(np.float32) / 255.0
    if nchw:
        img = img.transpose(2, 0, 1)                # HWC → CHW
    if img.ndim == 3:                               # додаємо batch, якщо треба
        img = img[None]

    if dtype != np.float32:
        qp = inp_det["quantization_parameters"]
        scale = qp["scales"][0] or 1.0
        zp = qp["zero_points"][0]
        img = (img / scale + zp).astype(dtype)
    else:
        img = img.astype(dtype)

    return img


# ─────────────────────── post-proc TFLite ───────────────────────
def parse_tflite(interp, out_det, thr: float, names: Sequence[str],
                 img_w: int, c_factor: float, max_chars: int
                 ) -> List[Tuple[float, str, float]]:
    """Витягує (cx,w,char,conf) → центр-фільтр → [(cx,char,conf), …]."""
    raw: List[Tuple[float, float, str, float]] = []
    σ = lambda x: 1 / (1 + math.exp(-x))

    # 4-тензорний вихід
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
            raw.append(((x1 + x2) * 0.5, x2 - x1, names[cid], cf))

    # ≥ 6-атрибутний єдиний тензор
    else:
        t = _dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])
        if t.ndim != 3:
            return []

        attrs = t.shape[2]
        for row in t[0]:
            if attrs == 6:                          # x1 y1 x2 y2 conf cls
                x1, _, x2, _, cf, cid = row
                cf, cid = float(cf), int(cid)
                if cf < thr or cid >= len(names):
                    continue
                raw.append(((x1 + x2) * 0.5, x2 - x1, names[cid], cf))

            else:                                   # ≥ 7 атрибутів (YOLO-vX)
                obj = σ(float(row[4])) if attrs >= 41 else float(row[4])
                if obj < 1e-6:
                    continue
                cls_logits = row[5:5 + len(names)]
                cid = int(np.argmax(cls_logits))
                cf = obj * (σ(float(cls_logits[cid])) if attrs >= 41 else 1.0)
                if cf < thr or cid >= len(names):
                    continue
                cx, w = float(row[0]) * img_w, float(row[2]) * img_w
                raw.append((cx, w, names[cid], cf))

    return center_filter(raw, c_factor, max_chars)


# ───────────────────── back-end -- TFLite -- ────────────────────
def run_tflite(path: str, img: np.ndarray, names: Sequence[str], thr: float,
               runs: int, img_sz: int, no_xnn: bool, threads: int,
               show: bool, cfactor: float, max_chars: int
               ) -> Tuple[str, float, float]:
    # delegate off, якщо треба ─ важливо зробити ДО імпорту!
    if no_xnn:
        os.environ["TFLITE_ENABLE_XNNPACK"] = "0"
    from tflite_runtime.interpreter import Interpreter  # імпорт після env!

    kwargs = {"model_path": path, "num_threads": threads}
    if no_xnn:
        kwargs["experimental_delegates"] = []           # на всяк випадок

    it = Interpreter(**kwargs)
    it.allocate_tensors()

    inp_det = it.get_input_details()[0]
    out_det = it.get_output_details()

    tin = preprocess(img, inp_det, img_sz)

    # прогрів
    it.set_tensor(inp_det["index"], tin)
    it.invoke()

    times, det_last = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        it.set_tensor(inp_det["index"], tin)
        it.invoke()
        times.append(time.perf_counter() - t0)

        det = parse_tflite(it, out_det, thr, names, img_sz,
                           cfactor, max_chars)
        plate, conf = det_to_str(det)
        if show:
            print(f"      run {i+1}/{runs}: {plate:<15} "
                  f"conf {conf:.3f}  {times[-1]*1000:.1f} ms")
        det_last = det

    plate_fin, conf_fin = det_to_str(det_last)
    return plate_fin, conf_fin, float(np.mean(times))


# ───────────────────── back-end -- PT / ONNX ────────────────────
def run_yolo(path: str, img: np.ndarray, names: Sequence[str], thr: float,
             runs: int, show: bool, cfactor: float, max_chars: int
             ) -> Tuple[str, float, float]:
    model = YOLO(path)
    _ = model(img, verbose=False, conf=thr)             # warm-up

    times, det_last = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        res = model(img, verbose=False, conf=thr)[0]
        times.append(time.perf_counter() - t0)

        raw = []
        for b in res.boxes:
            cf, cid = float(b.conf.squeeze()), int(b.cls.squeeze())
            if cf < thr or cid >= len(names):
                continue
            cx, w = float(b.xywh.squeeze()[0]), float(b.xywh.squeeze()[2])
            raw.append((cx, w, names[cid], cf))

        det = center_filter(raw, cfactor, max_chars)
        plate, conf = det_to_str(det)
        if show:
            print(f"      run {i+1}/{runs}: {plate:<15} "
                  f"conf {conf:.3f}  {times[-1]*1000:.1f} ms")
        det_last = det

    plate_fin, conf_fin = det_to_str(det_last)
    return plate_fin, conf_fin, float(np.mean(times))


# ───────────────────────── main ─────────────────────────
if __name__ == "__main__":
    args = cli()

    if not os.path.exists(args.image):
        sys.exit("⛔  Не знайдено файл зображення.")
    img = cv2.imread(args.image)
    if img is None:
        sys.exit("⛔  Неможливо відкрити зображення.")

    # для імен класів достатньо першої .pt-моделі, якщо є
    pt_first = next((p for p in args.model if p.endswith(".pt")), None)
    names = load_names(pt_first)

    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    for mp in args.model:
        if not os.path.exists(mp):
            print(f"[{mp}]  ❌  файл не знайдено"); continue
        print(f"[{os.path.basename(mp):>12}]")

        ext = os.path.splitext(mp)[1].lower()
        if ext == ".tflite":
            plate, conf, t = run_tflite(mp, img, names, args.conf, args.runs,
                                        args.input_size, args.no_xnnpack,
                                        args.threads, args.show_all,
                                        args.center_factor, args.max_chars)
        elif ext in (".pt", ".onnx"):
            plate, conf, t = run_yolo(mp, img, names, args.conf, args.runs,
                                      args.show_all, args.center_factor,
                                      args.max_chars)
        else:
            print("  ❌  Формат не підтримується.\n"); continue

        print(f"  ↪  Plate: {plate:<15}  Avg conf: {conf:.3f}  "
              f"Avg time: {t*1000:.1f} ms\n")
        gc.collect()

    print("====================================\n")
