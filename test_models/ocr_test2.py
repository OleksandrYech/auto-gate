#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-YOLO tester  —  PT / ONNX / TFLite-INT8
===================================================
• Дублікати прибираються за «відступом по центру»:
      |cx_i − cx_j| ≥ center_factor × max(w_i, w_j)
• CLI: --center-factor (0.6), --max-chars (8), --show-all,
       --no-xnnpack, --threads, --clear-cache
• Залежності: opencv-python, numpy, ultralytics, onnxruntime,
              tflite-runtime (або tensorflow ≥2.13), Python ≥3.8.
"""

from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from typing import List, Sequence, Tuple

# ───────────────────────────── CLI ──────────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("OCR-YOLO tester (PT / ONNX / TFLite-INT8)")
    p.add_argument("-m", "--model", dest="model_paths", action="append",
                   required=True, help="Шляхи до моделей (.pt / .onnx / .tflite)")
    p.add_argument("--image", required=True, help="Тестове зображення")
    p.add_argument("--input_size", type=int, default=320, help="Розмір інпуту")
    p.add_argument("--conf", type=float, default=0.12, help="Поріг conf")
    p.add_argument("--center-factor", type=float, default=0.6,
                   help="Множник відступу центрів (default 0.6)")
    p.add_argument("--max-chars", type=int, default=8,
                   help="Максимальна довжина рядка (0 = без обмеження)")
    p.add_argument("--runs", type=int, default=5, help="К-сть запусків для avg")
    p.add_argument("--no-xnnpack", action="store_true",
                   help="Повністю вимкнути XNNPACK delegate")
    p.add_argument("--threads", type=int, default=2, help="Потоки для TFLite")
    p.add_argument("--clear-cache", action="store_true",
                   help="Очищати кеш/GC після кожної моделі")
    p.add_argument("--show-all", action="store_true",
                   help="Друкувати результат кожного run")
    return p.parse_args()


# ─────────────────────── загальні утиліти ──────────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def _dequant(arr, det):
    if arr.dtype == "float32":
        return arr
    qp = det["quantization_parameters"]
    return (arr.astype("float32") - qp["zero_points"]) * qp["scales"]

def center_filter(cands: List[Tuple[float, float, str, float]],
                  factor: float, max_chars: int) -> List[Tuple[float, str, float]]:
    """
    Відбирає символи, центри яких рознесені на
    ≥ factor × max(w_i, w_j). Повертає ≤ max_chars детекцій.
    """
    keep: List[Tuple[float, float, str, float]] = []
    for cand in sorted(cands, key=lambda d: d[3], reverse=True):
        cx, w, ch, cf = cand
        if all(abs(cx - k[0]) >= factor * max(w, k[1]) for k in keep):
            keep.append(cand)
        if max_chars and len(keep) == max_chars:
            break
    keep.sort(key=lambda d: d[0])
    return [(cx, ch, cf) for cx, w, ch, cf in keep]

def order_string(det):
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0
    chars, confs = zip(*[(c, cf) for _, c, cf in det])
    return "".join(chars), float(sum(confs) / len(confs))

def load_names(pt_path):
    try:
        from ultralytics import YOLO
        return YOLO(pt_path).names or CLASS_NAMES
    except Exception:
        return CLASS_NAMES

def preprocess(img, in_det, img_sz):
    shp, dtype = in_det[0]["shape"], in_det[0]["dtype"]
    nchw = shp[1] == 3
    h = shp[2] if nchw else shp[1] or img_sz
    w = shp[3] if nchw else shp[2] or img_sz
    import cv2, numpy as np
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR).astype("float32")/255
    if nchw:
        img = img.transpose(2, 0, 1)
    img = img[None]
    if dtype in ("int8", "uint8"):
        qp = in_det[0]["quantization_parameters"]
        img = (img / (qp["scales"][0] or 1.0) + qp["zero_points"][0]).astype(dtype)
    return img.astype(dtype)


# ────────────── TFLite postprocess (YOLO-style) ───────────────
def parse_tflite_out(interp, out_det, thr, names, in_w,
                     center_factor, max_chars):
    σ = lambda x: 1/(1+math.exp(-x))
    raw: List[Tuple[float, float, str, float]] = []

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
            raw.append(((x1+x2)*0.5, x2-x1, names[cid], cf))
    else:  # один тензор ≥6 атрибутів
        t = _dequant(interp.get_tensor(out_det[0]["index"]), out_det[0])
        if t.ndim != 3:
            return []
        attrs = t.shape[2]
        for row in t[0]:
            if attrs == 6:  # x1 y1 x2 y2 conf cls
                x1, _, x2, _, cf, cid = row
                cf, cid = float(cf), int(cid)
                if cf < thr or cid >= len(names):
                    continue
                raw.append(((x1+x2)*0.5, x2-x1, names[cid], cf))
            else:           # ≥7 атрибутів
                obj = σ(float(row[4])) if attrs >= 41 else float(row[4])
                if obj < 1e-6:
                    continue
                cls_logits = row[5:5+len(names)]
                cid = int(max(range(len(cls_logits)), key=cls_logits.__getitem__))
                cf = obj * (σ(float(cls_logits[cid])) if attrs >= 41 else 1.0)
                if cf < thr or cid >= len(names):
                    continue
                cx, w = float(row[0])*in_w, float(row[2])*in_w
                if w < 2:
                    continue
                raw.append((cx, w, names[cid], cf))

    return center_filter(raw, center_factor, max_chars)


# ─────────────────── back-ends (PT / ONNX / TFLite) ──────────────────
def run_pt(path, img, names, thr, runs, show, center_f, max_chars):
    from ultralytics import YOLO
    model = YOLO(path); model(img, verbose=False, conf=thr)
    times=[]; det_final=[]
    for i in range(runs):
        t0=time.perf_counter()
        res=model(img, verbose=False, conf=thr)[0]
        dt=time.perf_counter()-t0; times.append(dt)
        raw=[]
        if res.boxes:
            for b in res.boxes:
                cf, cid=float(b.conf), int(b.cls)
                if cf<thr or cid>=len(names): continue
                cx, w=float(b.xywh[0]), float(b.xywh[2])
                raw.append((cx, w, names[cid], cf))
        det=parse_tflite_out.center_filter(raw, center_f, max_chars) if raw else []
        plate, conf=order_string(det)
        if show: print(f"      run {i+1}/{runs}: {plate:<15} conf {conf:.3f}  {dt*1000:.1f} ms")
        if i==runs-1: det_final=det
    return *order_string(det_final), sum(times)/len(times)


def run_tflite(path, img, names, thr, runs, img_sz, no_xnn, threads,
               show, center_f, max_chars):
    if no_xnn:
        os.environ["TFLITE_ENABLE_XNNPACK"]="0"
    from tflite_runtime.interpreter import Interpreter
    it=Interpreter(model_path=path, num_threads=threads); it.allocate_tensors()
    inp_det, out_det=it.get_input_details(), it.get_output_details()
    inp=preprocess(img, inp_det, img_sz)
    it.set_tensor(inp_det[0]["index"], inp); it.invoke()
    times=[]; det_final=[]
    for i in range(runs):
        t0=time.perf_counter()
        it.set_tensor(inp_det[0]["index"], inp); it.invoke()
        dt=time.perf_counter()-t0; times.append(dt)
        det=parse_tflite_out(it, out_det, thr, names, img_sz, center_f, max_chars)
        plate, conf=order_string(det)
        if show: print(f"      run {i+1}/{runs}: {plate:<15} conf {conf:.3f}  {dt*1000:.1f} ms")
        if i==runs-1: det_final=det
    return *order_string(det_final), sum(times)/len(times)


# ───────────────────────── main ──────────────────────────
if __name__ == "__main__":
    args = cli()
    for f in args.model_paths+[args.image]:
        if not os.path.exists(f):
            sys.exit(f"Файл не знайдено: "+f)

    import cv2, math, numpy as np, time
    img=cv2.imread(args.image); pt_first=next((p for p in args.model_paths if p.endswith(".pt")),None)
    names=load_names(pt_first)

    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    for mp in args.model_paths:
        ext=os.path.splitext(mp)[1].lower(); print(f"[{os.path.basename(mp):>12}]")
        if ext in (".pt",".onnx"):
            plate,conf,t=run_pt(mp,img,names,args.conf,args.runs,args.show_all,
                                args.center_factor,args.max_chars)
        elif ext==".tflite":
            plate,conf,t=run_tflite(mp,img,names,args.conf,args.runs,args.input_size,
                                    args.no_xnnpack,args.threads,args.show_all,
                                    args.center_factor,args.max_chars)
        else:
            print("  ❌ Формат не підтримується."); continue
        print(f"  ↪  Plate: {plate:<15}  Avg conf: {conf:.3f}  Avg time: {t*1000:.1f} ms\n")
        if args.clear_cache: del plate,conf,t; gc.collect()
    print("====================================\n")
