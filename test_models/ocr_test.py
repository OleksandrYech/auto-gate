#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-YOLO tester (PT / ONNX / TFLite-LiteRT)

• Автоматично обробляє три типи TFLite-виходу:
  ① 4-тензорний   ② raw 41-атрибут   ③ post-sigmoid 7-атрибут.
• 1-D IoU-NMS вздовж X прибирає «стіну нулів».

Використання приклад:

    python3 ocr_test.py --model --image image.png --input_size 320 --conf 0.12 --iou_thr 0.35 --runs 5

Залежності: opencv-python, numpy, (tflite_runtime | tensorflow), ultralytics,
onnxruntime.  Python ≥ 3.8.
"""

from __future__ import annotations
import argparse, math, os, sys, time
from typing import List, Sequence, Tuple
import cv2, numpy as np
from ultralytics import YOLO

# ─────────── parse_tflite_out ───────────────────────────────────────────────
def parse_tflite_out(interp, od, thr: float, names: Sequence[str],
                     img_w: int, iou_thr: float):
    """Повертає [(x_center_px, char, conf), …] для LiteRT."""
    def deq(a, d):
        if a.dtype == np.float32:
            return a
        qp = d["quantization_parameters"]
        return (a.astype(np.float32) - qp["zero_points"]) * qp["scales"]

    def iou1d(a1, a2):
        l, r = max(a1[0], a2[0]), min(a1[1], a2[1])
        inter = max(0.0, r - l)
        return inter / ((a1[1]-a1[0]) + (a2[1]-a2[0]) - inter + 1e-6)

    det_raw: List[Tuple[float,float,str,float]] = []

    # 1) 4-тензорний
    if len(od) == 4 and od[0]["shape"][-1] == 4:
        boxes   = deq(interp.get_tensor(od[0]["index"]), od[0])[0]
        scores  = deq(interp.get_tensor(od[1]["index"]), od[1])[0]
        classes = deq(interp.get_tensor(od[2]["index"]), od[2])[0]
        n = int(interp.get_tensor(od[3]["index"])[0])
        for j in range(n):
            cf = float(scores[j]); cid = int(classes[j])
            if cf < thr or cid >= len(names):         continue
            x1, _, x2, _ = boxes[j]
            det_raw.append((x1, x2, names[cid], cf))

    # 2) один тензор
    else:
        raw = deq(interp.get_tensor(od[0]["index"]), od[0])
        if raw.ndim != 3:
            return []
        attrs = raw.shape[2]

        # 2.a 6-атрибут
        if attrs == 6:
            for x1, _, x2, _, cf, cid in raw[0]:
                cf = float(cf); cid = int(cid)
                if cf < thr or cid >= len(names):     continue
                det_raw.append((x1, x2, names[cid], cf))

        # 2.b 7- або 41-атрибут
        elif attrs >= 7:
            sigm = lambda x: 1/(1+math.exp(-x))
            for row in raw[0]:
                if attrs == 7:  # post-sigmoid
                    cf  = float(row[4])
                    cid = int(row[5])
                    if cf < thr or cid >= len(names): continue
                    cx, w = float(row[0])*img_w, float(row[2])*img_w
                else:          # raw 41-attrib
                    obj = sigm(float(row[4]))
                    if obj < 1e-6:                   continue
                    logits = row[5:5+len(names)]
                    cid = int(np.argmax(logits))
                    cf = obj * sigm(float(logits[cid]))
                    if cf < thr or cid >= len(names): continue
                    cx, w = float(row[0])*img_w, float(row[2])*img_w
                if w < 2:                           continue
                x1, x2 = cx - w/2, cx + w/2
                det_raw.append((x1, x2, names[cid], cf))

    if not det_raw:
        return []

    # NMS
    det_raw.sort(key=lambda d: d[3], reverse=True)
    keep: List[Tuple[float,float,str,float]] = []
    for cand in det_raw:
        if all(iou1d((cand[0],cand[1]), (k[0],k[1])) <= iou_thr for k in keep):
            keep.append(cand)

    keep.sort(key=lambda d: (d[0]+d[1])/2)
    return [((k[0]+k[1])/2, k[2], k[3]) for k in keep]

# ─────────── решта допоміжних функцій ───────────────────────────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")

def preprocess(img, inp_det, img_sz):
    shp, dtype = inp_det[0]["shape"], inp_det[0]["dtype"]
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
        qp = inp_det[0]["quantization_parameters"]
        img = (img / (qp["scales"][0] or 1.0) + qp["zero_points"][0]).astype(dtype)
    return img.astype(dtype)

def order_string(det):
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0
    det.sort(key=lambda d: d[0])
    chars, confs = zip(*[(c, cf) for _, c, cf in det])
    return "".join(chars), float(np.mean(confs))

def load_names(pt_path):
    if pt_path:
        try:
            n = YOLO(pt_path).names
            if n:
                return n
        except Exception:
            pass
    return CLASS_NAMES

# ─────────── back-ends ──────────────────────────────────────────────────────
def run_pt_or_onnx(path, img, names, thr, runs):
    m = YOLO(path); _ = m(img, verbose=False, conf=thr)
    times, det = [], []
    for i in range(runs):
        t0 = time.perf_counter()
        r = m(img, verbose=False, conf=thr)[0]
        times.append(time.perf_counter()-t0)
        if i == runs-1 and r.boxes:
            for b in r.boxes:
                cf=float(b.conf.squeeze()); cid=int(b.cls.squeeze())
                if cf<thr or cid>=len(names): continue
                det.append((float(b.xywh.squeeze()[0]), names[cid], cf))
    return *order_string(det), np.mean(times)

def run_litert(path, img, names, thr, runs, img_sz, iou_thr):
    from ai_edge_litert.interpreter import Interpreter
    it = Interpreter(model_path=path); it.allocate_tensors()
    idet, odet = it.get_input_details(), it.get_output_details()
    inp = preprocess(img, idet, img_sz)
    it.set_tensor(idet[0]["index"], inp); it.invoke()       # прогрів
    times, det = [], []
    for i in range(runs):
        t0=time.perf_counter()
        it.set_tensor(idet[0]["index"], inp); it.invoke()
        times.append(time.perf_counter()-t0)
        if i==runs-1:
            det=parse_tflite_out(it, odet, thr, names, img_w=img_sz,
                                 iou_thr=iou_thr)
    return *order_string(det), np.mean(times)

# ─────────── CLI та main ────────────────────────────────────────────────────
def cli():
    p=argparse.ArgumentParser("OCR-YOLO tester (LiteRT)")
    p.add_argument("-m","--model","--path",dest="model_paths",
                   action="append",required=True)
    p.add_argument("--image",required=True)
    p.add_argument("--input_size",type=int,default=320)
    p.add_argument("--conf",type=float,default=0.12)
    p.add_argument("--iou_thr",type=float,default=0.35)
    p.add_argument("--runs",type=int,default=5)
    return p.parse_args()

if __name__=="__main__":
    args=cli()
    for f in args.model_paths+[args.image]:
        if not os.path.exists(f): sys.exit(f"Файл не знайдено: {f}")
    img=cv2.imread(args.image)
    if img is None: sys.exit(f"Не вдалося відкрити {args.image}")
    pt_first=next((p for p in args.model_paths if p.endswith(".pt")),None)
    names=load_names(pt_first)

    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    for mp in args.model_paths:
        ext=os.path.splitext(mp)[1].lower()
        if ext in(".pt",".onnx"):
            plate,conf,t=run_pt_or_onnx(mp,img,names,args.conf,args.runs)
        elif ext==".tflite":
            plate,conf,t=run_litert(mp,img,names,args.conf,args.runs,
                                    img_sz=args.input_size,
                                    iou_thr=args.iou_thr)
        else:
            print(f"[{os.path.basename(mp):>12}]  ❌ Формат не підтримується.")
            continue
        print(f"[{os.path.basename(mp):>12}]  Plate: {plate:<15} "
              f"Avg conf: {conf:.3f}  Time: {t*1000:.1f} ms")
    print("====================================\n")
