#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
ocr_test3.py  –  детермінований тестер PT / ONNX / TFLite-INT8
==============================================================
• --no-xnnpack   💡 тепер справді вимикає XNNPACK delegate
• --threads 1    рекомендовано для повної відтворюваності
• Центр-фільтр   |cx_i − cx_j| ≥ factor × max(w_i, w_j)
"""

from __future__ import annotations
import argparse, os, sys, math, time, gc
from typing import List, Sequence, Tuple
import cv2, numpy as np
from ultralytics import YOLO

# ─────────────────────────── CLI ────────────────────────────
def cli() -> argparse.Namespace:
    p = argparse.ArgumentParser("OCR-YOLO tester (PT / ONNX / TFLite)")
    p.add_argument("-m", "--model", dest="model", required=True,
                   help=".pt / .onnx / .tflite")
    p.add_argument("--image", required=True)
    p.add_argument("--input_size", type=int, default=320)
    p.add_argument("--conf", type=float, default=0.12)
    p.add_argument("--center-factor", type=float, default=0.6)
    p.add_argument("--max-chars", type=int, default=8)
    p.add_argument("--runs", type=int, default=5)
    p.add_argument("--no-xnnpack", action="store_true")
    p.add_argument("--threads", type=int, default=1)
    p.add_argument("--show-all", action="store_true")
    return p.parse_args()

# ───────────────────── help-функції ────────────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")
def _deq(a, d):  # dequant
    if a.dtype == np.float32: return a
    q = d["quantization_parameters"]; return (a.astype(np.float32)-q["zero_points"])*q["scales"]

def center_filter(c: List[Tuple[float,float,str,float]], k:float, n:int):
    keep: List[Tuple[float,float,str,float]]=[]
    for cx,w,ch,cf in sorted(c,key=lambda x:x[3],reverse=True):
        if all(abs(cx-x[0])>=k*max(w,x[1]) for x in keep):
            keep.append((cx,w,ch,cf))
        if n and len(keep)==n: break
    keep.sort(key=lambda x:x[0]);              # left→right
    return [(cx,ch,cf) for cx,_,ch,cf in keep]

def to_str(det):                               # (cx,char,conf)->string
    if not det: return "???",0.0
    s,cf=zip(*[(c,p) for _,c,p in det]); return "".join(s),float(np.mean(cf))

def names_from_pt(pt:str|None):
    try: n=YOLO(pt).names; return n if n else CLASS_NAMES
    except Exception: return CLASS_NAMES

def prep(img, in_det, sz):
    shp,dtype=in_det[0]["shape"],in_det[0]["dtype"]; nchw=shp[1]==3
    h=shp[2] if nchw else shp[1] or sz; w=shp[3] if nchw else shp[2] or sz
    img=cv2.cvtColor(img,cv2.COLOR_BGR2RGB); img=cv2.resize(img,(w,h)).astype(np.float32)/255
    if nchw: img=img.transpose(2,0,1); img=img[None]
    if dtype in (np.int8,np.uint8):
        q=in_det[0]["quantization_parameters"]; img=(img/(q["scales"][0]or 1)+q["zero_points"][0]).astype(dtype)
    return img.astype(dtype)

# ─────────────── back-end: TFLite INT8 ─────────────────────
def run_tflite(path,img,names,thr,runs,sz,no_xnn,threads,show,k,nch):
    if no_xnn:
        os.environ["TFLITE_ENABLE_XNNPACK"]="0"
    from tflite_runtime.interpreter import Interpreter
    kw={"model_path":path,"num_threads":threads}
    if no_xnn: kw["experimental_delegates"]=[]
    it=Interpreter(**kw); it.allocate_tensors()
    ind,outd=it.get_input_details(),it.get_output_details()
    tin=prep(img,ind,sz); it.set_tensor(ind[0]["index"],tin); it.invoke()  # warm-up
    t=[]; det_last=[]
    σ=lambda x:1/(1+math.exp(-x))
    for i in range(runs):
        t0=time.perf_counter(); it.set_tensor(ind[0]["index"],tin); it.invoke()
        t.append(time.perf_counter()-t0)
        raw=[]
        if len(outd)==4 and outd[0]["shape"][-1]==4:
            boxes=_deq(it.get_tensor(outd[0]["index"]),outd[0])[0]
            scrs=_deq(it.get_tensor(outd[1]["index"]),outd[1])[0]
            cls=_deq(it.get_tensor(outd[2]["index"]),outd[2])[0]
            n=int(it.get_tensor(outd[3]["index"])[0])
            for j in range(n):
                cf,cid=float(scrs[j]),int(cls[j])
                if cf<thr or cid>=len(names): continue
                x1,x2=boxes[j][0],boxes[j][2]; raw.append(((x1+x2)/2,x2-x1,names[cid],cf))
        else:
            r=_deq(it.get_tensor(outd[0]["index"]),outd[0]); attrs=r.shape[2]
            for row in r[0]:
                if attrs==6:
                    x1,x2,cf,cid=row[0],row[2],float(row[4]),int(row[5])
                    if cf<thr or cid>=len(names): continue
                    raw.append(((x1+x2)/2,x2-x1,names[cid],cf))
                else:
                    obj=σ(float(row[4])) if attrs>=41 else float(row[4]);
                    if obj<1e-6: continue
                    cls_logits=row[5:5+len(names)]; cid=int(np.argmax(cls_logits))
                    cf=obj*(σ(float(cls_logits[cid])) if attrs>=41 else 1.0)
                    if cf<thr or cid>=len(names): continue
                    cx,w=float(row[0])*sz,float(row[2])*sz
                    raw.append((cx,w,names[cid],cf))
        det=center_filter(raw,k,nch); s,c=to_str(det)
        if show: print(f"      run {i+1}/{runs}: {s:<15} conf {c:.3f}  {t[-1]*1000:.1f} ms")
        det_last=det
    return *to_str(det_last),float(np.mean(t))

# ─────────────── back-end: PT / ONNX ───────────────────────
def run_pt(path,img,names,thr,runs,show,k,nch):
    model=YOLO(path); model(img,verbose=False,conf=thr)
    t=[]; det_last=[]
    for i in range(runs):
        t0=time.perf_counter(); out=model(img,verbose=False,conf=thr)[0]; t.append(time.perf_counter()-t0)
        raw=[]
        for b in out.boxes:
            cf,cid=float(b.conf.squeeze()),int(b.cls.squeeze())
            if cf<thr or cid>=len(names): continue
            cx,w=float(b.xywh.squeeze()[0]),float(b.xywh.squeeze()[2])
            raw.append((cx,w,names[cid],cf))
        det=center_filter(raw,k,nch); s,c=to_str(det)
        if show: print(f"      run {i+1}/{runs}: {s:<15} conf {c:.3f}  {t[-1]*1000:.1f} ms")
        det_last=det
    return *to_str(det_last),float(np.mean(t))

# ───────────────────────── main ────────────────────────────
if __name__=="__main__":
    a=cli()
    if not os.path.exists(a.model) or not os.path.exists(a.image):
        sys.exit("Невірний шлях до файла.")
    img=cv2.imread(a.image); names=names_from_pt(a.model if a.model.endswith(".pt") else None)
    ext=os.path.splitext(a.model)[1].lower()
    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    print(f"[{os.path.basename(a.model):>12}]")
    if ext==".tflite":
        plate,conf,t=run_tflite(a.model,img,names,a.conf,a.runs,a.input_size,
                                a.no_xnnpack,a.threads,a.show_all,
                                a.center_factor,a.max_chars)
    else:
        plate,conf,t=run_pt(a.model,img,names,a.conf,a.runs,a.show_all,
                            a.center_factor,a.max_chars)
    print(f"  ↪  Plate: {plate:<15}  Avg conf: {conf:.3f}  Avg time: {t*1000:.1f} ms")
    print("====================================\n")
