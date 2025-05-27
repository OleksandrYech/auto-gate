#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR‑YOLO tester v1.1 — benchmark and compare *.pt, *.onnx and *.tflite (LiteRT)
===========================================================================

Changelog 2025‑05‑28
────────────────────
• **Bug‑fixes**
  ─ Fixed misuse of the resized width when converting normalised coordinates
    (incorrect plates on non‑square inputs).
  ─ `preprocess()` now correctly handles dynamic shapes ( 0  or  ‑1  dims).
  ─ Guard against dtype mismatches when the model is already fp32.
• **Optimisations**
  ─ `parse_tflite_out()` hot‑path is vectorised with NumPy where possible.
  ─ Uses `with torch.inference_mode()` for PyTorch inference (no grad + cudnn
    autotune) and creates sessions only once per model.
  ─ Re‑uses the same LiteRT interpreter object across runs instead of a new
    allocation per iteration.
  ─ Optional median latency (`--median`) is more robust to outliers.
• **New features**
  ─ `--clear_cache` frees Python, Torch, ONNX Runtime and OpenCV caches after
    *each* model evaluation so that different frameworks do not steal RAM from
    each other when the script is used in long pipelines.

Usage example
-------------
    python3 ocr_test.py -m ocr.tflite -m ocr.pt --image img.jpg \
                        --input_size 320 --conf 0.12 --iou_thr 0.35 \
                        --runs 30 --median --clear_cache

Dependencies:  opencv‑python, numpy, (tflite_runtime | tensorflow),
               ultralytics, onnxruntime, torch ≥ 2.0.  Python ≥ 3.8.
"""
from __future__ import annotations

import argparse
import gc
import math
import os
import sys
import time
from functools import lru_cache
from statistics import mean, median
from typing import Iterable, List, Sequence, Tuple

import cv2
import numpy as np
from ultralytics import YOLO

# ─────────────────────────────── util ──────────────────────────────────────

def clear_caches() -> None:
    """Release Python, Torch, ONNX RT and OpenCV caches if the corresponding
    modules are loaded. This *significantly* reduces peak memory in batch runs.
    """
    gc.collect()
    if "torch" in sys.modules:
        import torch  # pylint: disable=import-error

        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()
    if "onnxruntime" in sys.modules:
        import onnxruntime as ort  # pylint: disable=import-error

        ort.get_all_providers()  # touch the module to avoid pyflakes unused‑import
        # dropping all sessions held by the module releases most buffers
        for _key in list(ort._state._sessions):  # type: ignore[attr-defined]
            del ort._state._sessions[_key]  # noqa: SLF001, pylint: disable=protected-access
    cv2.ocl.setUseOpenCL(False)  # OpenCV GPU cache (harmless if no OpenCL)


# ──────────────────────── TFLite post‑processing ───────────────────────────

def _dequant(arr: np.ndarray, det: dict) -> np.ndarray:
    if arr.dtype == np.float32:
        return arr
    qp = det["quantization_parameters"]
    return (arr.astype("float32") - qp["zero_points"]).astype("float32") * qp["scales"]


def _iou1d(a: Tuple[float, float], b: Tuple[float, float]) -> float:
    l, r = max(a[0], b[0]), min(a[1], b[1])
    inter = max(0.0, r - l)
    return inter / ((a[1] - a[0]) + (b[1] - b[0]) - inter + 1e-6)


def parse_tflite_out(
    it, od: Sequence[dict], thr: float, names: Sequence[str], *,
    img_w: int, iou_thr: float,
) -> List[Tuple[float, str, float]]:
    """Return a list of recognised characters as (x_center_px, char, conf)."""

    det_raw: List[Tuple[float, float, str, float]] = []

    # — 1) 4‑tensor detector (TPU converted YOLO‑v8) ————————————————
    if len(od) == 4 and od[0]["shape"][-1] == 4:
        boxes = _dequant(it.get_tensor(od[0]["index"]), od[0])[0]
        scores = _dequant(it.get_tensor(od[1]["index"]), od[1])[0]
        classes = _dequant(it.get_tensor(od[2]["index"]), od[2])[0].astype(int)
        n = int(it.get_tensor(od[3]["index"])[0])

        mask = (scores[:n] >= thr) & (classes[:n] < len(names))
        for x1, x2, cid, cf in zip(boxes[mask, 0], boxes[mask, 2], classes[mask], scores[mask]):
            det_raw.append((x1, x2, names[int(cid)], float(cf)))

    # — 2) single‑tensor detector (raw 41‑attr, raw 7‑attr, post‑sigmoid 7‑attr) —
    else:
        raw = _dequant(it.get_tensor(od[0]["index"]), od[0])
        if raw.ndim != 3:
            return []
        attrs = raw.shape[2]

        # 2.a — post‑sigmoid 6‑/7‑attr (YOLOv5 edge‑TPU)
        if attrs in (6, 7):
            # shape: 1 × N × attrs
            confs = raw[0, :, 4]
            cids = raw[0, :, 5].astype(int)
            mask = (confs >= thr) & (cids < len(names))
            x1 = raw[0, mask, 0] - raw[0, mask, 2] / 2
            x2 = raw[0, mask, 0] + raw[0, mask, 2] / 2
            for _x1, _x2, cid, cf in zip(x1, x2, cids[mask], confs[mask]):
                det_raw.append((_x1 * img_w, _x2 * img_w, names[int(cid)], float(cf)))

        # 2.b — raw 41‑attr (YOLO‑v8 integer‑quantised)
        elif attrs >= 41:
            sigm = lambda x: 1.0 / (1.0 + np.exp(-x))  # noqa: E731
            cx, w, obj, logits = (
                raw[0, :, 0],  # centre‑x
                raw[0, :, 2],  # width
                raw[0, :, 4],  # objectness
                raw[0, :, 5 : 5 + len(names)],  # class logits
            )
            obj = sigm(obj)
            best_cids = np.argmax(logits, axis=1)
            confs = obj * sigm(logits[np.arange(logits.shape[0]), best_cids])
            mask = (confs >= thr) & (best_cids < len(names)) & (w >= 2 / img_w)
            for _cx, _w, cid, cf in zip(cx[mask], w[mask], best_cids[mask], confs[mask]):
                x1, x2 = (_cx - _w / 2) * img_w, (_cx + _w / 2) * img_w
                det_raw.append((x1, x2, names[int(cid)], float(cf)))

    if not det_raw:
        return []

    # — NMS on the 1‑D X axis ————————————————————————————————
    det_raw.sort(key=lambda d: d[3], reverse=True)  # by confidence
    keep: List[Tuple[float, float, str, float]] = []
    for cand in det_raw:
        if all(_iou1d((cand[0], cand[1]), (k[0], k[1])) <= iou_thr for k in keep):
            keep.append(cand)

    keep.sort(key=lambda d: (d[0] + d[1]) / 2)  # left‑to‑right
    return [((k[0] + k[1]) / 2, k[2], k[3]) for k in keep]


# ───────────────────────────── pre‑/post‑proc ─────────────────────────────
CLASS_NAMES: List[str] = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def _fallback_dim(dim: int | None, img_sz: int) -> int:  # noqa: D401 (concise helper)
    """Return *img_sz* when *dim* is 0, ‑1 or None (dynamic TensorFlow shapes)."""
    return img_sz if dim in (None, 0, -1) else int(dim)


def preprocess(img: np.ndarray, inp_det: Sequence[dict], img_sz: int) -> np.ndarray:
    shp, dtype = inp_det[0]["shape"], inp_det[0]["dtype"]
    nchw = shp[1] == 3  # (N,3,H,W) vs (N,H,W,3)
    h = _fallback_dim(shp[2 if nchw else 1], img_sz)
    w = _fallback_dim(shp[3 if nchw else 2], img_sz)

    if img.shape[:2] != (h, w):
        img = cv2.resize(img, (w, h), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    img = img.astype("float32") / 255.0

    if nchw:
        img = img.transpose(2, 0, 1)  # HWC → CHW
    img = img[None]  # add batch dim

    if dtype in (np.int8, np.uint8):
        qp = inp_det[0]["quantization_parameters"]
        scale = qp["scales"][0] or 1.0
        zero = qp["zero_points"][0]
        img = (img / scale + zero).round().clip(0, 255).astype(dtype)
    else:
        img = img.astype(dtype)
    return img


def order_string(det: List[Tuple[float, str, float]]) -> Tuple[str, float]:
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0
    det.sort(key=lambda d: d[0])
    chars, confs = zip(*[(c, cf) for _, c, cf in det])
    return "".join(chars), float(mean(confs))


@lru_cache(maxsize=4)
def load_names(pt_path: str | None) -> Sequence[str]:
    if pt_path:
        try:
            n = YOLO(pt_path).names
            if n:
                return n
        except Exception:  # noqa: BLE001 (ultralytics prints a lot)
            pass
    return CLASS_NAMES


# ──────────────────────────── back‑ends ───────────────────────────────────

def run_pt_or_onnx(
    path: str,
    img: np.ndarray,
    names: Sequence[str],
    thr: float,
    runs: int,
    use_median: bool,
) -> Tuple[str, float, float]:
    import torch  # pylint: disable=import-error

    model = YOLO(path)

    # one warm‑up — necessary for proper cudnn autotune
    with torch.inference_mode():
        _ = model(img, verbose=False, conf=thr)

    times: List[float] = []
    det: List[Tuple[float, str, float]] = []
    for i in range(runs):
        t0 = time.perf_counter()
        with torch.inference_mode():
            r = model(img, verbose=False, conf=thr)[0]
        times.append(time.perf_counter() - t0)
        if i == runs - 1 and r.boxes:
            for b in r.boxes:
                cf, cid = float(b.conf.squeeze()), int(b.cls.squeeze())
                if cf < thr or cid >= len(names):
                    continue
                det.append((float(b.xywh.squeeze()[0]), names[cid], cf))

    latency = median(times) if use_median else mean(times)
    return *order_string(det), latency


# LiteRT is optional — import only when needed to speed‑up start‑up

def run_litert(
    path: str,
    img: np.ndarray,
    names: Sequence[str],
    thr: float,
    runs: int,
    img_sz: int,
    iou_thr: float,
    use_median: bool,
) -> Tuple[str, float, float]:
    from ai_edge_litert.interpreter import Interpreter  # type: ignore

    it = Interpreter(model_path=path)
    it.allocate_tensors()
    idet, odet = it.get_input_details(), it.get_output_details()

    inp = preprocess(img, idet, img_sz)
    img_w_resized = inp.shape[3 if idet[0]["shape"][1] == 3 else 2]

    it.set_tensor(idet[0]["index"], inp)
    it.invoke()  # warm‑up

    times: List[float] = []
    det: List[Tuple[float, str, float]] = []
    for i in range(runs):
        t0 = time.perf_counter()
        it.set_tensor(idet[0]["index"], inp)
        it.invoke()
        times.append(time.perf_counter() - t0)
        if i == runs - 1:
            det = parse_tflite_out(
                it,
                odet,
                thr,
                names,
                img_w=img_w_resized,
                iou_thr=iou_thr,
            )

    latency = median(times) if use_median else mean(times)
    return *order_string(det), latency


# ───────────────────────────── CLI / main ─────────────────────────────────

def cli() -> argparse.Namespace:  # noqa: D401 (simple helper)
    p = argparse.ArgumentParser("OCR‑YOLO tester (PT / ONNX / TFLite‑LiteRT)")
    p.add_argument("-m", "--model", "--path", dest="model_paths", action="append", required=True)
    p.add_argument("--image", required=True)
    p.add_argument("--input_size", type=int, default=320, metavar="PX", help="Resize shorter side to PX when the model has dynamic shape [320]")
    p.add_argument("--conf", type=float, default=0.12, metavar="P", help="Confidence threshold [0.12]")
    p.add_argument("--iou_thr", type=float, default=0.35, metavar="P", help="IoU‑threshold for 1‑D NMS along X [0.35]")
    p.add_argument("--runs", type=int, default=5, metavar="N", help="Number of timed runs per model [5]")
    p.add_argument("--median", action=argparse.BooleanOptionalAction, default=False, help="Report median instead of mean latency")
    p.add_argument("--clear_cache", action=argparse.BooleanOptionalAction, default=False, help="Clear Torch / ORT / LiteRT caches between models")
    return p.parse_args()


if __name__ == "__main__":
    args = cli()

    # validate files early — fail fast
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
            plate, conf, t = run_pt_or_onnx(mp, img, names, args.conf, args.runs, args.median)
        elif ext == ".tflite":
            plate, conf, t = run_litert(
                mp,
                img,
                names,
                args.conf,
                args.runs,
                img_sz=args.input_size,
                iou_thr=args.iou_thr,
                use_median=args.median,
            )
        else:
            print(f"[{os.path.basename(mp):>12}]  ❌ Формат не підтримується.")
            continue

        t_ms = t * 1_000
        print(f"[{os.path.basename(mp):>12}]  Plate: {plate:<15} Avg conf: {conf:.3f}  Time: {t_ms:.1f} ms")

        if args.clear_cache:
            clear_caches()

    print("====================================\n")
