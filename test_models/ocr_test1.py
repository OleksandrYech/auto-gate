#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-YOLO tester (PT / ONNX / TFLite-LiteRT) - FIXED VERSION

• Автоматично обробляє три типи TFLite-виходу:
  ① 4-тензорний   ② raw 41-атрибут   ③ post-sigmoid 7-атрибут.
• 1-D IoU-NMS вздовж X прибирає «стіну нулів».
• Fixed deterministic behavior and proper model handling

Використання приклад:

    python3 ocr_test.py --model ocr.pt/onnx/tflite --image image.png --input_size 320 --conf 0.12 --iou_thr 0.35 --runs 5

Залежності: opencv-python, numpy, (tflite_runtime | tensorflow), ultralytics,
onnxruntime.  Python ≥ 3.8.
"""

from __future__ import annotations
import argparse, math, os, sys, time, random
from typing import List, Sequence, Tuple
import cv2, numpy as np
from ultralytics import YOLO

# Set seeds for reproducibility
np.random.seed(42)
random.seed(42)


# ─────────── parse_tflite_out ───────────────────────────────────────────────
def parse_tflite_out(interp, od, thr: float, names: Sequence[str],
                     img_w: int, iou_thr: float):
    """Повертає [(x_center_px, char, conf), …] для LiteRT."""

    def deq(a, d):
        if a.dtype == np.float32:
            return a
        qp = d["quantization_parameters"]
        scales = qp.get("scales", [1.0])
        zero_points = qp.get("zero_points", [0])
        if isinstance(scales, (list, tuple)) and len(scales) > 0:
            scale = scales[0]
        else:
            scale = float(scales) if scales else 1.0
        if isinstance(zero_points, (list, tuple)) and len(zero_points) > 0:
            zero_point = zero_points[0]
        else:
            zero_point = int(zero_points) if zero_points else 0
        return (a.astype(np.float32) - zero_point) * scale

    def iou1d(a1, a2):
        l, r = max(a1[0], a2[0]), min(a1[1], a2[1])
        inter = max(0.0, r - l)
        union = (a1[1] - a1[0]) + (a2[1] - a2[0]) - inter
        return inter / (union + 1e-6) if union > 0 else 0.0

    det_raw: List[Tuple[float, float, str, float]] = []

    # 1) 4-тензорний output format
    if len(od) == 4 and od[0]["shape"][-1] == 4:
        try:
            boxes = deq(interp.get_tensor(od[0]["index"]), od[0])[0]
            scores = deq(interp.get_tensor(od[1]["index"]), od[1])[0]
            classes = deq(interp.get_tensor(od[2]["index"]), od[2])[0]
            n = int(interp.get_tensor(od[3]["index"])[0])

            for j in range(min(n, len(scores))):
                cf = float(scores[j])
                cid = int(classes[j])
                if cf < thr or cid >= len(names) or cid < 0:
                    continue
                x1, _, x2, _ = boxes[j]
                if x2 > x1:  # Valid box
                    det_raw.append((float(x1), float(x2), names[cid], cf))
        except Exception as e:
            print(f"Error in 4-tensor parsing: {e}")
            return []

    # 2) Single tensor output
    else:
        try:
            raw = deq(interp.get_tensor(od[0]["index"]), od[0])
            if raw.ndim != 3:
                return []
            attrs = raw.shape[2]

            # 2.a 6-атрибут format
            if attrs == 6:
                for row in raw[0]:
                    x1, _, x2, _, cf, cid = row
                    cf = float(cf)
                    cid = int(cid)
                    if cf < thr or cid >= len(names) or cid < 0:
                        continue
                    if x2 > x1:  # Valid box
                        det_raw.append((float(x1), float(x2), names[cid], cf))

            # 2.b 7- або 41-атрибут format
            elif attrs >= 7:
                def sigmoid(x):
                    x = np.clip(x, -500, 500)  # Prevent overflow
                    return 1.0 / (1.0 + np.exp(-x))

                for row in raw[0]:
                    try:
                        if attrs == 7:  # post-sigmoid
                            cx, cy, w, h, cf, cid, extra = row[:7]
                            cf = float(cf)
                            cid = int(round(cid))
                            if cf < thr or cid >= len(names) or cid < 0:
                                continue
                            cx, w = float(cx) * img_w, float(w) * img_w
                        else:  # raw 41-attribute format
                            cx, cy, w, h, obj_conf = row[:5]
                            obj_conf = sigmoid(float(obj_conf))
                            if obj_conf < 1e-6:
                                continue

                            # Get class logits and apply sigmoid
                            logits = row[5:5 + len(names)]
                            class_probs = [sigmoid(float(logit)) for logit in logits]
                            cid = int(np.argmax(class_probs))
                            cf = obj_conf * class_probs[cid]

                            if cf < thr or cid >= len(names) or cid < 0:
                                continue
                            cx, w = float(cx) * img_w, float(w) * img_w

                        if w < 2:  # Skip very small boxes
                            continue

                        x1, x2 = cx - w / 2, cx + w / 2
                        if x2 > x1:  # Valid box
                            det_raw.append((float(x1), float(x2), names[cid], cf))

                    except (ValueError, IndexError, OverflowError) as e:
                        continue  # Skip malformed detections

        except Exception as e:
            print(f"Error in single tensor parsing: {e}")
            return []

    if not det_raw:
        return []

    # Apply NMS with stable sorting
    det_raw.sort(key=lambda d: (-d[3], d[0]))  # Sort by confidence desc, then x pos
    keep: List[Tuple[float, float, str, float]] = []

    for cand in det_raw:
        should_keep = True
        for k in keep:
            if iou1d((cand[0], cand[1]), (k[0], k[1])) > iou_thr:
                should_keep = False
                break
        if should_keep:
            keep.append(cand)

    # Sort by x position for final output
    keep.sort(key=lambda d: (d[0] + d[1]) / 2)
    return [((k[0] + k[1]) / 2, k[2], k[3]) for k in keep]


# ─────────── решта допоміжних функцій ───────────────────────────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def preprocess(img, inp_det, img_sz):
    """Preprocess image with consistent normalization"""
    try:
        shp, dtype = inp_det[0]["shape"], inp_det[0]["dtype"]
        nchw = len(shp) == 4 and shp[1] == 3

        if nchw:
            h, w = shp[2], shp[3]
        else:
            h, w = shp[1], shp[2]

        # Use provided size if shape is dynamic
        if h <= 0:
            h = img_sz
        if w <= 0:
            w = img_sz

        # Convert and resize with consistent interpolation
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        img_float = img_resized.astype(np.float32) / 255.0

        # Rearrange dimensions if needed
        if nchw:
            img_float = img_float.transpose(2, 0, 1)

        # Add batch dimension
        img_batch = img_float[None]

        # Quantize if needed
        if dtype in (np.int8, np.uint8):
            qp = inp_det[0]["quantization_parameters"]
            scales = qp.get("scales", [1.0])
            zero_points = qp.get("zero_points", [0])

            if isinstance(scales, (list, tuple)) and len(scales) > 0:
                scale = scales[0]
            else:
                scale = float(scales) if scales else 1.0

            if isinstance(zero_points, (list, tuple)) and len(zero_points) > 0:
                zero_point = zero_points[0]
            else:
                zero_point = int(zero_points) if zero_points else 0

            img_batch = (img_batch / scale + zero_point).astype(dtype)

        return img_batch.astype(dtype)

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        # Fallback preprocessing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_sz, img_sz))
        img_float = img_resized.astype(np.float32) / 255.0
        return img_float.transpose(2, 0, 1)[None]


def order_string(det):
    """Order detected characters by x position"""
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0

    # Sort by x position (already done in parse_tflite_out, but ensure it)
    det_sorted = sorted(det, key=lambda d: d[0])
    chars, confs = zip(*[(c, cf) for _, c, cf in det_sorted])
    avg_conf = float(np.mean(confs))

    return "".join(chars), avg_conf


def load_names(pt_path):
    """Load class names from PT model if available"""
    if pt_path:
        try:
            model = YOLO(pt_path)
            names = model.names
            if names and len(names) > 0:
                return [names[i] for i in sorted(names.keys())]
        except Exception as e:
            print(f"Warning: Could not load names from {pt_path}: {e}")
    return CLASS_NAMES


# ─────────── back-ends ──────────────────────────────────────────────────────
def run_pt_or_onnx(path, img, names, thr, runs):
    """Run PyTorch or ONNX model"""
    try:
        model = YOLO(path)
        # Warmup
        _ = model(img, verbose=False, conf=thr)

        times, det = [], []
        for i in range(runs):
            t0 = time.perf_counter()
            results = model(img, verbose=False, conf=thr)[0]
            times.append(time.perf_counter() - t0)

            if i == runs - 1 and results.boxes is not None:
                for box in results.boxes:
                    cf = float(box.conf.squeeze())
                    cid = int(box.cls.squeeze())
                    if cf < thr or cid >= len(names):
                        continue
                    x_center = float(box.xywh.squeeze()[0])
                    det.append((x_center, names[cid], cf))

        return *order_string(det), np.mean(times)
    except Exception as e:
        return f"ERROR: {e}", 0.0, 0.0


def run_litert(path, img, names, thr, runs, img_sz, iou_thr):
    """Run LiteRT/TFLite model with consistent behavior"""
    try:
        # Try different import options for TFLite
        interpreter_class = None
        try:
            from ai_edge_litert.interpreter import Interpreter
            interpreter_class = Interpreter
        except ImportError:
            try:
                import tflite_runtime.interpreter as tflite
                interpreter_class = tflite.Interpreter
            except ImportError:
                import tensorflow as tf
                interpreter_class = tf.lite.Interpreter

        # Initialize interpreter
        interpreter = interpreter_class(model_path=path, num_threads=1)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Preprocess image
        input_data = preprocess(img, input_details, img_sz)

        # Warmup run
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        times, det = [], []
        for i in range(runs):
            t0 = time.perf_counter()
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            times.append(time.perf_counter() - t0)

            # Only parse output on the last run for consistency
            if i == runs - 1:
                det = parse_tflite_out(interpreter, output_details, thr, names,
                                       img_w=img_sz, iou_thr=iou_thr)

        return *order_string(det), np.mean(times)

    except Exception as e:
        return f"ERROR: {e}", 0.0, 0.0


# ─────────── CLI та main ────────────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser("OCR-YOLO tester (LiteRT) - FIXED")
    p.add_argument("-m", "--model", "--path", dest="model_paths",
                   action="append", required=True, help="Model path(s)")
    p.add_argument("--image", required=True, help="Input image path")
    p.add_argument("--input_size", type=int, default=320, help="Input size")
    p.add_argument("--conf", type=float, default=0.12, help="Confidence threshold")
    p.add_argument("--iou_thr", type=float, default=0.35, help="IoU threshold for NMS")
    p.add_argument("--runs", type=int, default=5, help="Number of inference runs")
    return p.parse_args()


if __name__ == "__main__":
    args = cli()

    # Validate files
    for f in args.model_paths + [args.image]:
        if not os.path.exists(f):
            sys.exit(f"Файл не знайдено: {f}")

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Не вдалося відкрити {args.image}")

    # Load class names
    pt_first = next((p for p in args.model_paths if p.endswith(".pt")), None)
    names = load_names(pt_first)

    print("\n===========  РЕЗУЛЬТАТИ  ===========")
    for model_path in args.model_paths:
        ext = os.path.splitext(model_path)[1].lower()
        model_name = os.path.basename(model_path)

        if ext in (".pt", ".onnx"):
            plate, conf, avg_time = run_pt_or_onnx(model_path, img, names, args.conf, args.runs)
        elif ext == ".tflite":
            plate, conf, avg_time = run_litert(model_path, img, names, args.conf, args.runs,
                                               img_sz=args.input_size, iou_thr=args.iou_thr)
        else:
            print(f"[{model_name:>12}]  ❌ Формат не підтримується.")
            continue

        if isinstance(plate, str) and plate.startswith("ERROR"):
            print(f"[{model_name:>12}]  ❌ {plate}")
        else:
            print(f"[{model_name:>12}]  Plate: {plate:<15} "
                  f"Avg conf: {conf:.3f}  Time: {avg_time * 1000:.1f} ms")

    print("====================================\n")