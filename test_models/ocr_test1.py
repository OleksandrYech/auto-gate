#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-YOLO tester (PT / ONNX / TFLite-LiteRT) - FIXED VERSION

• Автоматично обробляє три типи TFLite-виходу:
  ① 4-тензорний   ② raw 41-атрибут   ③ post-sigmoid 7-атрибут.
• 1-D IoU-NMS вздовж X прибирає «стіну нулів».
• Виправлено проблеми з непостійністю результатів TFLite.

Використання приклад:

    python3 ocr_test.py --model ocr.pt/onnx/tflite --image image.png --input_size 320 --conf 0.12 --iou_thr 0.35 --runs 5

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
        scales = qp.get("scales", [1.0])
        zero_points = qp.get("zero_points", [0])

        # Handle scalar vs array scales/zero_points
        if isinstance(scales, (list, np.ndarray)) and len(scales) > 0:
            scale = scales[0]
        else:
            scale = float(scales) if scales else 1.0

        if isinstance(zero_points, (list, np.ndarray)) and len(zero_points) > 0:
            zero_point = zero_points[0]
        else:
            zero_point = int(zero_points) if zero_points else 0

        return (a.astype(np.float32) - zero_point) * scale

    def iou1d(a1, a2):
        l, r = max(a1[0], a2[0]), min(a1[1], a2[1])
        inter = max(0.0, r - l)
        union = (a1[1] - a1[0]) + (a2[1] - a2[0]) - inter
        return inter / (union + 1e-6)

    det_raw: List[Tuple[float, float, str, float]] = []

    try:
        # 1) 4-тензорний
        if len(od) == 4 and od[0]["shape"][-1] == 4:
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
                if x1 >= x2:  # Invalid box
                    continue
                det_raw.append((float(x1), float(x2), names[cid], cf))

        # 2) один тензор
        else:
            raw = deq(interp.get_tensor(od[0]["index"]), od[0])
            if raw.ndim != 3 or raw.shape[0] == 0:
                return []

            attrs = raw.shape[2]
            batch_data = raw[0]  # Take first batch

            # 2.a 6-атрибут
            if attrs == 6:
                for row in batch_data:
                    x1, _, x2, _, cf, cid = row
                    cf = float(cf)
                    cid = int(cid)
                    if cf < thr or cid >= len(names) or cid < 0 or x1 >= x2:
                        continue
                    det_raw.append((float(x1), float(x2), names[cid], cf))

            # 2.b 7- або 41-атрибут
            elif attrs >= 7:
                def sigmoid_safe(x):
                    x = float(x)
                    if x > 700:  # Prevent overflow
                        return 1.0
                    elif x < -700:
                        return 0.0
                    return 1.0 / (1.0 + math.exp(-x))

                for row in batch_data:
                    try:
                        if attrs == 7:  # post-sigmoid
                            cf = float(row[4])
                            cid = int(row[5])
                            if cf < thr or cid >= len(names) or cid < 0:
                                continue
                            cx, w = float(row[0]) * img_w, float(row[2]) * img_w
                        else:  # raw 41-attribute
                            obj = sigmoid_safe(row[4])
                            if obj < 1e-6:
                                continue

                            # Safely extract class logits
                            logits = row[5:5 + len(names)]
                            if len(logits) == 0:
                                continue

                            # Find class with highest logit
                            cid = int(np.argmax(logits))
                            if cid >= len(names) or cid < 0:
                                continue

                            class_prob = sigmoid_safe(logits[cid])
                            cf = obj * class_prob

                            if cf < thr:
                                continue

                            cx, w = float(row[0]) * img_w, float(row[2]) * img_w

                        # Validate box dimensions
                        if w < 2 or not (0 <= cx <= img_w):
                            continue

                        x1, x2 = cx - w / 2, cx + w / 2
                        if x1 >= x2:
                            continue

                        det_raw.append((float(x1), float(x2), names[cid], cf))

                    except (IndexError, ValueError, OverflowError) as e:
                        # Skip problematic detections
                        continue

    except Exception as e:
        print(f"Warning: Error parsing TFLite output: {e}")
        return []

    if not det_raw:
        return []

    # NMS with improved stability
    det_raw.sort(key=lambda d: d[3], reverse=True)
    keep: List[Tuple[float, float, str, float]] = []

    for cand in det_raw:
        # Check IoU with all kept detections
        suppress = False
        for kept in keep:
            if iou1d((cand[0], cand[1]), (kept[0], kept[1])) > iou_thr:
                suppress = True
                break

        if not suppress:
            keep.append(cand)

    # Sort by x-coordinate for final output
    keep.sort(key=lambda d: (d[0] + d[1]) / 2)
    return [((k[0] + k[1]) / 2, k[2], k[3]) for k in keep]


# ─────────── решта допоміжних функцій ───────────────────────────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def preprocess(img, inp_det, img_sz):
    """Improved preprocessing with better error handling."""
    try:
        shp, dtype = inp_det[0]["shape"], inp_det[0]["dtype"]
        nchw = len(shp) == 4 and shp[1] == 3

        if nchw:  # NCHW format
            h, w = shp[2], shp[3]
        else:  # NHWC format
            h, w = shp[1], shp[2]

        # Use provided img_sz if model dimensions are dynamic
        if h <= 0:
            h = img_sz
        if w <= 0:
            w = img_sz

        # Convert and resize image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (w, h), interpolation=cv2.INTER_LINEAR)
        img_norm = img_resized.astype(np.float32) / 255.0

        # Transpose if needed
        if nchw:
            img_norm = img_norm.transpose(2, 0, 1)

        # Add batch dimension
        img_batch = img_norm[None]

        # Apply quantization if needed
        if dtype in (np.int8, np.uint8):
            qp = inp_det[0]["quantization_parameters"]
            scales = qp.get("scales", [1.0])
            zero_points = qp.get("zero_points", [0])

            scale = scales[0] if isinstance(scales, (list, np.ndarray)) and len(scales) > 0 else (
                float(scales) if scales else 1.0)
            zero_point = zero_points[0] if isinstance(zero_points, (list, np.ndarray)) and len(zero_points) > 0 else (
                int(zero_points) if zero_points else 0)

            if scale > 0:
                img_batch = (img_batch / scale + zero_point).astype(dtype)
            else:
                img_batch = img_batch.astype(dtype)

        return img_batch.astype(dtype)

    except Exception as e:
        print(f"Error in preprocessing: {e}")
        # Fallback to simple preprocessing
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (img_sz, img_sz))
        img_norm = img_resized.astype(np.float32) / 255.0
        return img_norm[None].astype(np.float32)


def order_string(det):
    """Convert detections to ordered string with confidence."""
    if not det:
        return "НЕ РОЗПІЗНАНО", 0.0

    # Sort by x-coordinate
    det_sorted = sorted(det, key=lambda d: d[0])
    chars = [d[1] for d in det_sorted]
    confs = [d[2] for d in det_sorted]

    return "".join(chars), float(np.mean(confs))


def load_names(pt_path):
    """Load class names from PT model or use defaults."""
    if pt_path:
        try:
            model = YOLO(pt_path)
            if hasattr(model, 'names') and model.names:
                return [model.names[i] for i in sorted(model.names.keys())]
        except Exception as e:
            print(f"Warning: Could not load names from {pt_path}: {e}")
    return CLASS_NAMES


# ─────────── back-ends ──────────────────────────────────────────────────────
def run_pt_or_onnx(path, img, names, thr, runs):
    """Run PyTorch or ONNX model."""
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
                    if cf >= thr and 0 <= cid < len(names):
                        x_center = float(box.xywh.squeeze()[0])
                        det.append((x_center, names[cid], cf))

        return *order_string(det), np.mean(times)

    except Exception as e:
        print(f"Error running {path}: {e}")
        return "ERROR", 0.0, 0.0


def run_litert(path, img, names, thr, runs, img_sz, iou_thr):
    """Run TensorFlow Lite model with improved stability."""
    try:
        # Try different TFLite runtime imports
        try:
            from ai_edge_litert.interpreter import Interpreter
        except ImportError:
            try:
                from tflite_runtime.interpreter import Interpreter
            except ImportError:
                import tensorflow as tf
                Interpreter = tf.lite.Interpreter

        # Initialize interpreter
        interpreter = Interpreter(model_path=path)
        interpreter.allocate_tensors()

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Preprocess image
        input_data = preprocess(img, input_details, img_sz)

        # Warmup run
        interpreter.set_tensor(input_details[0]["index"], input_data)
        interpreter.invoke()

        times, all_results = [], []

        for i in range(runs):
            # Reset interpreter state for consistency
            interpreter.reset_all_variables() if hasattr(interpreter, 'reset_all_variables') else None

            t0 = time.perf_counter()
            interpreter.set_tensor(input_details[0]["index"], input_data)
            interpreter.invoke()
            times.append(time.perf_counter() - t0)

            # Parse output
            det = parse_tflite_out(interpreter, output_details, thr, names,
                                   img_w=img_sz, iou_thr=iou_thr)
            all_results.append(det)

        # Use the most common result or the last one
        if all_results:
            # Convert results to strings for comparison
            result_strings = [order_string(det)[0] for det in all_results]
            # Use the most recent result (last run)
            final_det = all_results[-1]

            # If we get multiple different results, print warning
            unique_results = set(result_strings)
            if len(unique_results) > 1:
                print(f"Warning: Inconsistent results detected: {unique_results}")
                print(f"Using result from last run: {result_strings[-1]}")
        else:
            final_det = []

        return *order_string(final_det), np.mean(times)

    except Exception as e:
        print(f"Error running TFLite model {path}: {e}")
        return "ERROR", 0.0, 0.0


# ─────────── CLI та main ────────────────────────────────────────────────────
def cli():
    p = argparse.ArgumentParser(description="OCR-YOLO tester (LiteRT) - Fixed Version")
    p.add_argument("-m", "--model", "--path", dest="model_paths",
                   action="append", required=True,
                   help="Path to model file(s) (.pt, .onnx, .tflite)")
    p.add_argument("--image", required=True,
                   help="Path to input image")
    p.add_argument("--input_size", type=int, default=320,
                   help="Input image size (default: 320)")
    p.add_argument("--conf", type=float, default=0.12,
                   help="Confidence threshold (default: 0.12)")
    p.add_argument("--iou_thr", type=float, default=0.35,
                   help="IoU threshold for NMS (default: 0.35)")
    p.add_argument("--runs", type=int, default=5,
                   help="Number of inference runs (default: 5)")
    p.add_argument("--verbose", action="store_true",
                   help="Enable verbose output")
    return p.parse_args()


def main():
    args = cli()

    # Check if files exist
    for f in args.model_paths + [args.image]:
        if not os.path.exists(f):
            sys.exit(f"Файл не знайдено: {f}")

    # Load image
    img = cv2.imread(args.image)
    if img is None:
        sys.exit(f"Не вдалося відкрити {args.image}")

    if args.verbose:
        print(f"Image loaded: {img.shape}")

    # Load class names from first PT model if available
    pt_first = next((p for p in args.model_paths if p.endswith(".pt")), None)
    names = load_names(pt_first)

    if args.verbose:
        print(f"Using {len(names)} classes: {names}")

    print("\n===========  РЕЗУЛЬТАТИ  ===========")

    for model_path in args.model_paths:
        ext = os.path.splitext(model_path)[1].lower()
        model_name = os.path.basename(model_path)

        try:
            if ext in (".pt", ".onnx"):
                plate, conf, avg_time = run_pt_or_onnx(
                    model_path, img, names, args.conf, args.runs
                )
            elif ext == ".tflite":
                plate, conf, avg_time = run_litert(
                    model_path, img, names, args.conf, args.runs,
                    img_sz=args.input_size, iou_thr=args.iou_thr
                )
            else:
                print(f"[{model_name:>12}]  ❌ Формат не підтримується.")
                continue

            print(f"[{model_name:>12}]  Plate: {plate:<15} "
                  f"Avg conf: {conf:.3f}  Time: {avg_time * 1000:.1f} ms")

        except Exception as e:
            print(f"[{model_name:>12}]  ❌ Помилка: {e}")
            if args.verbose:
                import traceback
                traceback.print_exc()

    print("====================================\n")


if __name__ == "__main__":
    main()