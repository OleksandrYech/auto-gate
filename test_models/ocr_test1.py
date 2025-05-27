#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
OCR-YOLO tester (PT / ONNX / TFLite-LiteRT) - Optimized Version

• Автоматично обробляє три типи TFLite-виходу:
  ① 4-тензорний   ② raw 41-атрибут   ③ post-sigmoid 7-атрибут.
• 1-D IoU-NMS вздовж X прибирає «стіну нулів».
• Додано кешування, обробку помилок, та оптимізації.

Використання приклад:

    python3 ocr_test.py --model ocr.pt/onnx/tflite --image image.png --input_size 320 --conf 0.12 --iou_thr 0.35 --runs 5

Залежності: opencv-python, numpy, (tflite_runtime | tensorflow), ultralytics,
onnxruntime.  Python ≥ 3.8.
"""

from __future__ import annotations
import argparse
import gc
import logging
import math
import os
import sys
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple, Union

import cv2
import numpy as np

# Suppress unnecessary warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Setup logging
logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')
logger = logging.getLogger(__name__)

# Cache for loaded models
MODEL_CACHE: Dict[str, object] = {}


# ─────────── Enhanced parse_tflite_out ─────────────────────────────────────
def parse_tflite_out(interp, od: List[Dict], thr: float, names: Sequence[str],
                     img_w: int, iou_thr: float) -> List[Tuple[float, str, float]]:
    """Повертає [(x_center_px, char, conf), …] для LiteRT з покращеною обробкою помилок."""

    def dequantize(tensor_array: np.ndarray, tensor_details: Dict) -> np.ndarray:
        """Дквантування тензора з перевіркою типів."""
        if tensor_array.dtype == np.float32:
            return tensor_array

        qp = tensor_details.get("quantization_parameters", {})
        scales = qp.get("scales", [1.0])
        zero_points = qp.get("zero_points", [0])

        # Handle scalar and array scales/zero_points
        scale = scales[0] if isinstance(scales, (list, np.ndarray)) else scales
        zero_point = zero_points[0] if isinstance(zero_points, (list, np.ndarray)) else zero_points

        return (tensor_array.astype(np.float32) - zero_point) * scale

    def iou1d(box1: Tuple[float, float], box2: Tuple[float, float]) -> float:
        """1D IoU обчислення з захистом від ділення на нуль."""
        left = max(box1[0], box2[0])
        right = min(box1[1], box2[1])
        intersection = max(0.0, right - left)
        union = (box1[1] - box1[0]) + (box2[1] - box2[0]) - intersection
        return intersection / (union + 1e-8)  # Increased epsilon for stability

    def safe_sigmoid(x: float) -> float:
        """Безпечна sigmoid функція."""
        try:
            x = max(-500, min(500, x))  # Prevent overflow
            return 1.0 / (1.0 + math.exp(-x))
        except (OverflowError, ValueError):
            return 0.5  # Return neutral value on error

    det_raw: List[Tuple[float, float, str, float]] = []

    try:
        # Validate inputs
        if not od or not names:
            return []

        # 1) 4-тензорний формат (SSD/YOLO detection output)
        if len(od) == 4 and od[0]["shape"][-1] == 4:
            try:
                boxes = dequantize(interp.get_tensor(od[0]["index"]), od[0])[0]
                scores = dequantize(interp.get_tensor(od[1]["index"]), od[1])[0]
                classes = dequantize(interp.get_tensor(od[2]["index"]), od[2])[0]
                num_detections = int(interp.get_tensor(od[3]["index"])[0])

                # Validate array shapes
                min_len = min(len(boxes), len(scores), len(classes))
                num_detections = min(num_detections, min_len)

                for j in range(num_detections):
                    try:
                        conf = float(scores[j])
                        class_id = int(classes[j])

                        if conf < thr or class_id < 0 or class_id >= len(names):
                            continue

                        x1, _, x2, _ = boxes[j][:4]  # Take only first 4 elements
                        if x2 > x1:  # Valid box
                            det_raw.append((float(x1), float(x2), names[class_id], conf))
                    except (IndexError, ValueError, TypeError) as e:
                        logger.warning(f"Error processing detection {j}: {e}")
                        continue

            except Exception as e:
                logger.error(f"Error in 4-tensor format processing: {e}")
                return []

        # 2) Один тензор (YOLO raw output)
        else:
            try:
                raw = dequantize(interp.get_tensor(od[0]["index"]), od[0])

                if raw.ndim != 3:
                    logger.warning(f"Unexpected tensor dimensions: {raw.ndim}")
                    return []

                attrs = raw.shape[2]

                # 2.a) 6-атрибут format
                if attrs == 6:
                    for row in raw[0]:
                        try:
                            x1, _, x2, _, conf, class_id = row[:6]
                            conf = float(conf)
                            class_id = int(class_id)

                            if conf < thr or class_id < 0 or class_id >= len(names):
                                continue
                            if x2 > x1:  # Valid box
                                det_raw.append((float(x1), float(x2), names[class_id], conf))
                        except (ValueError, TypeError, IndexError) as e:
                            continue

                # 2.b) 7- або 41-атрибут format
                elif attrs >= 7:
                    for row in raw[0]:
                        try:
                            if attrs == 7:  # post-sigmoid format
                                conf = float(row[4])
                                class_id = int(row[5])

                                if conf < thr or class_id < 0 or class_id >= len(names):
                                    continue

                                cx = float(row[0]) * img_w
                                w = float(row[2]) * img_w
                            else:  # raw 41-attribute format
                                obj_score = safe_sigmoid(float(row[4]))
                                if obj_score < 1e-6:
                                    continue

                                # Extract class logits safely
                                num_classes = min(len(names), attrs - 5)
                                if num_classes <= 0:
                                    continue

                                logits = row[5:5 + num_classes]
                                class_id = int(np.argmax(logits))
                                class_conf = safe_sigmoid(float(logits[class_id]))
                                conf = obj_score * class_conf

                                if conf < thr or class_id < 0 or class_id >= len(names):
                                    continue

                                cx = float(row[0]) * img_w
                                w = float(row[2]) * img_w

                            # Validate box dimensions
                            if w < 2:  # Too small
                                continue

                            x1 = cx - w / 2
                            x2 = cx + w / 2

                            # Ensure valid coordinates
                            if x2 > x1:
                                det_raw.append((x1, x2, names[class_id], conf))

                        except (ValueError, TypeError, IndexError) as e:
                            continue

            except Exception as e:
                logger.error(f"Error in single tensor processing: {e}")
                return []

        if not det_raw:
            return []

        # Enhanced NMS with validation
        det_raw.sort(key=lambda d: d[3], reverse=True)
        keep: List[Tuple[float, float, str, float]] = []

        for candidate in det_raw:
            try:
                # Check if candidate overlaps significantly with any kept detection
                should_keep = True
                for kept_det in keep:
                    if iou1d((candidate[0], candidate[1]), (kept_det[0], kept_det[1])) > iou_thr:
                        should_keep = False
                        break

                if should_keep:
                    keep.append(candidate)
            except Exception as e:
                logger.warning(f"Error in NMS: {e}")
                continue

        # Sort by x-coordinate (left to right)
        keep.sort(key=lambda d: (d[0] + d[1]) / 2)

        # Return final detections
        result = []
        for detection in keep:
            try:
                x_center = (detection[0] + detection[1]) / 2
                result.append((x_center, detection[2], detection[3]))
            except Exception as e:
                logger.warning(f"Error formatting detection: {e}")
                continue

        return result

    except Exception as e:
        logger.error(f"Critical error in parse_tflite_out: {e}")
        return []


# ─────────── Enhanced helper functions ─────────────────────────────────────
CLASS_NAMES = list("0123456789ABCDEFGHIJKLMNOPQRSTUVWXYZ")


def flush_cache():
    """Очищення кешу моделей та збірка сміття."""
    global MODEL_CACHE
    MODEL_CACHE.clear()
    gc.collect()
    logger.info("Model cache flushed and garbage collected")


def preprocess(img: np.ndarray, inp_details: List[Dict], img_size: int) -> np.ndarray:
    """Попередня обробка зображення з покращеною обробкою помилок."""
    try:
        shape = inp_details[0]["shape"]
        dtype = inp_details[0]["dtype"]

        # Determine input format (NCHW vs NHWC)
        is_nchw = len(shape) == 4 and shape[1] == 3

        if is_nchw:
            height, width = shape[2], shape[3]
        else:
            height, width = shape[1], shape[2]

        # Use provided size if shape dimensions are dynamic
        height = height if height and height > 0 else img_size
        width = width if width and width > 0 else img_size

        # Convert and resize image
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        img_resized = cv2.resize(img_rgb, (width, height), interpolation=cv2.INTER_LINEAR)

        # Normalize to [0, 1]
        img_float = img_resized.astype(np.float32) / 255.0

        # Transpose if NCHW format
        if is_nchw:
            img_float = img_float.transpose(2, 0, 1)

        # Add batch dimension
        img_batch = img_float[None, ...]

        # Apply quantization if needed
        if dtype in (np.int8, np.uint8):
            qp = inp_details[0].get("quantization_parameters", {})
            scale = qp.get("scales", [1.0])[0] if qp.get("scales") else 1.0
            zero_point = qp.get("zero_points", [0])[0] if qp.get("zero_points") else 0

            img_batch = (img_batch / scale + zero_point).astype(dtype)

        return img_batch.astype(dtype)

    except Exception as e:
        logger.error(f"Error in preprocessing: {e}")
        raise


def order_string(detections: List[Tuple[float, str, float]]) -> Tuple[str, float]:
    """Упорядкування детекцій у рядок з покращеною обробкою."""
    if not detections:
        return "НЕ РОЗПІЗНАНО", 0.0

    try:
        # Sort by x-coordinate
        detections.sort(key=lambda d: d[0])

        # Extract characters and confidences
        chars = [det[1] for det in detections]
        confidences = [det[2] for det in detections if det[2] > 0]

        result_string = "".join(chars)
        avg_confidence = float(np.mean(confidences)) if confidences else 0.0

        return result_string, avg_confidence

    except Exception as e:
        logger.error(f"Error in order_string: {e}")
        return "ПОМИЛКА", 0.0


def load_names(pt_path: Optional[str]) -> List[str]:
    """Завантаження назв класів з можливим кешуванням."""
    if pt_path and pt_path in MODEL_CACHE:
        cached_model = MODEL_CACHE[pt_path]
        if hasattr(cached_model, 'names') and cached_model.names:
            return list(cached_model.names.values())

    if pt_path and Path(pt_path).exists():
        try:
            from ultralytics import YOLO
            model = YOLO(pt_path)
            if hasattr(model, 'names') and model.names:
                return list(model.names.values())
        except Exception as e:
            logger.warning(f"Could not load names from {pt_path}: {e}")

    return CLASS_NAMES


def select_most_consistent_result(all_detections: List[List[Tuple[float, str, float]]]) -> List[
    Tuple[float, str, float]]:
    """Вибирає найбільш консистентний результат з декількох запусків."""
    if not all_detections or not any(all_detections):
        return []

    try:
        # Convert detections to strings for comparison
        detection_strings = []
        detection_confidences = []

        for detections in all_detections:
            if detections:
                plate_str, avg_conf = order_string(detections)
                detection_strings.append(plate_str)
                detection_confidences.append((detections, avg_conf))
            else:
                detection_strings.append("")
                detection_confidences.append(([], 0.0))

        if not detection_strings:
            return []

        # Count occurrences of each result
        from collections import Counter
        string_counts = Counter(detection_strings)

        # Find most common non-empty result
        most_common = string_counts.most_common()
        best_result = None
        best_confidence = 0.0

        for result_str, count in most_common:
            if result_str and result_str not in ["НЕ РОЗПІЗНАНО", "ПОМИЛКА"]:
                # Find the detection with highest confidence for this string
                for detections, conf in detection_confidences:
                    current_str, _ = order_string(detections)
                    if current_str == result_str and conf > best_confidence:
                        best_result = detections
                        best_confidence = conf
                break

        # If no good result found, use the one with highest confidence
        if best_result is None:
            best_confidence = 0.0
            for detections, conf in detection_confidences:
                if conf > best_confidence:
                    best_result = detections
                    best_confidence = conf

        return best_result or []

    except Exception as e:
        logger.warning(f"Error in consistency selection: {e}")
        # Return first non-empty result as fallback
        for detections in all_detections:
            if detections:
                return detections
        return []


def set_deterministic_behavior():
    """Встановлює детермінований режим для всіх можливих бібліотек."""
    try:
        # Set numpy random seed for reproducibility
        np.random.seed(42)

        # Try to set TensorFlow deterministic behavior
        try:
            import tensorflow as tf
            # Set TF to be deterministic
            tf.config.threading.set_inter_op_parallelism_threads(1)
            tf.config.threading.set_intra_op_parallelism_threads(1)
            # Set random seeds
            tf.random.set_seed(42)
        except ImportError:
            pass

        # Set environment variables for deterministic behavior
        os.environ["TF_DETERMINISTIC_OPS"] = "1"
        os.environ["TF_CUDNN_DETERMINISTIC"] = "1"
        os.environ["PYTHONHASHSEED"] = "42"

    except Exception as e:
        logger.warning(f"Could not set deterministic behavior: {e}")


# ─────────── Enhanced backends ─────────────────────────────────────────────
def run_pt_or_onnx(path: str, img: np.ndarray, names: List[str],
                   threshold: float, runs: int) -> Tuple[str, float, float]:
    """Запуск PT/ONNX моделей з кешуванням та детермінованою поведінкою."""
    try:
        # Check cache first
        if path in MODEL_CACHE:
            model = MODEL_CACHE[path]
        else:
            from ultralytics import YOLO
            model = YOLO(path)
            MODEL_CACHE[path] = model

        # Multiple warmup runs for stability
        for _ in range(3):
            _ = model(img, verbose=False, conf=threshold)

        times = []
        all_detections = []

        for i in range(runs):
            try:
                start_time = time.perf_counter()
                results = model(img, verbose=False, conf=threshold)[0]
                elapsed_time = time.perf_counter() - start_time
                times.append(elapsed_time)

                # Collect detections from all runs
                current_detections = []
                if results.boxes is not None:
                    for box in results.boxes:
                        try:
                            conf = float(box.conf.squeeze())
                            class_id = int(box.cls.squeeze())

                            if conf < threshold or class_id >= len(names):
                                continue

                            x_center = float(box.xywh.squeeze()[0])
                            current_detections.append((x_center, names[class_id], conf))

                        except Exception as e:
                            logger.warning(f"Error processing box: {e}")
                            continue

                all_detections.append(current_detections)

            except Exception as e:
                logger.error(f"Error in inference run {i}: {e}")
                times.append(0.0)
                all_detections.append([])
                continue

        # Use most consistent result
        final_detections = select_most_consistent_result(all_detections)
        plate_str, avg_conf = order_string(final_detections)
        avg_time = float(np.mean(times)) if times else 0.0

        return plate_str, avg_conf, avg_time

    except Exception as e:
        logger.error(f"Error in PT/ONNX inference: {e}")
        return "ПОМИЛКА", 0.0, 0.0


def run_litert(path: str, img: np.ndarray, names: List[str], threshold: float,
               runs: int, img_size: int, iou_threshold: float) -> Tuple[str, float, float]:
    """Запуск LiteRT моделей з детермінованою обробкою."""
    try:
        # Check cache first for consistent behavior
        cache_key = f"{path}_litert"
        if cache_key in MODEL_CACHE:
            interpreter = MODEL_CACHE[cache_key]
        else:
            # Try different import options for TensorFlow Lite
            interpreter = None
            try:
                from ai_edge_litert.interpreter import Interpreter
                interpreter = Interpreter(model_path=path, num_threads=1)  # Single thread for consistency
            except ImportError:
                try:
                    import tflite_runtime.interpreter as tflite
                    interpreter = tflite.Interpreter(model_path=path, num_threads=1)
                except ImportError:
                    import tensorflow as tf
                    interpreter = tf.lite.Interpreter(model_path=path, num_threads=1)

            if interpreter is None:
                raise RuntimeError("No TensorFlow Lite interpreter available")

            interpreter.allocate_tensors()
            MODEL_CACHE[cache_key] = interpreter

        input_details = interpreter.get_input_details()
        output_details = interpreter.get_output_details()

        # Preprocess image once for consistency
        processed_img = preprocess(img, input_details, img_size)

        # Multiple warmup runs for stability
        for _ in range(3):
            interpreter.set_tensor(input_details[0]["index"], processed_img)
            interpreter.invoke()

        times = []
        all_detections = []  # Store all runs for consistency analysis

        for i in range(runs):
            try:
                start_time = time.perf_counter()

                # Reset interpreter state
                interpreter.set_tensor(input_details[0]["index"], processed_img)
                interpreter.invoke()

                elapsed_time = time.perf_counter() - start_time
                times.append(elapsed_time)

                # Collect detections from all runs for consistency check
                current_detections = parse_tflite_out(
                    interpreter, output_details, threshold, names,
                    img_w=img_size, iou_thr=iou_threshold
                )
                all_detections.append(current_detections)

            except Exception as e:
                logger.error(f"Error in LiteRT run {i}: {e}")
                times.append(0.0)
                all_detections.append([])
                continue

        # Use most consistent result or majority vote
        final_detections = select_most_consistent_result(all_detections)

        plate_str, avg_conf = order_string(final_detections)
        avg_time = float(np.mean(times)) if times else 0.0

        return plate_str, avg_conf, avg_time

    except Exception as e:
        logger.error(f"Error in LiteRT inference: {e}")
        return "ПОМИЛКА", 0.0, 0.0


# ─────────── Enhanced CLI and main ─────────────────────────────────────────
def create_parser() -> argparse.ArgumentParser:
    """Створення парсера аргументів командного рядка."""
    parser = argparse.ArgumentParser(
        description="OCR-YOLO tester (LiteRT) - Optimized Version",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter
    )

    parser.add_argument(
        "-m", "--model", "--path",
        dest="model_paths",
        action="append",
        required=True,
        help="Path to model file(s). Can be specified multiple times."
    )

    parser.add_argument(
        "--image",
        required=True,
        help="Path to input image"
    )

    parser.add_argument(
        "--input_size",
        type=int,
        default=320,
        help="Input image size for preprocessing"
    )

    parser.add_argument(
        "--conf",
        type=float,
        default=0.12,
        help="Confidence threshold for detections"
    )

    parser.add_argument(
        "--iou_thr",
        type=float,
        default=0.35,
        help="IoU threshold for NMS"
    )

    parser.add_argument(
        "--runs",
        type=int,
        default=5,
        help="Number of inference runs for timing"
    )

    parser.add_argument(
        "--flush_cache",
        action="store_true",
        help="Flush model cache before and after execution"
    )

    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging"
    )

    parser.add_argument(
        "--deterministic",
        action="store_true",
        help="Enable deterministic mode for consistent results"
    )

    return parser


def validate_args(args: argparse.Namespace) -> bool:
    """Валідація аргументів командного рядка."""
    errors = []

    # Check file existence
    for model_path in args.model_paths:
        if not Path(model_path).exists():
            errors.append(f"Model file not found: {model_path}")

    if not Path(args.image).exists():
        errors.append(f"Image file not found: {args.image}")

    # Validate ranges
    if args.conf < 0 or args.conf > 1:
        errors.append("Confidence threshold must be between 0 and 1")

    if args.iou_thr < 0 or args.iou_thr > 1:
        errors.append("IoU threshold must be between 0 and 1")

    if args.runs < 1:
        errors.append("Number of runs must be at least 1")

    if args.input_size < 32:
        errors.append("Input size must be at least 32")

    if errors:
        for error in errors:
            logger.error(error)
        return False

    return True


def main():
    """Головна функція."""
    parser = create_parser()
    args = parser.parse_args()

    # Set logging level
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)

    # Enable deterministic behavior if requested
    if args.deterministic:
        set_deterministic_behavior()
        logger.info("Deterministic mode enabled")

    # Validate arguments
    if not validate_args(args):
        sys.exit(1)

    # Flush cache if requested
    if args.flush_cache:
        flush_cache()

    try:
        # Load image
        img = cv2.imread(args.image)
        if img is None:
            logger.error(f"Failed to load image: {args.image}")
            sys.exit(1)

        logger.info(f"Loaded image: {args.image} ({img.shape[1]}x{img.shape[0]})")

        # Load class names from first PT model if available
        pt_models = [p for p in args.model_paths if p.endswith(".pt")]
        class_names = load_names(pt_models[0] if pt_models else None)
        logger.info(f"Using {len(class_names)} class names")

        print("\n" + "=" * 60)
        print("                    РЕЗУЛЬТАТИ")
        print("=" * 60)

        successful_runs = 0
        total_time = 0.0

        for model_path in args.model_paths:
            model_name = Path(model_path).name
            extension = Path(model_path).suffix.lower()

            try:
                logger.info(f"Processing model: {model_name}")

                if extension in (".pt", ".onnx"):
                    plate, confidence, avg_time = run_pt_or_onnx(
                        model_path, img, class_names, args.conf, args.runs
                    )
                elif extension == ".tflite":
                    plate, confidence, avg_time = run_litert(
                        model_path, img, class_names, args.conf, args.runs,
                        args.input_size, args.iou_thr
                    )
                else:
                    print(f"[{model_name:>20}]  ❌ Формат не підтримується")
                    continue

                # Display results
                status = "✅" if plate != "НЕ РОЗПІЗНАНО" and plate != "ПОМИЛКА" else "❌"
                print(f"[{model_name:>20}]  {status} Plate: {plate:<15} "
                      f"Conf: {confidence:.3f}  Time: {avg_time * 1000:.1f} ms")

                if avg_time > 0:
                    successful_runs += 1
                    total_time += avg_time

            except Exception as e:
                logger.error(f"Error processing {model_name}: {e}")
                print(f"[{model_name:>20}]  ❌ Помилка виконання")
                continue

        print("=" * 60)

        if successful_runs > 0:
            avg_total_time = total_time / successful_runs
            print(f"Successful runs: {successful_runs}/{len(args.model_paths)}")
            print(f"Average time: {avg_total_time * 1000:.1f} ms")

        print()

    except KeyboardInterrupt:
        logger.info("Interrupted by user")
        sys.exit(130)
    except Exception as e:
        logger.error(f"Unexpected error: {e}")
        sys.exit(1)
    finally:
        # Flush cache if requested
        if args.flush_cache:
            flush_cache()


if __name__ == "__main__":
    main()