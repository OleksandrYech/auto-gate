#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
video_model_test.py

Бенчмарк відео-детекції для моделей:
  * TensorFlow Lite (.tflite)
  * PyTorch / Ultralytics YOLO (.pt)
  * ONNX (.onnx)

Вимірює FPS (інференс-кадри / чистий час), середню впевненість та
загальний час виконання.

Використання приклад:

  python licensedet.py --model model.tflite/onnx/pt --video video.mp4 --display --score-thr 0.3 --device 0 --imgsz 448

Залежності: opencv-python, numpy, (tflite_runtime | tensorflow), ultralytics,
onnxruntime.  Python ≥ 3.8.
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path
from typing import List, Sequence, Tuple

import cv2
import numpy as np

# ---------- Опційні імпорти (не всі потрібні одразу) -----------------------
try:
    from tflite_runtime.interpreter import Interpreter  # lightweight
except ImportError:  # noqa: WPS440
    try:
        import tensorflow as tf  # type: ignore
        Interpreter = tf.lite.Interpreter  # noqa: N806
    except ImportError:
        Interpreter = None  # type: ignore

try:
    from ultralytics import YOLO
except ImportError:
    YOLO = None  # type: ignore

try:
    import onnxruntime as ort
except ImportError:
    ort = None  # type: ignore


# ------------------------- Допоміжні типи ----------------------------------
BBox = Tuple[float, float, float, float]         # x1, y1, x2, y2 (нормаліз.)
Detection = Tuple[BBox, float, int]              # bbox, score, cls


# ----------------------------- Препроцес -----------------------------------
def preprocess_bgr(frame: np.ndarray,
                   size: int,
                   nchw: bool = False) -> np.ndarray:
    """
    Кадр BGR -> float32 тензор [1,H,W,3] чи [1,3,H,W] 0-1.

    Parameters
    ----------
    frame: BGR uint8 (H,W,3)
    size:  цільовий розмір (квадрат)
    nchw:  якщо True, повертає [1,3,H,W] (ONNX NCHW)

    Returns
    -------
    np.ndarray
    """
    img = cv2.resize(frame, (size, size), interpolation=cv2.INTER_LINEAR)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32) / 255.0
    if nchw:                                     # -> [1, 3, H, W]
        img = img.transpose(2, 0, 1)
    return np.expand_dims(img, 0)


# ---------------------- Парсинг детекцій TFLite/ONNX -----------------------
def parse_raw_detections(raw: np.ndarray,
                         conf_thr: float) -> List[Detection]:
    """
    Очікує raw shape (1, N, 6): [x1,y1,x2,y2,score,cls].

    Координати вважаються нормалізованими (0-1).
    """
    dets: List[Detection] = []
    for det in raw[0]:
        score = float(det[4])
        if score < conf_thr:
            continue
        x1, y1, x2, y2 = map(float, det[:4])
        cls = int(det[5])
        dets.append(((x1, y1, x2, y2), score, cls))
    return dets


# ----------------------------- Головне -------------------------------------
def main() -> None:  # noqa: WPS231
    parser = argparse.ArgumentParser(description="Video model benchmark")
    parser.add_argument("--model", required=True, type=Path,
                        help=".tflite | .pt | .onnx")
    parser.add_argument("--video", default="cam",
                        help="'cam' | відео-файл | rtsp-url")
    parser.add_argument("--score-thr", type=float, default=0.25,
                        help="Поріг для впевненості")
    parser.add_argument("--imgsz", type=int, default=448,
                        help="Сторона кадру для модельного інпуту")
    parser.add_argument("--threads", type=int, default=4,
                        help="TFLite threads / ONNX intra_op")
    parser.add_argument("--device", default="cpu",
                        help="PyTorch: 'cpu' | '0' | 'cuda:0'")
    parser.add_argument("--display", action="store_true",
                        help="Показувати live-вікно з детекціями")
    args = parser.parse_args()

    model_path: Path = args.model
    suffix = model_path.suffix.lower()

    if suffix == ".tflite":
        if Interpreter is None:
            raise ImportError("Немає tflite_runtime / tensorflow")
        detector = _TFLiteDetector(model_path, args.threads)

    elif suffix == ".pt":
        if YOLO is None:
            raise ImportError("Бібліотека 'ultralytics' не встановлена")
        detector = _PTDetector(model_path, args.device, args.imgsz, args.score_thr)

    elif suffix == ".onnx":
        if ort is None:
            raise ImportError("Бібліотека 'onnxruntime' не встановлена")
        detector = _ONNXDetector(model_path, args.threads, args.imgsz, args.score_thr)

    else:
        raise ValueError("Непідтримуваний формат моделі")

    # ------------- відео-джерело -------------------------------------------
    cap = (cv2.VideoCapture(0) if args.video == "cam"
           else cv2.VideoCapture(args.video))
    if not cap.isOpened():
        raise RuntimeError(f"Не вдалося відкрити «{args.video}»")

    frame_count, det_count = 0, 0
    conf_sum, infer_time = 0.0, 0.0
    overall_start = time.perf_counter()

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            t0 = time.perf_counter()
            detections = detector.infer(frame)
            infer_time += time.perf_counter() - t0

            if detections:
                scores = [d[1] for d in detections]
                conf_sum += sum(scores)
                det_count += len(scores)

            frame_count += 1
            if args.display:
                _draw_detections(frame, detections)
                cv2.imshow("Detections", frame)
                if cv2.waitKey(1) & 0xFF == 27:  # Esc
                    break
    finally:
        cap.release()
        cv2.destroyAllWindows()

    overall = time.perf_counter() - overall_start
    fps = frame_count / infer_time if infer_time else 0
    mean_conf = conf_sum / det_count if det_count else 0

    print("\n--- Підсумок ---")
    print(f"Кадрів оброблено      : {frame_count}")
    print(f"Чистий час інференсу  : {infer_time:.2f} c")
    print(f"FPS (інференс)        : {fps:.2f}")
    print(f"Середня впевненість   : {mean_conf:.3f}")
    print(f"Загальний час роботи  : {overall:.2f} c")


# --------------------- Детектори-обгортки -----------------------------------
class _TFLiteDetector:
    """TFLite модель із виходом (1,N,6)."""

    def __init__(self, model: Path, threads: int) -> None:
        self.inter = Interpreter(model_path=str(model), num_threads=threads)
        self.inter.allocate_tensors()
        self.in_idx = self.inter.get_input_details()[0]["index"]
        self.out_idx = self.inter.get_output_details()[0]["index"]
        shp = self.inter.get_input_details()[0]["shape"]
        self.imgsz = int(shp[1])

    def infer(self, frame: np.ndarray) -> List[Detection]:
        inp = preprocess_bgr(frame, self.imgsz)
        self.inter.set_tensor(self.in_idx, inp)
        self.inter.invoke()
        raw = self.inter.get_tensor(self.out_idx)
        return parse_raw_detections(raw, conf_thr=0.25)


class _PTDetector:
    """Ultralytics YOLO (.pt)."""

    def __init__(self, model: Path, device: str,
                 imgsz: int, conf_thr: float) -> None:
        self.model = YOLO(str(model), task="detect")
        self.model.fuse()
        self.device = device
        self.imgsz = imgsz
        self.conf_thr = conf_thr

    def infer(self, frame: np.ndarray) -> List[Detection]:
        res = self.model.predict(
            source=frame,
            imgsz=self.imgsz,
            device=self.device,
            conf=self.conf_thr,
            verbose=False,
            stream=False,
        )[0]

        h, w = frame.shape[:2]
        dets: List[Detection] = []
        if res.boxes is not None:
            for box in res.boxes:
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()  # pixels
                score = float(box.conf[0])
                cls = int(box.cls[0])
                dets.append((
                    (x1 / w, y1 / h, x2 / w, y2 / h),
                    score,
                    cls,
                ))
        return dets


class _ONNXDetector:
    """ONNX модель із виходом (1,N,6).  Автовизначає NHWC/NCHW вхід."""

    def __init__(self, model: Path, threads: int,
                 imgsz: int, conf_thr: float) -> None:
        opts = ort.SessionOptions()
        opts.intra_op_num_threads = threads
        self.sess = ort.InferenceSession(str(model), opts, providers=["CPUExecutionProvider"])
        self.inp_name = self.sess.get_inputs()[0].name
        shape = self.sess.get_inputs()[0].shape
        self.nchw = (shape[1] == 3)  # якщо (1,3,H,W) -> True
        self.imgsz = imgsz
        self.conf_thr = conf_thr

    def infer(self, frame: np.ndarray) -> List[Detection]:
        inp = preprocess_bgr(frame, self.imgsz, nchw=self.nchw)
        raw = self.sess.run(None, {self.inp_name: inp})[0]
        return parse_raw_detections(raw, self.conf_thr)


# ------------------------ Відмальовування ----------------------------------
def _draw_detections(frame: np.ndarray,
                     detections: Sequence[Detection]) -> None:
    """Накладає bbox та score на кадр (BGR)."""
    h, w = frame.shape[:2]
    for (x1, y1, x2, y2), score, _ in detections:
        pt1 = (int(x1 * w), int(y1 * h))
        pt2 = (int(x2 * w), int(y2 * h))
        cv2.rectangle(frame, pt1, pt2, (0, 255, 0), 2)
        cv2.putText(
            frame, f"{score:.2f}", (pt1[0], pt1[1] - 5),
            cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1,
        )


if __name__ == "__main__":
    main()
