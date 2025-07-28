# core/cv_processor.py
import logging
import time
import json
import os
import cv2
import numpy as np
import onnxruntime
from typing import Optional, List, Tuple, Dict

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

from utils.image_utils import save_image, crop_image, draw_bounding_box

logger = logging.getLogger(__name__)

CHAR_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'I', 'K', 'M', 'O', 'P', 'T', 'X']
TARGET_VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck

class CVProcessor:
    def __init__(self,
                 mobilenet_ssd_path: str,
                 license_model_path: str,
                 ocr_model_path: str,
                 roi_config_path: str,
                 vehicle_confidence_thresh: float = 0.5,
                 plate_confidence_thresh: float = 0.5,
                 ocr_confidence_thresh: float = 0.4):

        self._logger = logging.getLogger(f"{__name__}.CVProcessor")
        self.vehicle_confidence_thresh = vehicle_confidence_thresh
        self.plate_confidence_thresh = plate_confidence_thresh
        self.ocr_confidence_thresh = ocr_confidence_thresh

        self.vehicle_session = self._load_onnx_model(mobilenet_ssd_path)
        self.plate_session = self._load_onnx_model(license_model_path)

        if self.vehicle_session:
            self.vehicle_input_name = self.vehicle_session.get_inputs()[0].name
        if self.plate_session:
            self.plate_input_name = self.plate_session.get_inputs()[0].name

        self.ocr_model = self._load_yolo_model(ocr_model_path)
        self.roi_config = self._load_roi_config(roi_config_path)
        self._logger.info("CVProcessor успішно ініціалізовано.")

    def _load_onnx_model(self, model_path: str) -> Optional[onnxruntime.InferenceSession]:
        if not os.path.exists(model_path):
            self._logger.error(f"Файл моделі ONNX не знайдено: {model_path}")
            return None
        try:
            session = onnxruntime.InferenceSession(model_path, providers=['CPUExecutionProvider'])
            self._logger.info(f"Модель ONNX успішно завантажено: {model_path}")
            return session
        except Exception as e:
            self._logger.error(f"Не вдалося завантажити модель ONNX з {model_path}: {e}", exc_info=True)
            return None

    def _load_yolo_model(self, model_path: str) -> Optional[YOLO]:
        if not ULTRALYTICS_AVAILABLE:
            self._logger.error("Бібліотека 'ultralytics' не доступна. OCR неможливий.")
            return None
        if not os.path.exists(model_path):
            self._logger.error(f"Файл моделі OCR (YOLO) не знайдено: {model_path}")
            return None
        try:
            model = YOLO(model_path)
            self._logger.info(f"Модель OCR (YOLO) успішно завантажено: {model_path}")
            return model
        except Exception as e:
            self._logger.error(f"Не вдалося завантажити модель OCR (YOLO): {e}", exc_info=True)
            return None

    def _load_roi_config(self, config_path: str) -> Dict:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self._logger.error(f"Помилка завантаження ROI конфігурації: {e}", exc_info=True)
        self._logger.warning(f"Файл ROI конфігурації '{config_path}' не знайдено. ROI не буде використовуватися.")
        return {}

    def _apply_roi(self, image_bgr: np.ndarray, camera_type: str) -> np.ndarray:
        roi_key = f"{camera_type}_camera_roi"
        if roi_key in self.roi_config and self.roi_config[roi_key].get("enabled", False):
            roi = self.roi_config[roi_key]
            cropped = crop_image(image_bgr, (roi["x1"], roi["y1"], roi["x2"], roi["y2"]))
            return cropped if cropped is not None else image_bgr
        return image_bgr

    def _preprocess_for_mobilenet(self, image_bgr: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image_bgr, (300, 300))
        return np.expand_dims(resized_image, axis=0)

    def _preprocess_for_yolo_onnx(self, image_bgr: np.ndarray, target_size=(448, 448)) -> np.ndarray:
        img_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
        resized_image = cv2.resize(img_rgb, target_size)
        img_normalized = resized_image.astype(np.float32) / 255.0
        img_transposed = np.transpose(img_normalized, (2, 0, 1))
        return np.expand_dims(img_transposed, axis=0)

    def detect_vehicle_in_frame(self, image_bgr: np.ndarray, camera_type: str) -> List[Tuple[int, int, int, int]]:
        if not self.vehicle_session: return []

        img_for_detection = self._apply_roi(image_bgr, camera_type)
        input_tensor = self._preprocess_for_mobilenet(img_for_detection)

        outputs = self.vehicle_session.run(None, {self.vehicle_input_name: input_tensor})

        detections = []
        boxes, classes, scores, num_detections = outputs[0], outputs[1], outputs[2], outputs[3]

        count = int(num_detections[0])
        for i in range(count):
            class_id = int(classes[0][i])
            score = scores[0][i]
            if score > self.vehicle_confidence_thresh and class_id in TARGET_VEHICLE_CLASS_IDS:
                box = boxes[0][i]
                h, w = img_for_detection.shape[:2]
                y_min, x_min, y_max, x_max = int(box[0] * h), int(box[1] * w), int(box[2] * h), int(box[3] * w)
                detections.append((x_min, y_min, x_max, y_max))
        return detections

    def detect_license_plate(self, vehicle_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if not self.plate_session: return []

        input_tensor = self._preprocess_for_yolo_onnx(vehicle_image, target_size=(448, 448))
        outputs = self.plate_session.run(None, {self.plate_input_name: input_tensor})

        detections = []
        for detection in outputs[0][0]:
            confidence = detection[4]
            if confidence > self.plate_confidence_thresh:
                cx, cy, w, h = detection[:4]
                x1 = int((cx - w / 2) * vehicle_image.shape[1])
                y1 = int((cy - h / 2) * vehicle_image.shape[0])
                x2 = int((cx + w / 2) * vehicle_image.shape[1])
                y2 = int((cy + h / 2) * vehicle_image.shape[0])
                detections.append((x1, y1, x2, y2))
        return detections

    def recognize_plate_characters(self, plate_image: np.ndarray) -> Optional[str]:
        if not self.ocr_model: return None

        # --- ВИПРАВЛЕННЯ ТУТ ---
        # Передаємо оригінальне 3-канальне зображення (BGR) напряму в модель
        results = self.ocr_model.predict(source=plate_image, conf=self.ocr_confidence_thresh, verbose=False)
        # -------------------------

        if not results or not results[0].boxes: return None

        boxes = results[0].boxes.data.cpu().numpy()
        sorted_boxes = sorted(boxes, key=lambda b: b[0])

        plate_text = ""
        for box in sorted_boxes:
            class_id = int(box[5])
            if 0 <= class_id < len(CHAR_LIST):
                plate_text += CHAR_LIST[class_id]

        return plate_text if plate_text else None

    def get_plate_number_from_image(self, image_bgr: np.ndarray, cam_type: str, save_intermediate_steps: bool = False,
                                    save_path_prefix: str = "debug_cv") -> Optional[str]:
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        vehicle_boxes = self.detect_vehicle_in_frame(image_bgr, cam_type)
        if not vehicle_boxes:
            self._logger.info(f"[{cam_type.upper()}] Автомобіль не виявлено.")
            return None

        vehicle_box = max(vehicle_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        vehicle_image = crop_image(image_bgr, vehicle_box)
        if vehicle_image is None: return None

        if save_intermediate_steps:
            save_image(vehicle_image, save_path_prefix, f"{timestamp}_1_vehicle_crop.jpg")

        plate_boxes = self.detect_license_plate(vehicle_image)
        if not plate_boxes:
            self._logger.info(f"[{cam_type.upper()}] Номерний знак не виявлено на автомобілі.")
            return None

        plate_box = plate_boxes[0]
        plate_image = crop_image(vehicle_image, plate_box)
        if plate_image is None: return None

        if save_intermediate_steps:
            save_image(plate_image, save_path_prefix, f"{timestamp}_2_plate_crop.jpg")

        plate_text = self.recognize_plate_characters(plate_image)
        if plate_text:
            self._logger.info(f"[{cam_type.upper()}] Розпізнано номер: {plate_text}")
            if save_intermediate_steps:
                final_img = image_bgr.copy()
                draw_bounding_box(final_img, vehicle_box, "Vehicle", color=(0, 255, 0))
                plate_abs_box = (vehicle_box[0] + plate_box[0], vehicle_box[1] + plate_box[1],
                                 vehicle_box[0] + plate_box[2], vehicle_box[1] + plate_box[3])
                draw_bounding_box(final_img, plate_abs_box, plate_text, color=(255, 0, 0))
                save_image(final_img, save_path_prefix, f"{timestamp}_3_final_result.jpg")
        else:
            self._logger.info(f"[{cam_type.upper()}] Не вдалося розпізнати символи на номерному знаку.")

        return plate_text