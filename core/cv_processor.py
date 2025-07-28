# core/cv_processor.py
import logging
import time
import json
import os
import cv2
import numpy as np
import onnxruntime
from typing import Optional, List, Tuple

try:
    from ultralytics import YOLO
    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

from utils.image_utils import save_image, crop_image, draw_bounding_box, draw_text_with_background

logger = logging.getLogger(__name__)

CHAR_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'I', 'K', 'M', 'O', 'P', 'T', 'X']
TARGET_VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # car, motorcycle, bus, truck
LICENSE_PLATE_CLASS_ID = 0

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
            # Визначаємо розмір входу для моделі НЗ
            self.plate_model_input_size = tuple(self.plate_session.get_inputs()[0].shape[2:])

        self.ocr_model = self._load_yolo_model(ocr_model_path)
        self.roi_config = self._load_roi_config(roi_config_path)
        self._logger.info("CVProcessor успішно ініціалізовано.")

    def _load_onnx_model(self, model_path: str) -> Optional[onnxruntime.InferenceSession]:
        # ... (код без змін)
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
        # ... (код без змін)
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

    def _load_roi_config(self, config_path: str) -> dict:
        # ... (код без змін)
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self._logger.error(f"Помилка завантаження ROI конфігурації: {e}", exc_info=True)
        return {}

    def _apply_roi(self, image_bgr: np.ndarray, camera_type: str) -> np.ndarray:
        # ... (код без змін)
        roi_key = f"{camera_type}_camera_roi"
        if roi_key in self.roi_config and self.roi_config[roi_key].get("enabled", False):
            roi = self.roi_config[roi_key]
            cropped = crop_image(image_bgr, (roi["x1"], roi["y1"], roi["x2"], roi["y2"]))
            return cropped if cropped is not None else image_bgr
        return image_bgr

    def _preprocess_for_mobilenet(self, image_bgr: np.ndarray) -> np.ndarray:
        # ... (код без змін)
        resized_image = cv2.resize(image_bgr, (300, 300))
        return np.expand_dims(resized_image, axis=0)
        
    def _preprocess_image_yolo(self, image_bgr: np.ndarray, target_size: Tuple[int, int]) -> Tuple[np.ndarray, float, Tuple[int, int]]:
        img_h, img_w = image_bgr.shape[:2]
        input_h, input_w = target_size
        scale = min(input_w / img_w, input_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_image = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        
        top_pad = (input_h - new_h) // 2
        bottom_pad = input_h - new_h - top_pad
        left_pad = (input_w - new_w) // 2
        right_pad = input_w - new_w - left_pad
        
        padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad, cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        normalized_image = rgb_image.astype(np.float32) / 255.0
        transposed_image = np.transpose(normalized_image, (2, 0, 1))
        input_tensor = np.expand_dims(transposed_image, axis=0)
        
        return input_tensor, scale, (left_pad, top_pad)

    def _parse_license_plate_output(self, raw_output: np.ndarray, original_shape: tuple, scale: float, pad: tuple) -> list:
        # --- НОВА, ВИПРАВЛЕНА ЛОГІКА ---
        original_h, original_w = original_shape[:2]
        pad_x, pad_y = pad
        
        detections = []
        predictions = raw_output[0] # (N, 6) -> [x1, y1, x2, y2, score, class_id]
        
        for pred in predictions:
            score = float(pred[4])
            class_id = int(pred[5])

            if score < self.plate_confidence_thresh or class_id != LICENSE_PLATE_CLASS_ID:
                continue

            # Координати вже у пікселях на зображенні 448x448
            x1_padded, y1_padded, x2_padded, y2_padded = pred[:4]

            # 1. Повертаємо координати до масштабованого зображення (прибираємо поля)
            x1_resized = x1_padded - pad_x
            y1_resized = y1_padded - pad_y
            x2_resized = x2_padded - pad_x
            y2_resized = y2_padded - pad_y

            # 2. Повертаємо координати до оригінального розміру (ділимо на коеф. масштабування)
            x1_orig = int(x1_resized / scale)
            y1_orig = int(y1_resized / scale)
            x2_orig = int(x2_resized / scale)
            y2_orig = int(y2_resized / scale)

            # 3. Обмежуємо рамки розмірами оригінального зображення
            x1 = max(0, x1_orig)
            y1 = max(0, y1_orig)
            x2 = min(original_w, x2_orig)
            y2 = min(original_h, y2_orig)
            
            if x1 < x2 and y1 < y2:
                detections.append((x1, y1, x2, y2))
                
        return detections

    def detect_vehicle_in_frame(self, image_bgr: np.ndarray, camera_type: str) -> List[Tuple[int, int, int, int]]:
        # ... (код без змін)
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
        
        input_tensor, scale, pad = self._preprocess_image_yolo(vehicle_image, self.plate_model_input_size)
        outputs = self.plate_session.run(None, {self.plate_input_name: input_tensor})
        
        detections = self._parse_license_plate_output(outputs[0], vehicle_image.shape, scale, pad)
        return detections

    def recognize_plate_characters(self, plate_image: np.ndarray) -> Optional[str]:
        # ... (код без змін)
        if not self.ocr_model: return None
        results = self.ocr_model.predict(source=plate_image, conf=self.ocr_confidence_thresh, verbose=False)
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
        # ... (код без змін)
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