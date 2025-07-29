# core/cv_processor.py
import logging
import time
import json
import os
import re
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

from utils.image_utils import save_image, crop_image, draw_bounding_box

logger = logging.getLogger(__name__)

CHAR_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', 'A', 'B', 'C', 'E', 'H', 'I', 'K', 'M', 'O', 'P', 'T', 'X']
TARGET_VEHICLE_CLASS_IDS = [2, 3, 5, 7]
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
            shape = self.plate_session.get_inputs()[0].shape
            self.plate_model_input_height = shape[2]
            self.plate_model_input_width = shape[3]

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

    def _load_roi_config(self, config_path: str) -> dict:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self._logger.error(f"Помилка завантаження ROI конфігурації: {e}", exc_info=True)
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

    def _preprocess_for_plate_detection(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = self.plate_model_input_height, self.plate_model_input_width
        img = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        # Модель очікує BGR, тому конвертація в RGB не потрібна
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1) # HWC to CHW
        return np.expand_dims(img, 0) # Add batch dimension

    def _parse_plate_detections(self, raw_output: np.ndarray, original_shape: tuple) -> List[Tuple[int, int, int, int]]:
        original_h, original_w = original_shape[:2]

        detections = []
        for det in raw_output[0]:
            score = float(det[4])
            class_id = int(det[5])

            if score > self.plate_confidence_thresh and class_id == LICENSE_PLATE_CLASS_ID:
                x1_norm, y1_norm, x2_norm, y2_norm = det[:4]
                x1 = int(x1_norm * original_w)
                y1 = int(y1_norm * original_h)
                x2 = int(x2_norm * original_w)
                y2 = int(y2_norm * original_h)

                detections.append((x1, y1, x2, y2))
        return detections

    def _format_plate_number(self, raw_text: str) -> str:
        """Інтелектуально форматує та виправляє номер до стандарту AA1111AA."""
        if not raw_text:
            return ""

        # 1. Попереднє очищення від усіх символів, крім літер та цифр
        clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

        if len(clean_text) < 7 or len(clean_text) > 8:
            return clean_text # Повертаємо як є, якщо довжина некоректна

        # 2. Словники для корекції на основі позиції
        digit_to_letter = {'0': 'O', '1': 'I', '2': 'Z', '8':'B'}
        letter_to_digit = {'O': '0', 'I': 'I', 'Z': '2', 'B':'8'} # I->I, бо 1 може бути літерою

        # 3. Побудова списку символів з можливою корекцією
        plate_chars = list(clean_text)

        # Перші 2 символи (мають бути літерами)
        for i in range(2):
            char = plate_chars[i]
            if char.isdigit():
                plate_chars[i] = digit_to_letter.get(char, char)

        # Останні 2 символи (мають бути літерами)
        if len(plate_chars) == 8:
            for i in range(6, 8):
                char = plate_chars[i]
                if char.isdigit():
                    plate_chars[i] = digit_to_letter.get(char, char)

        # Середні 4 символи (мають бути цифрами)
        for i in range(2, 6):
            char = plate_chars[i]
            if char.isalpha():
                plate_chars[i] = letter_to_digit.get(char, char)

        formatted_plate = "".join(plate_chars)

        # 4. Фінальна перевірка формату
        if re.fullmatch(r'^[A-Z]{2}\d{4}[A-Z]{2}$', formatted_plate):
            return formatted_plate
        else:
            # Якщо навіть після корекції формат невірний, повертаємо очищений текст
            return clean_text

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

        input_tensor = self._preprocess_for_plate_detection(vehicle_image)
        outputs = self.plate_session.run(None, {self.plate_input_name: input_tensor})

        detections = self._parse_plate_detections(outputs[0], vehicle_image.shape)
        return detections

    def recognize_plate_characters(self, plate_image: np.ndarray) -> Optional[str]:
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
        timestamp = time.strftime('%Y%m%d_%H%M%S')

        vehicle_boxes = self.detect_vehicle_in_frame(image_bgr, cam_type)
        if not vehicle_boxes:
            self._logger.info(f"[{cam_type.upper()}] Автомобіль не виявлено.")
            return None

        vehicle_box = max(vehicle_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        vehicle_image = crop_image(image_bgr, vehicle_box)
        if vehicle_image is None or vehicle_image.size == 0:
            self._logger.warning("Зображення автомобіля порожнє після обрізки.")
            return None

        if save_intermediate_steps:
            save_image(vehicle_image, save_path_prefix, f"{timestamp}_1_vehicle_crop.jpg")

        plate_boxes = self.detect_license_plate(vehicle_image)
        if not plate_boxes:
            self._logger.info(f"[{cam_type.upper()}] Номерний знак не виявлено на автомобілі.")
            return None

        plate_box = max(plate_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))

        h, w = vehicle_image.shape[:2]
        x1, y1, x2, y2 = plate_box
        padding_x = int((x2 - x1) * 0.05)
        padding_y = int((y2 - y1) * 0.2)
        x1 = max(0, x1 - padding_x)
        y1 = max(0, y1 - padding_y)
        x2 = min(w, x2 + padding_x)
        y2 = min(h, y2 + padding_y)
        padded_plate_box = (x1, y1, x2, y2)

        plate_image = crop_image(vehicle_image, padded_plate_box)

        if plate_image is None or plate_image.size == 0:
            self._logger.warning("Зображення номерного знака порожнє після обрізки.")
            return None

        if save_intermediate_steps:
            save_image(plate_image, save_path_prefix, f"{timestamp}_2_plate_crop.jpg")

        raw_plate_text = self.recognize_plate_characters(plate_image)

        if raw_plate_text:
            formatted_plate = self._format_plate_number(raw_plate_text)
            self._logger.info(f"[{cam_type.upper()}] Розпізнано: '{raw_plate_text}', відформатовано: '{formatted_plate}'")

            if save_intermediate_steps:
                final_img = image_bgr.copy()
                draw_bounding_box(final_img, vehicle_box, "Vehicle", color=(0, 255, 0))
                plate_abs_box = (vehicle_box[0] + plate_box[0], vehicle_box[1] + plate_box[1],
                                 vehicle_box[0] + plate_box[2], vehicle_box[1] + plate_box[3])
                draw_bounding_box(final_img, plate_abs_box, formatted_plate, color=(255, 0, 0))
                save_image(final_img, save_path_prefix, f"{timestamp}_3_final_result.jpg")

            return formatted_plate
        else:
            self._logger.info(f"[{cam_type.upper()}] Не вдалося розпізнати символи на номерному знаку.")
            return None
