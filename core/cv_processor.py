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

# EasyOCR імпортується як основний інструмент для розпізнавання
import easyocr

from utils.image_utils import save_image, crop_image, draw_bounding_box

logger = logging.getLogger(__name__)

# Константи для класів об'єктів у моделях
TARGET_VEHICLE_CLASS_IDS = [2, 3, 5, 7]  # ID для легкових авто, мотоциклів, автобусів, вантажівок
LICENSE_PLATE_CLASS_ID = 0 # ID для номерного знака

def _check_bbox_roi_intersection(bbox: Tuple[int, int, int, int], roi: Dict[str, int]) -> bool:
    """Перевіряє, чи перетинається рамка об'єкта (bbox) із зоною інтересу (ROI)."""
    car_x1, car_y1, car_x2, car_y2 = bbox
    # Якщо ROI неактивний або неправильно налаштований, вважаємо, що перетин є
    if not roi.get("enabled") or roi['x2'] <= roi['x1'] or roi['y2'] <= roi['y1']:
        return True

    roi_x1, roi_y1, roi_x2, roi_y2 = roi['x1'], roi['y1'], roi['x2'], roi['y2']

    # Логіка перевірки перетину
    if car_x2 < roi_x1 or car_x1 > roi_x2 or car_y2 < roi_y1 or car_y1 > roi_y2:
        return False # Немає перетину
    return True # Є перетин


class CVProcessor:
    def __init__(self,
                 mobilenet_ssd_path: str,
                 license_model_path: str,
                 roi_config_path: str,
                 vehicle_confidence_thresh: float = 0.5,
                 plate_confidence_thresh: float = 0.5):

        self._logger = logging.getLogger(f"{__name__}.CVProcessor")
        self.vehicle_confidence_thresh = vehicle_confidence_thresh
        self.plate_confidence_thresh = plate_confidence_thresh

        # Завантаження моделей ONNX
        self.vehicle_session = self._load_onnx_model(mobilenet_ssd_path)
        self.plate_session = self._load_onnx_model(license_model_path)

        # Ініціалізація EasyOCR
        self._logger.info("Ініціалізація EasyOCR...")
        self.ocr_reader = easyocr.Reader(['en'], gpu=False)
        self._logger.info("EasyOCR успішно ініціалізовано.")

        # --- НОВИЙ КОД: ОПТИМІЗАЦІЯ №1 ---
        # Створюємо "білий список" символів, які можуть бути на українських номерах.
        # Це значно прискорює роботу OCR та підвищує точність.
        self.OCR_ALLOWLIST = 'ABCEHIKMOPTXYZ0123456789'
        self._logger.info(f"Для OCR встановлено білий список символів: {self.OCR_ALLOWLIST}")
        # --- КІНЕЦЬ НОВОГО КОДУ ---

        if self.vehicle_session:
            self.vehicle_input_name = self.vehicle_session.get_inputs()[0].name
        if self.plate_session:
            self.plate_input_name = self.plate_session.get_inputs()[0].name
            shape = self.plate_session.get_inputs()[0].shape
            self.plate_model_input_height = shape[2]
            self.plate_model_input_width = shape[3]

        self.roi_config = self._load_roi_config(roi_config_path)
        self._logger.info("CVProcessor успішно ініціалізовано.")

    def _format_plate_number(self, raw_text: str) -> Optional[str]:
        """
        Очищує, виправляє та валідує номерний знак.
        Повертає номер, тільки якщо він відповідає стандарту AA1111AA, інакше повертає None.
        """
        if not raw_text:
            return None

        # 1. Попередня очистка: видаляємо все, крім літер та цифр, переводимо у верхній регістр.
        clean_text = re.sub(r'[^A-Z0-9]', '', raw_text.upper())

        # 2. Перевірка довжини. Для стандарту AA1111AA довжина має бути 8.
        if len(clean_text) != 8:
            self._logger.debug(f"Номер '{clean_text}' відхилено через некоректну довжину ({len(clean_text)}).")
            return None

        # 3. Інтелектуальне виправлення символів на основі їх позиції
        plate_chars = list(clean_text)
        digit_to_letter = {'0': 'O', '1': 'I', '8': 'B'}
        letter_to_digit = {'O': '0', 'I': '1', 'B': '8', 'A': '4'}

        for i in [0, 1, 6, 7]:  # Позиції для літер
            if plate_chars[i].isdigit(): plate_chars[i] = digit_to_letter.get(plate_chars[i], plate_chars[i])

        for i in range(2, 6):  # Позиції для цифр
            if plate_chars[i].isalpha(): plate_chars[i] = letter_to_digit.get(plate_chars[i], plate_chars[i])

        corrected_plate = "".join(plate_chars)

        # 4. Фінальна валідація за допомогою регулярного виразу
        if re.fullmatch(r'^[A-Z]{2}\d{4}[A-Z]{2}$', corrected_plate):
            self._logger.debug(f"Номер '{raw_text}' -> '{corrected_plate}' успішно валідовано.")
            return corrected_plate
        else:
            self._logger.warning(f"Номер '{raw_text}' -> '{corrected_plate}' відхилено, оскільки він не відповідає стандарту AA1111AA.")
            return None

    def recognize_plate_characters(self, plate_image: np.ndarray) -> Optional[str]:
        """Розпізнає символи на зображенні номерного знака за допомогою EasyOCR."""
        try:
            # --- НОВИЙ КОД: ОПТИМІЗАЦІЯ №2 ---
            # 1. Перетворюємо зображення в градації сірого. Колір для OCR не потрібен.
            gray_plate = cv2.cvtColor(plate_image, cv2.COLOR_BGR2GRAY)

            # 2. Застосовуємо CLAHE для адаптивного вирівнювання гістограми.
            # Це значно покращує локальний контраст і робить символи чіткішими,
            # особливо в умовах поганого освітлення.
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            enhanced_plate = clahe.apply(gray_plate)
            # --- КІНЕЦЬ НОВОГО КОДУ ---

            # Передаємо покращене зображення та "білий список" в OCR
            result = self.ocr_reader.readtext(
                enhanced_plate,
                detail=0,
                paragraph=True,
                allowlist=self.OCR_ALLOWLIST # Використовуємо наш білий список
            )

            if not result:
                return None

            # Об'єднуємо результат в один рядок, видаляючи пробіли
            return "".join(result).replace(" ", "").upper()

        except Exception as e:
            self._logger.error(f"Помилка під час роботи EasyOCR: {e}", exc_info=True)
            return None

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

    def _load_roi_config(self, config_path: str) -> dict:
        if os.path.exists(config_path):
            try:
                with open(config_path, 'r') as f:
                    return json.load(f)
            except Exception as e:
                self._logger.warning(f"Помилка завантаження ROI конфігурації: {e}.")
        return {}

    def _preprocess_for_mobilenet(self, image_bgr: np.ndarray) -> np.ndarray:
        resized_image = cv2.resize(image_bgr, (300, 300))
        return np.expand_dims(resized_image, axis=0)

    def _preprocess_for_plate_detection(self, image_bgr: np.ndarray) -> np.ndarray:
        h, w = self.plate_model_input_height, self.plate_model_input_width
        img = cv2.resize(image_bgr, (w, h), interpolation=cv2.INTER_LINEAR)
        img = img.astype(np.float32) / 255.0
        img = img.transpose(2, 0, 1)
        return np.expand_dims(img, 0)

    def _parse_plate_detections(self, raw_output: np.ndarray, original_shape: tuple) -> List[Tuple[int, int, int, int]]:
        original_h, original_w = original_shape[:2]
        detections = []
        for det in raw_output[0]:
            score, class_id = float(det[4]), int(det[5])
            if score > self.plate_confidence_thresh and class_id == LICENSE_PLATE_CLASS_ID:
                x1, y1, x2, y2 = map(int, (det[0] * original_w, det[1] * original_h, det[2] * original_w, det[3] * original_h))
                detections.append((x1, y1, x2, y2))
        return detections

    def detect_vehicle_in_frame(self, image_bgr: np.ndarray, camera_type: str) -> List[Tuple[int, int, int, int]]:
        if not self.vehicle_session: return []

        input_tensor = self._preprocess_for_mobilenet(image_bgr)
        outputs = self.vehicle_session.run(None, {self.vehicle_input_name: input_tensor})

        all_detections = []
        boxes, classes, scores, _ = outputs[0], outputs[1], outputs[2], outputs[3]
        h, w = image_bgr.shape[:2]

        for i in range(len(scores[0])):
            class_id, score = int(classes[0][i]), scores[0][i]
            if score > self.vehicle_confidence_thresh and class_id in TARGET_VEHICLE_CLASS_IDS:
                box = boxes[0][i]
                y_min, x_min, y_max, x_max = int(box[0] * h), int(box[1] * w), int(box[2] * h), int(box[3] * w)
                all_detections.append((x_min, y_min, x_max, y_max))

        # Фільтруємо знайдені автомобілі за зоною інтересу (ROI)
        roi_key = f"{camera_type}_camera_roi"
        roi_settings = self.roi_config.get(roi_key)

        if roi_settings:
            return [bbox for bbox in all_detections if _check_bbox_roi_intersection(bbox, roi_settings)]

        return all_detections

    def detect_license_plate(self, vehicle_image: np.ndarray) -> List[Tuple[int, int, int, int]]:
        if not self.plate_session: return []
        input_tensor = self._preprocess_for_plate_detection(vehicle_image)
        outputs = self.plate_session.run(None, {self.plate_input_name: input_tensor})
        return self._parse_plate_detections(outputs[0], vehicle_image.shape)

    def get_plate_number_from_image(self, image_bgr: np.ndarray, cam_type: str, save_intermediate_steps: bool = False,
                                    save_path_prefix: str = "debug_cv") -> Tuple[Optional[str], Optional[str]]:
        timestamp = time.strftime('%Y%m%d_%H%M%S')
        final_image_path = None

        vehicle_boxes = self.detect_vehicle_in_frame(image_bgr, cam_type)
        if not vehicle_boxes:
            self._logger.info(f"[{cam_type.upper()}] Автомобілі, що перетинають ROI, не виявлено.")
            return None, None

        vehicle_box = max(vehicle_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        vehicle_image = crop_image(image_bgr, vehicle_box)
        if vehicle_image is None or vehicle_image.size == 0:
            self._logger.warning("Зображення автомобіля порожнє після обрізки.")
            return None, None

        if save_intermediate_steps: save_image(vehicle_image, save_path_prefix, f"{timestamp}_1_vehicle_crop.jpg")

        plate_boxes = self.detect_license_plate(vehicle_image)
        if not plate_boxes:
            self._logger.info(f"[{cam_type.upper()}] Номерний знак не виявлено на автомобілі.")
            return None, None

        plate_box = max(plate_boxes, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
        h, w = vehicle_image.shape[:2]
        x1, y1, x2, y2 = plate_box
        padding_x, padding_y = int((x2 - x1) * 0.05), int((y2 - y1) * 0.2)
        padded_plate_box = (max(0, x1 - padding_x), max(0, y1 - padding_y), min(w, x2 + padding_x), min(h, y2 + padding_y))

        plate_image = crop_image(vehicle_image, padded_plate_box)
        if plate_image is None or plate_image.size == 0:
            self._logger.warning("Зображення номерного знака порожнє після обрізки.")
            return None, None

        if save_intermediate_steps: save_image(plate_image, save_path_prefix, f"{timestamp}_2_plate_crop.jpg")

        raw_plate_text = self.recognize_plate_characters(plate_image)
        formatted_plate = self._format_plate_number(raw_plate_text)

        if formatted_plate:
            self._logger.info(f"[{cam_type.upper()}] Розпізнано та валідовано: '{raw_plate_text}' -> '{formatted_plate}'")
            if save_intermediate_steps:
                final_img = image_bgr.copy()
                draw_bounding_box(final_img, vehicle_box, "Vehicle", color=(0, 255, 0))
                plate_abs_box = (vehicle_box[0] + plate_box[0], vehicle_box[1] + plate_box[1],
                                 vehicle_box[0] + plate_box[2], vehicle_box[1] + plate_box[3])
                draw_bounding_box(final_img, plate_abs_box, formatted_plate, color=(255, 0, 0))
                final_image_path = os.path.join(save_path_prefix, f"{timestamp}_3_final_result.jpg")
                save_image(final_img, save_path_prefix, os.path.basename(final_image_path))
            return formatted_plate, final_image_path
        else:
            self._logger.info(f"[{cam_type.upper()}] Номер '{raw_plate_text}' розпізнано, але він не пройшов валідацію.")
            return None, None
