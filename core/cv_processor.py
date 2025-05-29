# core/cv_processor.py
import logging
import time
import json
import os
import cv2
import numpy as np
import onnxruntime

try:
    from ultralytics import YOLO

    ULTRALYTICS_AVAILABLE = True
except ImportError:
    ULTRALYTICS_AVAILABLE = False
    YOLO = None

# Імпорт з image_utils
from utils.image_utils import save_image, crop_image, draw_bounding_box, draw_text_with_background

logger = logging.getLogger(__name__)

CHAR_LIST = ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9',
             'A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J',
             'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T',
             'U', 'V', 'W', 'X', 'Y', 'Z']

COCO_LABEL_MAP = {
    1: 'person', 2: 'bicycle', 3: 'car', 4: 'motorcycle', 5: 'airplane',
    6: 'bus', 7: 'train', 8: 'truck', 9: 'boat', 10: 'traffic light',
}
TARGET_VEHICLE_CLASS_IDS = [3, 4, 6, 8]
LICENSE_PLATE_CLASS_ID = 0


class CVProcessor:
    def __init__(self,
                 mobilenet_ssd_path: str,
                 license_model_path: str,
                 ocr_model_path: str,
                 roi_config_path: str = None,
                 vehicle_confidence_thresh: float = 0.5,
                 plate_confidence_thresh: float = 0.6,
                 ocr_confidence_thresh: float = 0.4,
                 ocr_nms_thresh: float = 0.35,
                 plate_nms_thresh: float = 0.45,
                 vehicle_input_target_size: tuple = (300, 300),
                 ocr_input_target_size_for_ultralytics: int = 320
                 ):
        self._logger = logging.getLogger(f"{__name__}.CVProcessor")
        self._logger.info("Ініціалізація CVProcessor...")

        self.vehicle_confidence_thresh = vehicle_confidence_thresh
        self.plate_confidence_thresh = plate_confidence_thresh
        self.ocr_confidence_thresh = ocr_confidence_thresh
        self.ocr_nms_thresh = ocr_nms_thresh
        self.plate_nms_thresh = plate_nms_thresh

        self.vehicle_session = self._load_onnx_model(mobilenet_ssd_path)
        self.plate_session = self._load_onnx_model(license_model_path)

        self.ocr_model_ultralytics = None
        if ULTRALYTICS_AVAILABLE:
            try:
                if ocr_model_path and os.path.exists(ocr_model_path):
                    self.ocr_model_ultralytics = YOLO(ocr_model_path)
                    self._logger.info(f"Модель OCR '{ocr_model_path}' успішно завантажено через Ultralytics YOLO.")
                else:
                    self._logger.error(f"Файл моделі OCR '{ocr_model_path}' не знайдено.")
            except Exception as e:
                self._logger.error(f"Не вдалося завантажити модель OCR '{ocr_model_path}' через Ultralytics YOLO: {e}",
                                   exc_info=True)
        else:
            self._logger.error("Бібліотека 'ultralytics' не доступна. OCR через YOLO неможливий.")

        self.vehicle_input_name = "image_tensor:0" if self.vehicle_session else None
        self.vehicle_output_names_map = {
            "boxes": "detection_boxes:0", "classes": "detection_classes:0",
            "scores": "detection_scores:0", "num": "num_detections:0"
        }
        if self.vehicle_session:
            actual_vehicle_outputs = [output.name for output in self.vehicle_session.get_outputs()]
            for key, name_val in self.vehicle_output_names_map.items():
                if name_val not in actual_vehicle_outputs:
                    self._logger.error(
                        f"Очікуване вихідне ім'я '{name_val}' для MobileNet SSD (для '{key}') не знайдено. Знайдено: {actual_vehicle_outputs}")

        self.plate_input_name = self.plate_session.get_inputs()[0].name if self.plate_session else None
        self.plate_output_names = [output.name for output in
                                   self.plate_session.get_outputs()] if self.plate_session else None

        self.vehicle_model_input_size = vehicle_input_target_size
        self._logger.info(f"Цільовий розмір для cv2.resize (vehicle model): {self.vehicle_model_input_size}")

        default_plate_w, default_plate_h = 448, 448
        self.ocr_model_ultralytics_input_size = ocr_input_target_size_for_ultralytics

        if self.plate_session:
            try:
                plate_input_shape = self.plate_session.get_inputs()[0].shape
                h_plate, w_plate = plate_input_shape[2], plate_input_shape[3]
                self.plate_model_input_size = (int(w_plate), int(h_plate))
            except:
                self.plate_model_input_size = (default_plate_w, default_plate_h)
            self._logger.info(f"Plate detection model (ONNX) input size для cv2.resize: {self.plate_model_input_size}")
        else:
            self.plate_model_input_size = (default_plate_w, default_plate_h)

        self.roi_config = {}
        if roi_config_path and os.path.exists(roi_config_path):
            try:
                with open(roi_config_path, 'r') as f:
                    self.roi_config = json.load(f)
                self._logger.info(f"Конфігурацію ROI завантажено з {roi_config_path}: {self.roi_config}")
            except Exception as e:
                self._logger.error(f"Не вдалося завантажити конфігурацію ROI з {roi_config_path}: {e}", exc_info=True)
        else:
            self._logger.warning(f"Файл конфігурації ROI '{roi_config_path}' не знайдено або не вказано.")

        self._logger.info("CVProcessor успішно ініціалізовано.")

    def _iou1d(self, box1_x1x2: tuple, box2_x1x2: tuple) -> float:
        x1_inter = max(box1_x1x2[0], box2_x1x2[0])
        x2_inter = min(box1_x1x2[1], box2_x1x2[1])
        intersection = max(0.0, x2_inter - x1_inter)
        if intersection == 0: return 0.0
        len1 = box1_x1x2[1] - box1_x1x2[0]
        len2 = box2_x1x2[1] - box2_x1x2[0]
        union = len1 + len2 - intersection
        return intersection / (union + 1e-6)

    def _load_onnx_model(self, model_path: str):
        if not model_path or not os.path.exists(model_path):
            self._logger.error(f"Файл моделі ONNX не знайдено за шляхом: {model_path}")
            return None
        try:
            providers = ['CPUExecutionProvider']
            session = onnxruntime.InferenceSession(model_path, providers=providers)
            self._logger.info(f"Модель ONNX успішно завантажено з: {model_path} (Провайдер: {session.get_providers()})")
            return session
        except Exception as e:
            self._logger.error(f"Не вдалося завантажити модель ONNX з {model_path}: {e}", exc_info=True)
            return None

    def _preprocess_image_yolo(self, image_bgr: np.ndarray, target_size: tuple) -> tuple[
        np.ndarray, float, tuple[int, int]]:
        img_h, img_w = image_bgr.shape[:2]
        input_w, input_h = target_size
        scale = min(input_w / img_w, input_h / img_h)
        new_w, new_h = int(img_w * scale), int(img_h * scale)
        resized_image = cv2.resize(image_bgr, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
        top_pad = (input_h - new_h) // 2
        bottom_pad = input_h - new_h - top_pad
        left_pad = (input_w - new_w) // 2
        right_pad = input_w - new_w - left_pad
        padded_image = cv2.copyMakeBorder(resized_image, top_pad, bottom_pad, left_pad, right_pad,
                                          cv2.BORDER_CONSTANT, value=(114, 114, 114))
        rgb_image = cv2.cvtColor(padded_image, cv2.COLOR_BGR2RGB)
        normalized_image = rgb_image.astype(np.float32) / 255.0
        transposed_image = np.transpose(normalized_image, (2, 0, 1))
        input_tensor = np.expand_dims(transposed_image, axis=0)
        return input_tensor, scale, (left_pad, top_pad)

    def _preprocess_image_mobilenet_ssd(self, image_bgr: np.ndarray, target_size: tuple) -> np.ndarray:
        self._logger.debug(
            f"Preprocessing for SSD. Input image shape: {image_bgr.shape}, target_size for resize: {target_size}")
        resized_image_bgr = cv2.resize(image_bgr, target_size, interpolation=cv2.INTER_LINEAR)
        resized_image_rgb = cv2.cvtColor(resized_image_bgr, cv2.COLOR_BGR2RGB)
        if resized_image_rgb.dtype != np.uint8:
            self._logger.warning(f"Тип даних зображення для SSD був {resized_image_rgb.dtype}, конвертується в uint8.")
            resized_image_rgb = resized_image_rgb.astype(np.uint8)
        input_tensor_nhwc_uint8 = np.expand_dims(resized_image_rgb, axis=0)
        self._logger.debug(
            f"SSD preprocessed tensor shape: {input_tensor_nhwc_uint8.shape}, dtype: {input_tensor_nhwc_uint8.dtype}")
        return input_tensor_nhwc_uint8

    def _apply_roi(self, image_bgr: np.ndarray, camera_type: str) -> tuple[
        np.ndarray, tuple[int, int, int, int] | None, tuple[int, int]]:
        # ... (без змін) ...
        roi_key = f"{camera_type}_camera_roi"
        if self.roi_config and roi_key in self.roi_config and self.roi_config[roi_key].get("enabled", False):
            roi = self.roi_config[roi_key]
            try:
                x1, y1, x2, y2 = int(roi["x1"]), int(roi["y1"]), int(roi["x2"]), int(roi["y2"])
            except KeyError as e:
                self._logger.error(f"Ключ {e} відсутній у конфігурації ROI для {roi_key}. ROI не застосовано.")
                return image_bgr, None, (0, 0)
            except ValueError as e:
                self._logger.error(f"Некоректне значення координат у ROI для {roi_key}: {e}. ROI не застосовано.")
                return image_bgr, None, (0, 0)

            h, w = image_bgr.shape[:2]
            x1 = max(0, min(x1, w - 1))
            y1 = max(0, min(y1, h - 1))
            x2 = max(0, min(x2, w))
            y2 = max(0, min(y2, h))

            if x1 >= x2 or y1 >= y2:
                self._logger.warning(
                    f"ROI для {camera_type} має некоректні або нульові розміри ({x1},{y1},{x2},{y2}). ROI не застосовано.")
                return image_bgr, None, (0, 0)
            # Використовуємо crop_image з image_utils
            cropped_image = crop_image(image_bgr, (x1, y1, x2, y2))
            if cropped_image is not None:
                self._logger.info(
                    f"Застосовано ROI [{x1},{y1},{x2},{y2}] для камери '{camera_type}'. Розмір обрізаного: {cropped_image.shape}")
                return cropped_image, (x1, y1, x2, y2), (x1, y1)
            else:  # crop_image повернув None
                self._logger.warning(f"crop_image повернув None для ROI [{x1},{y1},{x2},{y2}]. ROI не застосовано.")
                return image_bgr, None, (0, 0)

        self._logger.debug(f"ROI для камери '{camera_type}' не застосовано. Обробка повного кадру.")
        return image_bgr, None, (0, 0)

    def detect_vehicle_in_frame(self, image_bgr: np.ndarray, camera_type: str = "entry", save_debug_image: bool = False,
                                save_path_prefix: str = "debug_cv", timestamp: str = "") -> list | None:
        # ... (без змін, але тепер _apply_roi використовує crop_image з image_utils) ...
        if not self.vehicle_session:
            self._logger.error("Модель детекції автомобілів (vehicle_session) не завантажена.")
            return None
        if not self.vehicle_input_name or not self.vehicle_output_names_map.get("num"):
            self._logger.error("Імена вхідних/вихідних тензорів для моделі автомобілів не визначені або неповні.")
            return None

        img_to_process_roi, roi_abs_coords, (roi_offset_x, roi_offset_y) = self._apply_roi(image_bgr, camera_type)

        if img_to_process_roi is None or img_to_process_roi.size == 0:
            self._logger.warning("Зображення після ROI порожнє або None для детекції автомобілів.")
            return []

        input_tensor = self._preprocess_image_mobilenet_ssd(img_to_process_roi, self.vehicle_model_input_size)

        detections_on_original = []
        try:
            all_model_outputs = self.vehicle_session.run(None, {self.vehicle_input_name: input_tensor})
            output_names_from_session = [output.name for output in self.vehicle_session.get_outputs()]
            output_dict = {name: data for name, data in zip(output_names_from_session, all_model_outputs)}

            num_detections_val = output_dict.get(self.vehicle_output_names_map["num"])
            detection_boxes_val = output_dict.get(self.vehicle_output_names_map["boxes"])
            detection_classes_val = output_dict.get(self.vehicle_output_names_map["classes"])
            detection_scores_val = output_dict.get(self.vehicle_output_names_map["scores"])

            if num_detections_val is None or detection_boxes_val is None or \
                    detection_classes_val is None or detection_scores_val is None:
                self._logger.error(f"Не вдалося отримати всі необхідні виходи з моделі MobileNet SSD. "
                                   f"Очікувані ключі: {self.vehicle_output_names_map}. "
                                   f"Наявні в output_dict: {list(output_dict.keys())}")
                return None

            num_detections = int(num_detections_val[0])
            all_boxes = detection_boxes_val[0]
            all_classes = detection_classes_val[0].astype(np.int32)
            all_scores = detection_scores_val[0]

            proc_h, proc_w = img_to_process_roi.shape[:2]
            temp_detections_on_roi = []

            for i in range(min(num_detections, all_scores.shape[0])):
                score = all_scores[i]
                if score < self.vehicle_confidence_thresh: continue
                class_id = all_classes[i]
                if class_id not in TARGET_VEHICLE_CLASS_IDS: continue
                class_name = COCO_LABEL_MAP.get(class_id, f"class_{class_id}")

                box_roi_normalized = all_boxes[i]
                ymin, xmin, ymax, xmax = box_roi_normalized
                x1_roi_rel = max(0, int(xmin * proc_w))
                y1_roi_rel = max(0, int(ymin * proc_h))
                x2_roi_rel = min(proc_w - 1, int(xmax * proc_w))
                y2_roi_rel = min(proc_h - 1, int(ymax * proc_h))
                if x1_roi_rel >= x2_roi_rel or y1_roi_rel >= y2_roi_rel: continue
                temp_detections_on_roi.append((x1_roi_rel, y1_roi_rel, x2_roi_rel, y2_roi_rel, score, class_name))

                x1_orig, y1_orig = x1_roi_rel + roi_offset_x, y1_roi_rel + roi_offset_y
                x2_orig, y2_orig = x2_roi_rel + roi_offset_x, y2_roi_rel + roi_offset_y
                detections_on_original.append((x1_orig, y1_orig, x2_orig, y2_orig, score, class_name))

            if save_debug_image and img_to_process_roi is not None and img_to_process_roi.size > 0:
                img_with_vehicle_detections = img_to_process_roi.copy()
                for x1, y1, x2, y2, scr, cls_name in temp_detections_on_roi:
                    # Використовуємо draw_bounding_box з image_utils
                    draw_bounding_box(img_with_vehicle_detections, (x1, y1, x2, y2), cls_name, scr, color=(0, 255, 0))

                vehicle_det_filename_base = f"{timestamp}_1a_vehicle_detections_on_"
                vehicle_det_filename_suffix = "roi.jpg" if roi_abs_coords else "processed.jpg"
                save_image(img_with_vehicle_detections, save_path_prefix,
                           vehicle_det_filename_base + vehicle_det_filename_suffix)

            self._logger.info(
                f"Реальна детекція: Виявлено {len(detections_on_original)} цільових транспортних засобів на камері '{camera_type}'.")
            return detections_on_original
        except Exception as e:
            self._logger.error(f"Помилка під час реальної детекції автомобілів: {e}", exc_info=True)
            return None

    def _parse_license_plate_output(self, raw_lp_output: np.ndarray,
                                    original_lp_input_shape: tuple,
                                    scale_to_lp_input: float,
                                    pad_xy_for_lp_input: tuple) -> list:
        # ... (без змін) ...
        detections = []
        if raw_lp_output is None or raw_lp_output.ndim != 3 or raw_lp_output.shape[0] != 1 or raw_lp_output.shape[
            2] != 6:
            self._logger.warning(
                f"Неочікуваний формат виходу моделі НЗ: {raw_lp_output.shape if raw_lp_output is not None else 'None'}")
            return []
        predictions = raw_lp_output[0]
        lp_input_h, lp_input_w = self.plate_model_input_size[1], self.plate_model_input_size[0]
        orig_img_h, orig_img_w = original_lp_input_shape[:2]
        pad_x, pad_y = pad_xy_for_lp_input
        for pred in predictions:
            x1_norm, y1_norm, x2_norm, y2_norm = pred[:4]
            score = float(pred[4])
            cls_id = int(pred[5])
            if score < self.plate_confidence_thresh: continue
            x1_padded, y1_padded = x1_norm * lp_input_w, y1_norm * lp_input_h
            x2_padded, y2_padded = x2_norm * lp_input_w, y2_norm * lp_input_h
            x1_resized, y1_resized = x1_padded - pad_x, y1_padded - pad_y
            x2_resized, y2_resized = x2_padded - pad_x, y2_padded - pad_y
            x1_orig, y1_orig = int(x1_resized / scale_to_lp_input), int(y1_resized / scale_to_lp_input)
            x2_orig, y2_orig = int(x2_resized / scale_to_lp_input), int(y2_resized / scale_to_lp_input)
            x1_orig, y1_orig = max(0, x1_orig), max(0, y1_orig)
            x2_orig, y2_orig = min(orig_img_w, x2_orig), min(orig_img_h, y2_orig)
            if x1_orig >= x2_orig or y1_orig >= y2_orig: continue
            detections.append((x1_orig, y1_orig, x2_orig, y2_orig, score, cls_id))
        return detections

    def detect_license_plate(self, vehicle_image_bgr: np.ndarray, save_debug_image: bool = False,
                             save_path_prefix: str = "debug_cv", timestamp: str = "") -> tuple | None:
        # ... (без змін, крім використання save_image з image_utils) ...
        if not self.plate_session:
            self._logger.error("Модель детекції номерних знаків не завантажена.")
            return None
        if vehicle_image_bgr is None or vehicle_image_bgr.size == 0:
            self._logger.warning("Зображення автомобіля для детекції НЗ порожнє.")
            return None
        input_tensor, scale, pad_xy = self._preprocess_image_yolo(vehicle_image_bgr, self.plate_model_input_size)
        try:
            raw_outputs = self.plate_session.run(self.plate_output_names, {self.plate_input_name: input_tensor})[0]
            detections = self._parse_license_plate_output(raw_outputs, vehicle_image_bgr.shape, scale, pad_xy)

            if save_debug_image:
                img_with_plate_detections = vehicle_image_bgr.copy()
                for x1, y1, x2, y2, score, cls_id in detections:
                    draw_bounding_box(img_with_plate_detections, (x1, y1, x2, y2), f"LP (ID:{cls_id})", score,
                                      color=(255, 0, 0))
                save_image(img_with_plate_detections, save_path_prefix,
                           f"{timestamp}_1b_plate_detections_on_vehicle.jpg")

            if detections:
                plate_detections = [d for d in detections if d[5] == LICENSE_PLATE_CLASS_ID]
                if not plate_detections:
                    self._logger.info(
                        f"Номерний знак не виявлено (class_id != {LICENSE_PLATE_CLASS_ID} або не знайдено).")
                    return None
                best_plate = max(plate_detections, key=lambda item: item[4])
                self._logger.info(f"Номерний знак виявлено з впевненістю {best_plate[4]:.2f}")
                return best_plate
            else:
                self._logger.info("Номерний знак не виявлено на зображенні автомобіля (немає детекцій після парсингу).")
                return None
        except Exception as e:
            self._logger.error(f"Помилка під час детекції номерного знаку: {e}", exc_info=True)
            return None

    # _postprocess_yolo_detections - залишаємо для потенційного використання з іншими YOLO ONNX моделями,
    # але для OCR зараз використовується Ultralytics YOLO().predict()
    def _postprocess_yolo_detections(self, outputs: list, original_image_shape: tuple,
                                     scale: float, pad_xy: tuple, conf_thresh: float, nms_thresh: float,
                                     is_ocr: bool = False) -> list:
        if is_ocr and self.ocr_model_ultralytics:
            self._logger.debug("_postprocess_yolo_detections пропущено для OCR, використовується Ultralytics YOLO.")
            return []
            # ... (решта коду _postprocess_yolo_detections з 1D NMS, як було раніше)
        self._logger.debug(f"_postprocess_yolo_detections викликано (is_ocr={is_ocr}).")
        detections = []
        if not outputs or outputs[0] is None:
            self._logger.warning(f"Порожній або None вихід з моделі {'OCR' if is_ocr else 'Plate'}.")
            return []

        predictions = outputs[0][0]
        all_candidates = []
        if is_ocr:
            input_w_model, input_h_model = getattr(self, 'ocr_model_input_size_onnxruntime', (320, 320))
            current_conf_thresh = self.ocr_confidence_thresh
            current_nms_thresh = self.ocr_nms_thresh
        else:
            input_w_model, input_h_model = self.plate_model_input_size
            current_conf_thresh = self.plate_confidence_thresh
            current_nms_thresh = self.plate_nms_thresh
        for pred_idx in range(predictions.shape[0]):
            pred = predictions[pred_idx]
            object_confidence = pred[4]
            if is_ocr and object_confidence > 0.01:
                temp_class_id_log = int(pred[5])
                temp_cx_log, _, temp_w_norm_log, _ = pred[:4]
                temp_x_center_padded_log = temp_cx_log * input_w_model
                temp_w_padded_log = temp_w_norm_log * input_w_model
                self._logger.debug(
                    f"OCR Raw Candidate (on padded {input_w_model}x{input_h_model} input, via _postprocess_yolo_detections): "
                    f"cx_p={temp_x_center_padded_log:.1f}, w_p={temp_w_padded_log:.1f}, "
                    f"score={object_confidence:.2f}, class_id={temp_class_id_log}"
                )
            if object_confidence < current_conf_thresh: continue
            class_id = int(pred[5]);
            score = object_confidence
            cx, cy, w_norm, h_norm = pred[:4]
            x_center_padded, y_center_padded = cx * input_w_model, cy * input_h_model
            w_padded, h_padded = w_norm * input_w_model, h_norm * input_h_model
            if w_padded < 1 or h_padded < 1: continue
            x1_padded, y1_padded = int(x_center_padded - w_padded / 2), int(y_center_padded - h_padded / 2)
            x2_padded = int(x_center_padded + w_padded / 2)
            all_candidates.append({"x_interval_padded": (x1_padded, x2_padded),
                                   "full_box_padded": [x1_padded, y1_padded, int(w_padded), int(h_padded)],
                                   "score": float(score), "class_id": class_id})
        if not all_candidates: return []
        if is_ocr:
            all_candidates.sort(key=lambda c: c["score"], reverse=True)
            kept_ocr_candidates = []
            for cand in all_candidates:
                is_suppressed = False
                for kept_cand in kept_ocr_candidates:
                    if self._iou1d(cand["x_interval_padded"], kept_cand["x_interval_padded"]) > current_nms_thresh:
                        is_suppressed = True;
                        break
                if not is_suppressed: kept_ocr_candidates.append(cand)
            processed_candidates = kept_ocr_candidates
        else:
            boxes_for_2d_nms = [cand["full_box_padded"] for cand in all_candidates]
            scores_for_2d_nms = [cand["score"] for cand in all_candidates]
            raw_indices = cv2.dnn.NMSBoxes(boxes_for_2d_nms, scores_for_2d_nms,
                                           score_threshold=current_conf_thresh, nms_threshold=current_nms_thresh)
            processed_candidates = [all_candidates[i] for i in raw_indices.flatten()] if len(raw_indices) > 0 else []
        img_h_orig, img_w_orig = original_image_shape[:2];
        pad_x, pad_y = pad_xy
        for selected_cand in processed_candidates:
            x1_p, y1_p, w_p, h_p = selected_cand["full_box_padded"]
            x1_r, y1_r = x1_p - pad_x, y1_p - pad_y
            x1, y1 = int(x1_r / scale), int(y1_r / scale)
            x2, y2 = int((x1_r + w_p) / scale), int((y1_r + h_p) / scale)
            x1, y1, x2, y2 = max(0, x1), max(0, y1), min(img_w_orig, x2), min(img_h_orig, y2)
            if x1 >= x2 or y1 >= y2: continue
            final_cls_id, scr_val = selected_cand["class_id"], selected_cand["score"]
            lbl = CHAR_LIST[final_cls_id] if is_ocr and 0 <= final_cls_id < len(CHAR_LIST) else \
                (final_cls_id if not is_ocr else (self._logger.warning(f"OCR: Invalid class_id {final_cls_id}"), None))
            if lbl is not None: detections.append((x1, y1, x2, y2, scr_val, lbl))
        return detections

    def recognize_plate_characters(self, plate_image_bgr: np.ndarray, save_debug_image: bool = False,
                                   save_path_prefix: str = "debug_cv", timestamp: str = "") -> str | None:
        # ... (код з попереднього виправлення, використовує self.ocr_model_ultralytics,
        #      але тепер використовує image_utils.save_image для збереження ..._INPUT_TO_OCR.jpg
        #      та image_utils.draw_bounding_box для візуалізації ..._ocr_detections_on_plate_YOLO.jpg ) ...
        if not self.ocr_model_ultralytics:
            self._logger.error("Модель OCR (Ultralytics) не завантажена або бібліотека 'ultralytics' недоступна.")
            return None

        if plate_image_bgr is None or plate_image_bgr.size == 0:
            self._logger.warning("Зображення номерного знаку для OCR порожнє.")
            return None

        if save_debug_image:
            save_image(plate_image_bgr, save_path_prefix, f"{timestamp}_2d_INPUT_TO_OCR.jpg")

        char_detections_for_vis = []
        recognized_text = ""
        try:
            results = self.ocr_model_ultralytics.predict(
                source=plate_image_bgr,
                conf=self.ocr_confidence_thresh,
                iou=self.ocr_nms_thresh,
                imgsz=self.ocr_model_ultralytics_input_size,
                verbose=False,
            )

            if results and results[0].boxes:
                boxes = results[0].boxes
                orig_h, orig_w = plate_image_bgr.shape[:2]
                temp_chars_for_sorting = []

                for i in range(len(boxes)):
                    box_data = boxes[i]
                    class_id = int(box_data.cls[0])

                    if 0 <= class_id < len(CHAR_LIST):
                        char_str = CHAR_LIST[class_id]
                        score = float(box_data.conf[0])
                        x1, y1, x2, y2 = map(int, box_data.xyxy[0].cpu().numpy())
                        x1, y1 = max(0, x1), max(0, y1)
                        x2, y2 = min(orig_w, x2), min(orig_h, y2)
                        if x1 >= x2 or y1 >= y2: continue

                        char_detections_for_vis.append((x1, y1, x2, y2, score, char_str))
                        temp_chars_for_sorting.append({'x': x1, 'char': char_str})
                    else:
                        self._logger.warning(f"OCR (YOLO): Отримано некоректний class_id ({class_id}).")

                temp_chars_for_sorting.sort(key=lambda item: item['x'])
                recognized_text = "".join([item['char'] for item in temp_chars_for_sorting])
            else:
                self._logger.info("OCR (YOLO): Символи на номерному знаку не виявлено.")

            if save_debug_image:
                img_with_ocr_detections = plate_image_bgr.copy()
                if char_detections_for_vis:
                    for x1_vis, y1_vis, x2_vis, y2_vis, score_vis, char_str_vis in char_detections_for_vis:
                        # Використовуємо draw_bounding_box з image_utils
                        draw_bounding_box(img_with_ocr_detections, (x1_vis, y1_vis, x2_vis, y2_vis),
                                          label=char_str_vis, score=score_vis, color=(0, 255, 255), thickness=1,
                                          text_color=(0, 0, 0), bg_text_color=(0, 255, 255))  # Жовтий фон для тексту
                else:
                    draw_text_with_background(img_with_ocr_detections, "No Chars Detected (YOLO)", (10, 30),
                                              bg_color=(0, 0, 150))
                save_image(img_with_ocr_detections, save_path_prefix,
                           f"{timestamp}_2b_ocr_detections_on_plate_YOLO.jpg")

            if not recognized_text:
                self._logger.info("OCR (YOLO): Текст не розпізнано.")
                return None

            self._logger.info(
                f"OCR (YOLO): Розпізнано текст: '{recognized_text}' з {len(char_detections_for_vis)} символів.")
            return recognized_text

        except Exception as e:
            self._logger.error(f"Помилка під час розпізнавання символів через Ultralytics YOLO: {e}", exc_info=True)
            return None

    def get_plate_number_from_image(self, image_bgr: np.ndarray, camera_type: str = "entry",
                                    save_intermediate_steps: bool = False,
                                    save_path_prefix: str = "debug_cv") -> str | None:
        if image_bgr is None or image_bgr.size == 0:
            self._logger.warning("Вхідне зображення для get_plate_number_from_image порожнє або None.")
            return None

        timestamp = time.strftime('%Y%m%d_%H%M%S')
        if save_intermediate_steps:
            save_image(image_bgr, save_path_prefix, f"{timestamp}_0_orig_{camera_type}.jpg")

        img_after_roi, roi_abs_coords, _ = self._apply_roi(image_bgr, camera_type)
        if save_intermediate_steps and roi_abs_coords:
            if img_after_roi is not None and img_after_roi.size > 0:
                save_image(img_after_roi, save_path_prefix, f"{timestamp}_0a_roi_applied_{camera_type}.jpg")

        vehicles = self.detect_vehicle_in_frame(
            image_bgr, camera_type=camera_type,
            save_debug_image=save_intermediate_steps,
            save_path_prefix=save_path_prefix,
            timestamp=timestamp
        )

        if not vehicles:
            self._logger.info(f"Автомобілі не виявлено на камері '{camera_type}' (з get_plate_number_from_image).")
            return None

        best_vehicle_bbox = vehicles[0][:4]
        # Використовуємо crop_image з image_utils
        vehicle_image = crop_image(image_bgr, best_vehicle_bbox)

        if vehicle_image is None:
            self._logger.warning("Не вдалося обрізати зображення автомобіля (з get_plate_number_from_image).")
            return None
        if save_intermediate_steps:
            save_image(vehicle_image, save_path_prefix, f"{timestamp}_1c_vehicle_crop_final.jpg")

        plate_detection_result = self.detect_license_plate(
            vehicle_image,
            save_debug_image=save_intermediate_steps,
            save_path_prefix=save_path_prefix,
            timestamp=timestamp
        )
        if not plate_detection_result:
            return None

        plate_bbox_on_vehicle = plate_detection_result[:4]
        # Використовуємо crop_image з image_utils
        plate_image = crop_image(vehicle_image, plate_bbox_on_vehicle)

        if plate_image is None:
            self._logger.warning("Не вдалося обрізати зображення номерного знаку (з get_plate_number_from_image).")
            return None
        if save_intermediate_steps:
            save_image(plate_image, save_path_prefix, f"{timestamp}_2c_plate_crop_final.jpg")

        plate_text = self.recognize_plate_characters(
            plate_image,
            save_debug_image=save_intermediate_steps,
            save_path_prefix=save_path_prefix,
            timestamp=timestamp
        )

        if plate_text:
            self._logger.info(f"Успішно розпізнано номерний знак: '{plate_text}'")
            if save_intermediate_steps:
                final_img_display = image_bgr.copy()
                if roi_abs_coords:
                    draw_bounding_box(final_img_display, roi_abs_coords, "ROI", color=(0, 0, 255))

                draw_bounding_box(final_img_display, best_vehicle_bbox, vehicles[0][5], score=vehicles[0][4],
                                  color=(0, 255, 0))

                plate_x1_abs = best_vehicle_bbox[0] + plate_bbox_on_vehicle[0]
                plate_y1_abs = best_vehicle_bbox[1] + plate_bbox_on_vehicle[1]
                plate_x2_abs = best_vehicle_bbox[0] + plate_bbox_on_vehicle[2]
                plate_y2_abs = best_vehicle_bbox[1] + plate_bbox_on_vehicle[3]

                plate_full_abs_bbox = (plate_x1_abs, plate_y1_abs, plate_x2_abs, plate_y2_abs)
                draw_bounding_box(final_img_display, plate_full_abs_bbox, plate_text, score=plate_detection_result[4],
                                  color=(255, 0, 0))

                save_image(final_img_display, save_path_prefix, f"{timestamp}_3_final_all_detections.jpg")
            return plate_text
        else:
            return None


# --- Приклад використання (для тестування модуля окремо) ---
if __name__ == '__main__':
    # ... (без змін)
    if not ULTRALYTICS_AVAILABLE:
        logger.error("Бібліотека 'ultralytics' не встановлена. Тестування OCR через YOLO() неможливе.")

    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s')

    logger.info("Тестування модуля cv_processor.py...")

    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(current_dir)

    models_base_dir = os.path.join(project_root, "models")
    config_base_dir = os.path.join(project_root, "config")
    test_output_dir = os.path.join(project_root, "cv_test_output")

    os.makedirs(models_base_dir, exist_ok=True)
    os.makedirs(config_base_dir, exist_ok=True)
    os.makedirs(test_output_dir, exist_ok=True)

    for model_name_onnx in ["ssd_mobilenetv1.onnx", "license.onnx"]:
        model_file_path = os.path.join(models_base_dir, model_name_onnx)
        if not os.path.exists(model_file_path):
            logger.warning(f"Створюю фіктивний файл моделі: {model_file_path}")
            try:
                open(model_file_path, 'a').close()
            except OSError as e:
                logger.error(f"Не вдалося створити фіктивний файл {model_file_path}: {e}")

    TEST_MOBILENET_PATH = os.path.join(models_base_dir, "ssd_mobilenetv1.onnx")
    TEST_LICENSE_MODEL_PATH = os.path.join(models_base_dir, "license.onnx")
    TEST_OCR_MODEL_PATH = os.path.join(models_base_dir, "ocr.pt")

    if not os.path.exists(TEST_OCR_MODEL_PATH) and ULTRALYTICS_AVAILABLE:
        logger.error(
            f"Файл моделі OCR '{TEST_OCR_MODEL_PATH}' (.pt) не знайдено! Будь ласка, розмістіть його для тестування.")

    roi_example_path = os.path.join(config_base_dir, "roi_config_cv_test.json")
    with open(roi_example_path, 'w') as f_roi:
        json.dump({
            "entry_camera_roi": {"x1": 50, "y1": 50, "x2": 590, "y2": 430, "enabled": False},
            "exit_camera_roi": {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "enabled": False}
        }, f_roi, indent=2)
    logger.info(f"Приклад конфігурації ROI збережено в: {roi_example_path}")

    TEST_IMAGE_PATH = "test_output_dir/cv_processor_test_input_image.png"
    os.makedirs(os.path.dirname(TEST_IMAGE_PATH), exist_ok=True)

    test_image_bgr = None
    if os.path.exists(TEST_IMAGE_PATH) and os.path.isfile(TEST_IMAGE_PATH):
        logger.info(f"Завантаження тестового зображення з: {TEST_IMAGE_PATH}")
        test_image_bgr = cv2.imread(TEST_IMAGE_PATH)

    if test_image_bgr is None:
        logger.warning(f"Тестове зображення не знайдено/не завантажено з {TEST_IMAGE_PATH}. Створюю фіктивне.")
        test_image_bgr = np.zeros((480, 640, 3), dtype=np.uint8)
        cv2.putText(test_image_bgr, "DUMMY IMAGE - REPLACE ME", (10, 240), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 255),
                    2)
        cv2.rectangle(test_image_bgr, (100, 100), (540, 380), (0, 100, 0), -1)
        cv2.rectangle(test_image_bgr, (250, 250), (390, 290), (200, 200, 200), -1)
        cv2.putText(test_image_bgr, "AB1234CD", (255, 275), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 0), 1)
        try:
            cv2.imwrite(TEST_IMAGE_PATH, test_image_bgr)
            logger.info(
                f"Фіктивне тестове зображення збережено як: {TEST_IMAGE_PATH}. Будь ласка, замініть його реальним зображенням.")
        except Exception as e:
            logger.error(f"Не вдалося зберегти фіктивне зображення: {e}")

    if test_image_bgr is not None:
        try:
            logger.info("СПРОБА ІНІЦІАЛІЗАЦІЇ CVProcessor З МОДЕЛЯМИ.")

            cv_proc = CVProcessor(
                mobilenet_ssd_path=TEST_MOBILENET_PATH,
                license_model_path=TEST_LICENSE_MODEL_PATH,
                ocr_model_path=TEST_OCR_MODEL_PATH,
                roi_config_path=roi_example_path,
                vehicle_input_target_size=(300, 300),
                ocr_nms_thresh=0.35,
                ocr_input_target_size_for_ultralytics=320
            )

            logger.info("\n--- Тест: Повний цикл розпізнавання номера з візуалізацією ---")
            plate = cv_proc.get_plate_number_from_image(
                test_image_bgr.copy(),
                camera_type="entry",
                save_intermediate_steps=True,
                save_path_prefix=test_output_dir
            )
            if plate:
                logger.info(f"Фінальний розпізнаний номер: {plate}")
            else:
                logger.info("Номерний знак не розпізнано (повний цикл).")
        except Exception as e:
            logger.error(f"Помилка під час тестування CVProcessor: {e}", exc_info=True)
    else:
        logger.error("Не вдалося завантажити або створити тестове зображення. Тестування CVProcessor неможливе.")

    logger.info(
        f"Тестування модуля cv_processor.py завершено. Результати та візуалізації (якщо ввімкнено) збережено у: {test_output_dir}")