# core/vehicle_event_handler.py
import logging
import time
import threading
import os
import numpy as np
import cv2
from typing import Optional, Dict, Any

from utils.image_utils import save_image

# from .camera_manager import CameraController
# from .sensor_manager import SensorManager, UltrasonicSensor
# from .sheet_handler import SheetHandler
# from .cv_processor import CVProcessor
# from .gate_controller import GateController

logger = logging.getLogger(__name__)

# --- Конфігураційні константи (значення за замовчуванням) ---
DEFAULT_SHEETS_ANTIDUPLICATE_DELAY_S = 60
DEFAULT_REED_OPEN_TIMEOUT_S = 15  # Таймаут очікування відкриття герконом (сценарій VEH)
DEFAULT_REED_OPEN_RETRIES = 1  # Кількість повторів відкриття (сценарій VEH)
DEFAULT_PASSAGE_CONFIRMATION_TIMEOUT_S = 20  # Таймаут для wait_for_clear_after_pass
DEFAULT_ULTRASONIC_PASSAGE_THRESHOLD = 0.3  # Поріг УЗД для "в проїзді" / "проїзд вільний"
DEFAULT_AUTO_CLOSE_TIMER_DURATION_S = 4  # 4-секундний таймер на закриття (сценарій VEH)
DEFAULT_REED_CLOSE_TIMEOUT_S = 5  # Таймаут очікування закриття герконом (сценарій VEH)
DEFAULT_REED_CLOSE_RETRIES = 1  # Кількість повторів закриття (сценарій VEH)
DEFAULT_GATE_FINISH_CLOSING_DELAY_S = 10  # Фінальна затримка після закриття (сценарій VEH)
DEFAULT_POLL_INTERVAL_IDLE_S = 1.0
DEFAULT_POLL_INTERVAL_GATE_CLOSING_S = 0.3

CAPTURED_IMAGES_BASE_PATH = "captured_images"
ENTRY_IMAGES_PATH = os.path.join(CAPTURED_IMAGES_BASE_PATH, "entry")
EXIT_IMAGES_PATH = os.path.join(CAPTURED_IMAGES_BASE_PATH, "exit")
CV_DEBUG_SAVE_PATH = os.path.join(CAPTURED_IMAGES_BASE_PATH, "cv_debug")


class VehicleEventHandler:
    def __init__(self,
                 camera_entry,  # Очікується CameraController або None
                 camera_exit,  # Очікується CameraController або None
                 sensor_manager,
                 sheet_handler,
                 cv_processor,
                 gate_controller,
                 config: Optional[Dict[str, Any]] = None):

        self._logger = logging.getLogger(f"{__name__}.VehicleEventHandler")
        self.camera_entry = camera_entry
        self.camera_exit = camera_exit
        self.sensor_manager = sensor_manager
        self.sheet_handler = sheet_handler
        self.cv_processor = cv_processor
        self.gate_controller = gate_controller

        self.config = config if config else {}
        self.sheets_antiduplicate_delay_s = self.config.get(
            "sheets_antiduplicate_delay_s", DEFAULT_SHEETS_ANTIDUPLICATE_DELAY_S
        )
        self.reed_open_timeout_s = self.config.get("reed_open_timeout_s", DEFAULT_REED_OPEN_TIMEOUT_S)
        self.reed_open_retries = self.config.get("reed_open_retries", DEFAULT_REED_OPEN_RETRIES)
        self.passage_confirmation_timeout_s = self.config.get(
            "passage_confirmation_timeout_s", DEFAULT_PASSAGE_CONFIRMATION_TIMEOUT_S
        )
        self.ultrasonic_passage_threshold = self.config.get(  # Цей поріг для detect_object_in_passage
            "ultrasonic_passage_threshold", DEFAULT_ULTRASONIC_PASSAGE_THRESHOLD
        )
        # Пороги для wait_for_clear_after_pass (якщо вони потрібні, або беруться з UltrasonicSensor.DEFAULT_*)
        self.ultrasonic_clear_threshold_for_passage = self.config.get(
            "ultrasonic_clear_threshold_for_passage",
            getattr(
                self.sensor_manager.ultrasonic_sensor_entry if self.sensor_manager and self.sensor_manager.ultrasonic_sensor_entry else None,
                'DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR', 2.0)
        )
        self.ultrasonic_clear_confirmation_s = self.config.get(
            "ultrasonic_clear_confirmation_s",
            getattr(
                self.sensor_manager.ultrasonic_sensor_entry if self.sensor_manager and self.sensor_manager.ultrasonic_sensor_entry else None,
                'PASS_CONFIRMATION_TIME_S', 1.5)
        )

        self.auto_close_timer_duration_s = self.config.get(
            "auto_close_timer_duration_s", DEFAULT_AUTO_CLOSE_TIMER_DURATION_S
        )
        self.reed_close_timeout_s = self.config.get("reed_close_timeout_s", DEFAULT_REED_CLOSE_TIMEOUT_S)
        self.reed_close_retries = self.config.get("reed_close_retries", DEFAULT_REED_CLOSE_RETRIES)
        self.gate_finish_closing_delay_s = self.config.get(
            "gate_finish_closing_delay_s", DEFAULT_GATE_FINISH_CLOSING_DELAY_S
        )
        self.poll_interval_idle_s = self.config.get(
            "poll_interval_idle_s", DEFAULT_POLL_INTERVAL_IDLE_S
        )
        self.poll_interval_gate_closing_s = self.config.get(
            "poll_interval_gate_closing_s", DEFAULT_POLL_INTERVAL_GATE_CLOSING_S
        )

        self.is_running = False
        self.shutdown_event: Optional[threading.Event] = None

        self.recently_logged_plates: Dict[str, float] = {}
        self._plate_cache_lock = threading.Lock()

        os.makedirs(ENTRY_IMAGES_PATH, exist_ok=True)
        os.makedirs(EXIT_IMAGES_PATH, exist_ok=True)
        os.makedirs(os.path.join(CV_DEBUG_SAVE_PATH, "entry"), exist_ok=True)
        os.makedirs(os.path.join(CV_DEBUG_SAVE_PATH, "exit"), exist_ok=True)

        self._logger.info("VehicleEventHandler ініціалізовано.")
        self._logger.info(f"  Конфігурація VEH: {self.config}")

    def _is_duplicate_log(self, plate_number: str) -> bool:
        # ... (без змін) ...
        with self._plate_cache_lock:
            now = time.time()
            if plate_number in self.recently_logged_plates:
                last_log_time = self.recently_logged_plates[plate_number]
                if (now - last_log_time) < self.sheets_antiduplicate_delay_s:
                    self._logger.info(
                        f"Дублікатний запис для НЗ '{plate_number}'. Минуло менше {self.sheets_antiduplicate_delay_s} с.")
                    return True
            self.recently_logged_plates[plate_number] = now
            if len(self.recently_logged_plates) > 100:
                max_age = self.sheets_antiduplicate_delay_s * 10
                self.recently_logged_plates = {
                    p: t for p, t in self.recently_logged_plates.items() if (now - t) < max_age
                }
            return False

    def _attempt_open_gate_with_retry(self) -> bool:
        # ... (без змін від попередньої версії) ...
        if not self.gate_controller or not self.sensor_manager or \
                not hasattr(self.sensor_manager, 'reed_switch') or not self.sensor_manager.reed_switch:
            self._logger.error("Неможливо відкрити ворота: GateController або ReedSwitch не ініціалізовано.")
            return False
        for attempt in range(self.reed_open_retries + 1):
            self._logger.info(f"Спроба відкриття воріт #{attempt + 1}...")
            self.gate_controller.open_gate()
            start_wait = time.monotonic()
            opened_by_reed = False
            while time.monotonic() - start_wait < self.reed_open_timeout_s:
                if self.shutdown_event and self.shutdown_event.is_set(): return False
                reed_state = self.sensor_manager.reed_switch.are_gates_open
                if reed_state is True:
                    self._logger.info(f"Ворота успішно відкрито (підтверджено герконом на спробі #{attempt + 1}).")
                    opened_by_reed = True;
                    break
                elif reed_state is None:
                    self._logger.warning("Не вдалося отримати стан геркона під час очікування відкриття.")
                time.sleep(0.2)
            if opened_by_reed: return True
            self._logger.warning(
                f"Ворота не відкрилися (геркон) протягом {self.reed_open_timeout_s}с після спроби #{attempt + 1}.")
            if attempt < self.reed_open_retries:
                self._logger.info("Повторна спроба відкриття через 1 секунду..."); time.sleep(1)
            else:
                self._logger.error(
                    f"Не вдалося відкрити ворота після {self.reed_open_retries + 1} спроб."); return False
        return False

    def _wait_for_vehicle_passage_after_open(self, gate_side_name: str) -> bool:
        """
        Очікує проїзду авто, використовуючи метод wait_for_clear_after_pass з UltrasonicSensor,
        який відповідає логіці "був об'єкт, потім зона стала вільною".
        Перевіряє, чи ворота відкриті перед початком.
        """
        passage_sensor = self.sensor_manager.ultrasonic_sensor_entry
        if gate_side_name == "exit":
            passage_sensor = getattr(self.sensor_manager, 'ultrasonic_sensor_exit',
                                     self.sensor_manager.ultrasonic_sensor_entry)

        if not passage_sensor:
            self._logger.error(f"УЗД для контролю проїзду ({gate_side_name}) недоступний.")
            return False

        self._logger.info(f"Очікування проїзду автомобіля через УЗД ({gate_side_name}).")

        if not self.sensor_manager.reed_switch.are_gates_open:
            self._logger.warning(f"Ворота не відкриті (геркон). Скасування очікування проїзду УЗД ({gate_side_name}).")
            return False
        self._logger.info(f"Ворота відкриті (геркон). УЗД ({gate_side_name}) активний для детекції проїзду.")

        # Використовуємо wait_for_clear_after_pass з відповідними порогами
        # Цей метод з sensors_manager.UltrasonicSensor вже реалізує логіку "був близько -> став далеко на певний час"
        if passage_sensor.wait_for_clear_after_pass(
                threshold_clear_m=self.ultrasonic_clear_threshold_for_passage,  # Наприклад, 2.0м
                confirmation_s=self.ultrasonic_clear_confirmation_s,  # Наприклад, 1.5с
                timeout=self.passage_confirmation_timeout_s  # Загальний таймаут на всю операцію
        ):
            self._logger.info(f"Автомобіль повністю проїхав зону УЗД ({gate_side_name}).")
            return True
        else:
            self._logger.warning(f"Автомобіль не підтвердив проїзд через УЗД ({gate_side_name}) або таймаут.")
            # Додаткова перевірка, чи є зараз перешкода
            if passage_sensor.detect_object_in_passage(self.ultrasonic_passage_threshold):
                self._logger.warning(
                    f"УВАГА: Автомобіль або перешкода все ще в зоні УЗД '{gate_side_name}' (<{self.ultrasonic_passage_threshold}м)!")
            return False

    def _manage_auto_close_with_obstruction_check(self) -> bool:
        # ... (без змін від попередньої версії) ...
        if not (self.gate_controller and self.sensor_manager and self.sensor_manager.ultrasonic_sensor_entry):
            self._logger.error("Неможливо керувати автозакриттям: компоненти не ініціалізовані.")
            return False
        passage_check_sensor = self.sensor_manager.ultrasonic_sensor_entry
        start_time = time.monotonic()
        self.gate_controller.start_auto_close_timer(timeout_s=self.auto_close_timer_duration_s)
        timer_end_time = time.monotonic() + self.auto_close_timer_duration_s
        while time.monotonic() < timer_end_time:
            if self.shutdown_event and self.shutdown_event.is_set():
                self.gate_controller.cancel_auto_close_timer();
                return False
            if passage_check_sensor.detect_object_in_passage(self.ultrasonic_passage_threshold):  # Поріг 0.3м
                self._logger.info(
                    f"Перешкода виявлена УЗД під час {self.auto_close_timer_duration_s}с таймера! Зупинка та очікування.")
                self.gate_controller.cancel_auto_close_timer()
                obstacle_cleared_time = time.monotonic()
                while passage_check_sensor.detect_object_in_passage(self.ultrasonic_passage_threshold):
                    if self.shutdown_event and self.shutdown_event.is_set(): return False
                    if time.monotonic() - obstacle_cleared_time > 60:
                        self._logger.warning("Перешкода не зникла з зони УЗД після 60с. Автозакриття скасовано.")
                        return False
                    time.sleep(0.1)
                self._logger.info("Перешкода зникла. Перезапуск таймера автозакриття.")
                self.gate_controller.start_auto_close_timer(timeout_s=self.auto_close_timer_duration_s)
                timer_end_time = time.monotonic() + self.auto_close_timer_duration_s
            time.sleep(0.05)
        self._logger.info(f"{self.auto_close_timer_duration_s}с період моніторингу завершився.")
        return True

    def _attempt_close_gate_with_retry(self) -> bool:
        # ... (без змін від попередньої версії) ...
        if not self.gate_controller or not self.sensor_manager or not self.sensor_manager.reed_switch:
            self._logger.error("Неможливо закрити ворота: GateController або ReedSwitch не ініціалізовано.")
            return False
        for attempt in range(self.reed_close_retries + 1):
            self._logger.info(f"Спроба закриття воріт #{attempt + 1}...")
            if not self.gate_controller.close_gate():
                self._logger.warning(f"Команда close_gate() не виконана на спробі #{attempt + 1}.")
                if self.gate_controller.get_current_gate_state() == "CLOSED": return True
                if self.gate_controller.get_current_gate_state() == "OBSTRUCTED" and attempt >= self.reed_close_retries: return False
                if attempt < self.reed_close_retries:
                    time.sleep(1); continue
                else:
                    return False
            start_wait = time.monotonic()
            closed_by_reed = False
            while time.monotonic() - start_wait < self.reed_close_timeout_s:
                if self.shutdown_event and self.shutdown_event.is_set(): return False
                reed_state = self.sensor_manager.reed_switch.are_gates_closed
                if reed_state is True:
                    self._logger.info(f"Ворота успішно закрито (підтверджено герконом на спробі #{attempt + 1}).")
                    closed_by_reed = True;
                    break
                elif reed_state is None:
                    self._logger.warning("Не вдалося отримати стан геркона під час очікування закриття.")
                time.sleep(0.2)
            if closed_by_reed: return True
            self._logger.warning(
                f"Ворота не закрилися (геркон) протягом {self.reed_close_timeout_s}с після спроби #{attempt + 1}.")
            if attempt < self.reed_close_retries:
                self._logger.info("Повторна спроба закриття через 1 секунду..."); time.sleep(1)
            else:
                self._logger.error(
                    f"Не вдалося закрити ворота після {self.reed_close_retries + 1} спроб."); return False
        return False

    def entry_scenario_loop(self):
        # ... (без змін від попередньої версії, використовує CV-only тригер) ...
        self._logger.info("Запуск циклу обробки В'ЇЗДУ...")
        while self.is_running and (self.shutdown_event is None or not self.shutdown_event.is_set()):
            current_poll_interval = self.poll_interval_idle_s
            if hasattr(self.gate_controller, '_auto_close_timer') and \
                    self.gate_controller._auto_close_timer is not None and \
                    self.gate_controller._auto_close_timer.is_alive():
                current_poll_interval = self.poll_interval_gate_closing_s
                if self._handle_gate_closing_interruption("entry"):
                    self._logger.info("В'їзд: Закриття перервано новим авто. Повторний цикл детекції.")
                    time.sleep(0.1);
                    continue
            self._logger.debug("В'їзд: Очікування автомобіля (тільки CV)...")
            vehicle_detected_by_cv, initial_detection_frame = False, None
            if self.camera_entry and self.camera_entry.is_initialized_successfully and self.cv_processor:
                frame_for_detection = self.camera_entry.capture_array()
                if frame_for_detection is not None:
                    vehicle_detections = self.cv_processor.detect_vehicle_in_frame(frame_for_detection,
                                                                                   camera_type="entry")
                    if vehicle_detections:
                        self._logger.info("В'їзд: CV зафіксував автомобіль.")
                        vehicle_detected_by_cv, initial_detection_frame = True, frame_for_detection
            if vehicle_detected_by_cv:
                self._logger.info("В'їзд: Автомобіль виявлено через CV. Початок обробки.")
                if initial_detection_frame is None:
                    self._logger.error("В'їзд: Помилка - CV виявив авто, але кадр не збережено.")
                    time.sleep(current_poll_interval);
                    continue
                timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                plate_text = self.cv_processor.get_plate_number_from_image(
                    initial_detection_frame, camera_type="entry", save_intermediate_steps=True,
                    save_path_prefix=os.path.join(CV_DEBUG_SAVE_PATH, "entry")
                )
                if plate_text:
                    if not self._is_duplicate_log(plate_text):
                        if self.sheet_handler.find_vehicle_and_update_entry_time(plate_text):
                            self._logger.info(f"В'їзд: Авто '{plate_text}' АВТОРИЗОВАНО.")
                            if self._attempt_open_gate_with_retry():
                                if self._wait_for_vehicle_passage_after_open("в'їзду"):
                                    if self._manage_auto_close_with_obstruction_check():
                                        if self._attempt_close_gate_with_retry():
                                            self._logger.info(
                                                f"В'їзд: Цикл для '{plate_text}' завершено, ворота закрито. Очікування {self.gate_finish_closing_delay_s}с.")
                                            time.sleep(self.gate_finish_closing_delay_s)
                                        else:
                                            self._logger.error("В'їзд: Не вдалося підтвердити закриття воріт.")
                                else:
                                    self._logger.warning(
                                        "В'їзд: Авто не підтвердило проїзд. Ворота можуть бути відкриті.")
                            else:
                                self._logger.error("В'їзд: Не вдалося відкрити ворота для авторизованого авто.")
                        else:
                            self._logger.info(f"В'їзд: Авто '{plate_text}' НЕ АВТОРИЗОВАНО.")
                            self.sheet_handler.add_unauthorized_attempt(plate_text)
                else:
                    self._logger.warning("В'їзд: Номерний знак не розпізнано.")
                    save_image(initial_detection_frame, ENTRY_IMAGES_PATH, f"entry_ocr_failed_{timestamp_str}.jpg")
            time.sleep(current_poll_interval)
        self._logger.info("Цикл обробки В'ЇЗДУ завершено.")

    def exit_scenario_loop(self):
        # ... (Аналогічно, без змін від попередньої версії, використовує CV-only тригер) ...
        self._logger.info("Запуск циклу обробки ВИЇЗДУ...")
        while self.is_running and (self.shutdown_event is None or not self.shutdown_event.is_set()):
            if hasattr(self.gate_controller, '_auto_close_timer') and \
                    self.gate_controller._auto_close_timer is not None and \
                    self.gate_controller._auto_close_timer.is_alive():
                current_poll_interval = self.poll_interval_gate_closing_s
                if self._handle_gate_closing_interruption("exit"):
                    self._logger.info("Виїзд: Закриття перервано новим авто. Повторний цикл детекції.")
                    time.sleep(0.1);
                    continue
            else:
                current_poll_interval = self.poll_interval_idle_s
            self._logger.debug("Виїзд: Очікування автомобіля (тільки CV)...")
            vehicle_detected_by_cv_exit, initial_detection_frame_exit = False, None
            if self.camera_exit and self.camera_exit.is_initialized_successfully and self.cv_processor:
                frame_for_detection = self.camera_exit.capture_array()
                if frame_for_detection is not None:
                    vehicle_detections = self.cv_processor.detect_vehicle_in_frame(frame_for_detection,
                                                                                   camera_type="exit")
                    if vehicle_detections:
                        self._logger.info("Виїзд: CV зафіксував автомобіль.")
                        vehicle_detected_by_cv_exit, initial_detection_frame_exit = True, frame_for_detection
            if vehicle_detected_by_cv_exit:
                self._logger.info("Виїзд: Автомобіль виявлено через CV. Негайне відкриття воріт.")
                if self._attempt_open_gate_with_retry():
                    if initial_detection_frame_exit is None and self.camera_exit and self.camera_exit.is_initialized_successfully:
                        initial_detection_frame_exit = self.camera_exit.capture_array()
                    if initial_detection_frame_exit is not None:
                        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                        plate_text = self.cv_processor.get_plate_number_from_image(
                            initial_detection_frame_exit, camera_type="exit", save_intermediate_steps=True,
                            save_path_prefix=os.path.join(CV_DEBUG_SAVE_PATH, "exit")
                        )
                        if plate_text:
                            if not self._is_duplicate_log(plate_text):
                                self.sheet_handler.log_vehicle_exit(plate_text)
                        else:
                            self._logger.warning("Виїзд: Номерний знак не розпізнано, але виїзд дозволено.")
                            save_image(initial_detection_frame_exit, EXIT_IMAGES_PATH,
                                       f"exit_ocr_failed_{timestamp_str}.jpg")
                    else:
                        self._logger.error("Виїзд: Не вдалося отримати кадр для логування НЗ.")
                    if self._wait_for_vehicle_passage_after_open("виїзду"):
                        if self._manage_auto_close_with_obstruction_check():
                            if self._attempt_close_gate_with_retry():
                                self._logger.info(
                                    f"Виїзд: Цикл завершено, ворота закрито. Очікування {self.gate_finish_closing_delay_s}с.")
                                time.sleep(self.gate_finish_closing_delay_s)
                            else:
                                self._logger.error("Виїзд: Не вдалося підтвердити закриття воріт.")
                    else:
                        self._logger.warning("Виїзд: Авто не підтвердило проїзд. Ворота можуть бути відкриті.")
                else:
                    self._logger.error("Виїзд: Не вдалося відкрити ворота для авто на виїзд.")
            time.sleep(current_poll_interval)
        self._logger.info("Цикл обробки ВИЇЗДУ завершено.")

    def start(self, shutdown_event_main: threading.Event):
        # ... (без змін) ...
        if self.is_running: self._logger.warning("VehicleEventHandler вже запущено."); return
        self._logger.info("Запуск VehicleEventHandler...");
        self.is_running = True
        self.shutdown_event = shutdown_event_main
        self.entry_thread = threading.Thread(target=self.entry_scenario_loop, name="EntryScenarioThread")
        self.exit_thread = threading.Thread(target=self.exit_scenario_loop, name="ExitScenarioThread")
        self.entry_thread.daemon = True;
        self.exit_thread.daemon = True
        self.entry_thread.start();
        self.exit_thread.start()
        self._logger.info("Потоки обробки в'їзду та виїзду запущено.")

    def stop(self):
        # ... (без змін) ...
        if not self.is_running:
            if self.shutdown_event and self.shutdown_event.is_set():
                self._logger.info("VEH вже отримав сигнал на зупинку.")
            else:
                self._logger.info("VEH вже зупинено або не було запущено."); return
        self._logger.info("Зупинка VehicleEventHandler...");
        self.is_running = False
        self._logger.info("VehicleEventHandler: прапор is_running встановлено в False.")