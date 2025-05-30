# core/vehicle_event_handler.py
import logging
import time
import threading
import os
import numpy as np
import cv2
from typing import Optional, Dict, Any

# Імпорт з ваших модулів
from utils.image_utils import save_image

# Для type hinting (фактичні екземпляри будуть передані в __init__)
# from .camera_manager import CameraController
# from .sensor_manager import SensorManager, UltrasonicSensor
# from .sheet_handler import SheetHandler
# from .cv_processor import CVProcessor
# from .gate_controller import GateController

logger = logging.getLogger(__name__)

# --- Конфігураційні константи (значення за замовчуванням) ---
DEFAULT_SHEETS_ANTIDUPLICATE_DELAY_S = 60
DEFAULT_REED_OPEN_TIMEOUT_S = 15
DEFAULT_REED_OPEN_RETRIES = 1
DEFAULT_PASSAGE_CONFIRMATION_TIMEOUT_S = 20
DEFAULT_ULTRASONIC_PASSAGE_THRESHOLD = 0.3
DEFAULT_AUTO_CLOSE_TIMER_DURATION_S = 4
DEFAULT_REED_CLOSE_TIMEOUT_S = 5
DEFAULT_REED_CLOSE_RETRIES = 1
DEFAULT_GATE_FINISH_CLOSING_DELAY_S = 10
DEFAULT_POLL_INTERVAL_IDLE_S = 1.0
DEFAULT_POLL_INTERVAL_GATE_CLOSING_S = 0.3

CAPTURED_IMAGES_BASE_PATH = "captured_images"
ENTRY_IMAGES_PATH = os.path.join(CAPTURED_IMAGES_BASE_PATH, "entry")
EXIT_IMAGES_PATH = os.path.join(CAPTURED_IMAGES_BASE_PATH, "exit")
CV_DEBUG_SAVE_PATH = os.path.join(CAPTURED_IMAGES_BASE_PATH, "cv_debug")


class VehicleEventHandler:
    def __init__(self,
                 camera_entry,
                 camera_exit,
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
        self.ultrasonic_passage_threshold = self.config.get(
            "ultrasonic_passage_threshold", DEFAULT_ULTRASONIC_PASSAGE_THRESHOLD
        )

        us_sensor_for_defaults = self.sensor_manager.ultrasonic_sensor_entry if self.sensor_manager and hasattr(
            self.sensor_manager, 'ultrasonic_sensor_entry') and self.sensor_manager.ultrasonic_sensor_entry else None

        self.ultrasonic_clear_threshold_for_passage = self.config.get(
            "ultrasonic_clear_threshold_for_passage",
            getattr(us_sensor_for_defaults, 'DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR', 2.0)
        )
        self.ultrasonic_clear_confirmation_s = self.config.get(
            "ultrasonic_clear_confirmation_s",
            getattr(us_sensor_for_defaults, 'PASS_CONFIRMATION_TIME_S', 1.5)
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

                reed = self.sensor_manager.reed_switch
                if reed and (not hasattr(reed, '_device') or (
                        hasattr(reed, '_device') and reed._device is not None)):  # Для моків та реального
                    reed_state = reed.are_gates_open
                    if reed_state is True:
                        self._logger.info(f"Ворота успішно відкрито (підтверджено герконом на спробі #{attempt + 1}).")
                        opened_by_reed = True;
                        break
                    elif reed_state is None:
                        self._logger.warning("Не вдалося отримати стан геркона під час очікування відкриття.")
                else:
                    self._logger.warning("Геркон недоступний для перевірки відкриття.")
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
        passage_sensor = self.sensor_manager.ultrasonic_sensor_entry
        if gate_side_name == "exit":
            passage_sensor = getattr(self.sensor_manager, 'ultrasonic_sensor_exit',
                                     None) or self.sensor_manager.ultrasonic_sensor_entry

        if not (passage_sensor and (not hasattr(passage_sensor, '_sensor') or passage_sensor._sensor)):
            self._logger.error(f"УЗД для контролю проїзду ({gate_side_name}) недоступний або не ініціалізований.")
            return False

        self._logger.info(f"Очікування проїзду автомобіля через УЗД ({gate_side_name}).")

        # Перевірка стану воріт перед початком очікування УЗД
        reed = self.sensor_manager.reed_switch
        if not (reed and (not hasattr(reed, '_device') or (hasattr(reed, '_device') and reed._device is not None))):
            self._logger.warning(
                f"Геркон недоступний. Неможливо підтвердити, що ворота відкриті для УЗД ({gate_side_name}).")
        elif not reed.are_gates_open:
            self._logger.warning(f"Ворота не відкриті (геркон). Скасування очікування проїзду УЗД ({gate_side_name}).")
            return False

        self._logger.info(
            f"Ворота відкриті (геркон підтвердив або недоступний). УЗД ({gate_side_name}) активний для детекції проїзду.")

        # Використовуємо wait_for_clear_after_pass, що відповідає логіці ultrasonic_test.py
        # Цей метод в UltrasonicSensor вже інкапсулює логіку "був об'єкт -> зона стала вільною"
        if passage_sensor.wait_for_clear_after_pass(
                threshold_clear_m=self.ultrasonic_clear_threshold_for_passage,
                confirmation_s=self.ultrasonic_clear_confirmation_s,
                timeout=self.passage_confirmation_timeout_s
        ):
            self._logger.info(f"Автомобіль повністю проїхав зону УЗД ({gate_side_name}).")
            return True
        else:
            self._logger.warning(f"Автомобіль не підтвердив проїзд через УЗД ({gate_side_name}) або таймаут.")
            # Додаткова перевірка, чи є зараз перешкода
            if hasattr(passage_sensor, 'detect_object_in_passage') and \
                    passage_sensor.detect_object_in_passage(self.ultrasonic_passage_threshold):
                self._logger.warning(
                    f"УВАГА: Автомобіль або перешкода все ще в зоні УЗД '{gate_side_name}' (<{self.ultrasonic_passage_threshold}м)!")
            return False

    def _manage_auto_close_with_obstruction_check(self) -> bool:
        if not (self.gate_controller and self.sensor_manager and self.sensor_manager.ultrasonic_sensor_entry):
            self._logger.error("Неможливо керувати автозакриттям: компоненти не ініціалізовані.")
            return False

        passage_check_sensor = self.sensor_manager.ultrasonic_sensor_entry
        if not (passage_check_sensor and (
                not hasattr(passage_check_sensor, '_sensor') or passage_check_sensor._sensor)):
            self._logger.error("УЗД для перевірки перешкод при автозакритті недоступний.")
            return False

        self.gate_controller.start_auto_close_timer(timeout_s=self.auto_close_timer_duration_s)
        timer_end_time = time.monotonic() + self.auto_close_timer_duration_s

        while time.monotonic() < timer_end_time:
            if self.shutdown_event and self.shutdown_event.is_set():
                self.gate_controller.cancel_auto_close_timer();
                return False

            if passage_check_sensor.detect_object_in_passage(self.ultrasonic_passage_threshold):
                self._logger.info(
                    f"Перешкода виявлена УЗД під час {self.auto_close_timer_duration_s}с таймера! Зупинка та очікування.")
                self.gate_controller.cancel_auto_close_timer()

                # Очікуємо, поки перешкода зникне
                # Використовуємо wait_for_object_to_clear_passage з sensors_manager
                if not passage_check_sensor.wait_for_object_to_clear_passage(
                        passage_threshold_m=self.ultrasonic_passage_threshold,
                        timeout_s=60
                ):
                    self._logger.warning("Перешкода не зникла з зони УЗД після 60с. Автозакриття скасовано.")
                    return False

                self._logger.info("Перешкода зникла. Перезапуск таймера автозакриття.")
                self.gate_controller.start_auto_close_timer(timeout_s=self.auto_close_timer_duration_s)
                timer_end_time = time.monotonic() + self.auto_close_timer_duration_s

            time.sleep(0.05)

        self._logger.info(f"{self.auto_close_timer_duration_s}с період моніторингу завершився.")
        return True

    def _attempt_close_gate_with_retry(self) -> bool:
        if not self.gate_controller or not self.sensor_manager or \
                not hasattr(self.sensor_manager, 'reed_switch') or not self.sensor_manager.reed_switch:
            self._logger.error("Неможливо закрити ворота: GateController або ReedSwitch не ініціалізовано.")
            return False

        for attempt in range(self.reed_close_retries + 1):
            self._logger.info(f"Спроба закриття воріт #{attempt + 1}...")

            if not self.gate_controller.close_gate():
                self._logger.warning(
                    f"Команда close_gate() не виконана на спробі #{attempt + 1} (можливо, перешкода або вже закрито).")
                current_gate_state = self.gate_controller.get_current_gate_state()
                if current_gate_state == "CLOSED": return True
                if current_gate_state == "OBSTRUCTED" and attempt >= self.reed_close_retries: return False
                if attempt < self.reed_close_retries:
                    time.sleep(1); continue
                else:
                    return False

            start_wait = time.monotonic()
            closed_by_reed = False
            while time.monotonic() - start_wait < self.reed_close_timeout_s:
                if self.shutdown_event and self.shutdown_event.is_set(): return False

                reed = self.sensor_manager.reed_switch
                if reed and (not hasattr(reed, '_device') or (hasattr(reed, '_device') and reed._device is not None)):
                    reed_state = reed.are_gates_closed
                    if reed_state is True:
                        self._logger.info(f"Ворота успішно закрито (підтверджено герконом на спробі #{attempt + 1}).")
                        closed_by_reed = True;
                        break
                    elif reed_state is None:
                        self._logger.warning("Не вдалося отримати стан геркона під час очікування закриття.")
                else:
                    self._logger.warning("Геркон недоступний для перевірки закриття.")
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

    def _handle_gate_closing_interruption(self, camera_type_for_check: str) -> bool:
        is_timer_active = False
        if hasattr(self.gate_controller, '_auto_close_timer') and \
                self.gate_controller._auto_close_timer is not None and \
                self.gate_controller._auto_close_timer.is_alive():
            is_timer_active = True

        if is_timer_active:
            self._logger.debug(f"Перевірка переривання закриття для {camera_type_for_check} (таймер активний)...")
            vehicle_detected_by_cv = False
            if self.cv_processor:
                cam_to_check = self.camera_entry if camera_type_for_check == "entry" else self.camera_exit
                if cam_to_check and cam_to_check.is_initialized_successfully:
                    current_frame = cam_to_check.capture_array()
                    if current_frame is not None:
                        detections = self.cv_processor.detect_vehicle_in_frame(current_frame, camera_type_for_check)
                        if detections:
                            vehicle_detected_by_cv = True
                            self._logger.info(
                                f"CV ВИЯВИВ НОВЕ АВТО ({camera_type_for_check}) під час таймера закриття!")

            vehicle_detected_by_ultrasonic = False
            ultrasonic_to_check = None
            if camera_type_for_check == "entry" and self.sensor_manager.ultrasonic_sensor_entry:
                ultrasonic_to_check = self.sensor_manager.ultrasonic_sensor_entry
            elif camera_type_for_check == "exit":
                ultrasonic_to_check = getattr(self.sensor_manager, 'ultrasonic_sensor_exit', None) or \
                                      self.sensor_manager.ultrasonic_sensor_entry

            if ultrasonic_to_check and hasattr(ultrasonic_to_check, 'is_vehicle_approaching') and \
                    ultrasonic_to_check.is_vehicle_approaching():
                vehicle_detected_by_ultrasonic = True
                self._logger.info(f"УЗД ВИЯВИВ НОВЕ АВТО ({camera_type_for_check}) під час таймера закриття!")

            if vehicle_detected_by_cv or vehicle_detected_by_ultrasonic:
                self._logger.warning(
                    f"НОВИЙ АВТОМОБІЛЬ ({camera_type_for_check}) ВИЯВЛЕНО ПІД ЧАС АКТИВНОГО ТАЙМЕРА ЗАКРИТТЯ!")
                self.gate_controller.interrupt_closing_procedure()
                return True
            else:
                self._logger.debug(f"Нових авто для переривання закриття ({camera_type_for_check}) не виявлено.")
        return False

    def entry_scenario_loop(self):
        self._logger.info("Запуск циклу обробки В'ЇЗДУ...")
        while self.is_running and (self.shutdown_event is None or not self.shutdown_event.is_set()):
            current_poll_interval = self.poll_interval_idle_s
            if self.gate_controller and hasattr(self.gate_controller, '_auto_close_timer') and \
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
                    self._logger.error("В'їзд: Помилка - CV виявив авто, але кадр не збережено.");
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
        self._logger.info("Запуск циклу обробки ВИЇЗДУ...")
        while self.is_running and (self.shutdown_event is None or not self.shutdown_event.is_set()):
            if self.gate_controller and hasattr(self.gate_controller, '_auto_close_timer') and \
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
        if not self.is_running:
            if self.shutdown_event and self.shutdown_event.is_set():
                self._logger.info("VEH вже отримав сигнал на зупинку.")
            else:
                self._logger.info("VEH вже зупинено або не було запущено."); return
        self._logger.info("Зупинка VehicleEventHandler...");
        self.is_running = False
        self._logger.info("VehicleEventHandler: прапор is_running встановлено в False.")


# --- Блок для тестування ---
if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    from unittest.mock import MagicMock

    logger_main_test = logging.getLogger("EventHandlerTest")


    # Мок-класи (мають бути тут для запуску як окремий скрипт)
    class MockCamera:
        def __init__(self, name="mock_cam"):
            self.name, self.is_initialized_successfully = name, True
            self._logger = logging.getLogger(f"MockCamera.{name}")
            self._logger.info(f"Мок-камера '{name}' створена.");
            self.image_counter = 0

        def capture_array(self):
            self.image_counter += 1;
            self._logger.info(f"[{self.name}] capture_array() (кадр {self.image_counter})")
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, f"Frame {self.image_counter} {self.name}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            return dummy_frame

        def capture_image(self, filename): self._logger.info(
            f"[{self.name}] capture_image({filename})"); return filename

        def close(self): self._logger.info(f"[{self.name}] close()")


    class MockUltrasonicSensor:
        def __init__(self, name="mock_us"):
            self.name = name;
            self._logger = logging.getLogger(f"MockUltrasonic.{name}")
            self.should_detect_approach = False;
            self.should_confirm_pass = True
            self.mock_distance = 2.0  # Для detect_object_in_passage

        def is_vehicle_approaching(self, threshold_m=None): time.sleep(0.01); return self.should_detect_approach

        def wait_for_clear_after_pass(self, threshold_clear_m=None, confirmation_s=None, timeout=None):
            self._logger.info(f"[{self.name}] wait_for_clear_after_pass -> {self.should_confirm_pass}")
            if self.should_confirm_pass: time.sleep(0.1)
            return self.should_confirm_pass

        def detect_object_in_passage(self, passage_threshold_m=None): return self.mock_distance < (
                    passage_threshold_m or 0.3)

        def wait_for_object_to_enter_passage(self, passage_threshold_m=None, timeout_s=None): self._logger.info(
            f"[{self.name}] MOCK: wait_for_object_to_enter..."); return self.detect_object_in_passage(
            passage_threshold_m)

        def wait_for_object_to_clear_passage(self, passage_threshold_m=None, timeout_s=None): self._logger.info(
            f"[{self.name}] MOCK: wait_for_object_to_clear..."); return not self.detect_object_in_passage(
            passage_threshold_m)

        def cleanup(self): self._logger.info(f"[{self.name}] cleanup()")


    class MockSensorManager:  # ... (як у попередній версії)
        def __init__(self):
            self.ultrasonic_sensor_entry = MockUltrasonicSensor("US_Entry")
            self.ultrasonic_sensor_exit = MockUltrasonicSensor("US_Exit")
            self.reed_switch = MagicMock()  # Використовуємо MagicMock для геркона
            self.reed_switch.are_gates_open = False
            self.reed_switch.are_gates_closed = True
            self._logger = logging.getLogger("MockSensorManager")
            self._logger.info("Мок-менеджер датчиків створено.")

        def cleanup(self): self._logger.info("MockSensorManager cleanup() викликано.")


    class MockSheetHandler:  # ... (як у попередній версії)
        def __init__(self): self._logger = logging.getLogger("MockSheetHandler")

        def find_vehicle_and_update_entry_time(self, p): self._logger.info(
            f"SHEETS: find_vehicle_and_update_entry_time '{p}'"); return "AUTH" in p

        def add_unauthorized_attempt(self, p): self._logger.info(f"SHEETS: add_unauthorized_attempt '{p}'")

        def log_vehicle_exit(self, p): self._logger.info(f"SHEETS: log_vehicle_exit '{p}'")


    class MockCVProcessor:  # ... (як у попередній версії)
        def __init__(self):
            self._logger = logging.getLogger("MockCVProcessor");
            self.simulated_plate = "AUTH123AA"
            self.should_detect_vehicle_map = {"entry": False, "exit": False};
            self.should_recognize_plate = True

        def detect_vehicle_in_frame(self, img, cam_type, **kwargs):
            d = self.should_detect_vehicle_map.get(cam_type, False);
            self._logger.info(f"CV: detect_vehicle_in_frame ({cam_type}) -> {d}");
            return [(10, 10, 100, 100, 0.9, "car")] if d else None

        def get_plate_number_from_image(self, img, cam_type, **kwargs):
            self._logger.info(f"CV: get_plate_number_from_image ({cam_type})")
            if self.should_recognize_plate: self._logger.info(
                f"CV: Імітація розпізнавання -> '{self.simulated_plate}'"); return self.simulated_plate
            self._logger.info("CV: Імітація НЕ розпізнавання НЗ");
            return None


    class MockGateController:  # ... (як у попередній версії)
        def __init__(self):
            self._logger = logging.getLogger("MockGateController");
            self._auto_close_timer_obj = None;
            self._lock = threading.Lock()

        def open_gate(self):
            self._logger.info("GATE: open_gate()"); return True  # Завжди успішно

        def close_gate(self):
            self._logger.info("GATE: close_gate()"); return True  # Завжди успішно

        def _timer_callback_mock(self):
            with self._lock: self._logger.info(
                "GATE: (Мок) Таймер авто-закриття спрацював."); self._auto_close_timer_obj = None

        def start_auto_close_timer(self, timeout_s=None):
            with self._lock:
                eff_time = timeout_s if timeout_s else 1;
                self._logger.info(f"GATE: start_auto_close_timer(timeout={eff_time})")
                if self._auto_close_timer_obj and self._auto_close_timer_obj.is_alive(): self._auto_close_timer_obj.cancel()
                self._auto_close_timer_obj = threading.Timer(eff_time, self._timer_callback_mock);
                self._auto_close_timer_obj.daemon = True
                self._auto_close_timer_obj.start();
                self._logger.info("GATE: (Мок) Таймер авто-закриття запущено.")

        def interrupt_closing_procedure(self):
            with self._lock:
                self._logger.info("GATE: interrupt_closing_procedure()");
                if self._auto_close_timer_obj and self._auto_close_timer_obj.is_alive(): self._auto_close_timer_obj.cancel(); self._logger.info(
                    "GATE: (Мок) Активний таймер скасовано.")
                self._auto_close_timer_obj = None

        def get_current_gate_state(self):
            return "UNKNOWN"

        def cleanup(self):
            self._logger.info("GATE: cleanup()")


    logger_main_test.info("--- Початок тестування VehicleEventHandler (з виправленим конструктором) ---")
    # ... (решта тестового сценарію з попередньої відповіді) ...
    # ... (або напишіть нові сценарії для тестування уточненої логіки) ...
    mock_cam_entry = MockCamera("EntryCam")
    mock_cam_exit = MockCamera("ExitCam")
    mock_sensors = MockSensorManager()
    mock_sheets = MockSheetHandler()
    mock_cv = MockCVProcessor()
    mock_gate = MockGateController()

    test_config = {
        "sheets_antiduplicate_delay_s": 2, "passage_confirmation_timeout_s": 1.5,
        "poll_interval_idle_s": 0.2, "poll_interval_gate_closing_s": 0.1,
        "reed_open_timeout_s": 1, "reed_open_retries": 0,  # Швидкі таймаути для тестів
        "reed_close_timeout_s": 1, "reed_close_retries": 0,
        "auto_close_timer_duration_s": 1, "gate_finish_closing_delay_s": 0.5,
        "ultrasonic_passage_threshold": 0.3
    }
    event_handler = VehicleEventHandler(
        camera_entry=mock_cam_entry, camera_exit=mock_cam_exit,
        sensor_manager=mock_sensors, sheet_handler=mock_sheets,
        cv_processor=mock_cv, gate_controller=mock_gate, config=test_config
    )
    test_shutdown_event = threading.Event()
    event_handler.start(test_shutdown_event)
    try:
        logger_main_test.info("\n>>> СЦЕНАРІЙ 1: Авторизований в'їзд (тільки CV тригер) <<<")
        mock_sensors.ultrasonic_sensor_entry.should_detect_approach = False  # УЗД НЕ бачить
        mock_cv.should_detect_vehicle_map["entry"] = False
        logger_main_test.info("УЗД неактивний, CV - ні. Очікуємо, що обробка в'їзду НЕ почнеться...")
        time.sleep(0.5)

        logger_main_test.info("Тепер активуємо CV для в'їзду...")
        mock_cv.should_detect_vehicle_map["entry"] = True
        mock_cv.simulated_plate = "AUTH123AA"
        mock_cv.should_recognize_plate = True
        mock_sensors.reed_switch.are_gates_open = True  # Імітуємо, що ворота відкрилися
        mock_sensors.ultrasonic_sensor_entry.should_confirm_pass = True  # Імітуємо, що авто проїхало

        time.sleep(
            event_handler.auto_close_timer_duration_s + event_handler.gate_finish_closing_delay_s + 1)  # Час на цикл

        mock_cv.should_detect_vehicle_map["entry"] = False
        logger_main_test.info(">>> Сценарій 1 завершено <<<")
    finally:
        logger_main_test.info("Завершення тестів VehicleEventHandler...")
        test_shutdown_event.set();
        event_handler.stop()
        if hasattr(event_handler,
                   'entry_thread') and event_handler.entry_thread.is_alive(): event_handler.entry_thread.join(timeout=2)
        if hasattr(event_handler,
                   'exit_thread') and event_handler.exit_thread.is_alive(): event_handler.exit_thread.join(timeout=2)
        logger_main_test.info("Тестування завершено.")