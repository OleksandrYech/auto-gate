# core/vehicle_event_handler.py
import logging
import time
import threading
import os
import numpy as np
import cv2

logger = logging.getLogger(__name__)

# --- Конфігураційні константи для обробника (приклади) ---
# Можна передавати через словник config в __init__
DEFAULT_SHEETS_ANTIDUPLICATE_DELAY_S = 60  # Сек, щоб уникнути дублів у Sheets
DEFAULT_PASSAGE_CONFIRMATION_TIMEOUT_S = 20  # Сек, очікування проїзду авто через УЗД
DEFAULT_POLL_INTERVAL_IDLE_S = 2  # Сек, інтервал перевірки наявності авто в режимі очікування
DEFAULT_POLL_INTERVAL_GATE_CLOSING_S = 0.5  # Сек, інтервал перевірки під час закритого таймера

# Директорії для збереження зображень (якщо ще не визначені глобально)
CAPTURED_IMAGES_BASE_PATH = "captured_images"
ENTRY_IMAGES_PATH = os.path.join(CAPTURED_IMAGES_BASE_PATH, "entry")
EXIT_IMAGES_PATH = os.path.join(CAPTURED_IMAGES_BASE_PATH, "exit")


class VehicleEventHandler:
    """
    Обробляє сценарії в'їзду та виїзду автомобілів,
    координуючи роботу всіх системних модулів.
    """

    def __init__(self,
                 camera_entry,  # Екземпляр CameraController для в'їзду
                 camera_exit,  # Екземпляр CameraController для виїзду
                 sensor_manager,
                 sheet_handler,
                 cv_processor,
                 gate_controller,
                 config: dict = None):  # Словник з конфігураціями

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
        self.passage_confirmation_timeout_s = self.config.get(
            "passage_confirmation_timeout_s", DEFAULT_PASSAGE_CONFIRMATION_TIMEOUT_S
        )
        self.poll_interval_idle_s = self.config.get(
            "poll_interval_idle_s", DEFAULT_POLL_INTERVAL_IDLE_S
        )
        self.poll_interval_gate_closing_s = self.config.get(
            "poll_interval_gate_closing_s", DEFAULT_POLL_INTERVAL_GATE_CLOSING_S
        )

        self.is_running = False
        self.shutdown_event = None  # Буде встановлено з main.py

        # Кеш для запобігання дублюванню записів у Sheets
        self.recently_logged_plates = {}  # формат: {'plate_number': timestamp_last_logged}
        self._plate_cache_lock = threading.Lock()

        # Створюємо директорії для зображень, якщо їх немає
        os.makedirs(ENTRY_IMAGES_PATH, exist_ok=True)
        os.makedirs(EXIT_IMAGES_PATH, exist_ok=True)

        self._logger.info("VehicleEventHandler ініціалізовано.")
        self._logger.info(f"  Анти-дублікат для Sheets: {self.sheets_antiduplicate_delay_s} с")
        self._logger.info(f"  Таймаут підтвердження проїзду УЗД: {self.passage_confirmation_timeout_s} с")

    def _is_duplicate_log(self, plate_number: str) -> bool:
        """
        Перевіряє, чи не був цей номерний знак залогований нещодавно.
        Оновлює час останнього логування, якщо це не дублікат.
        """
        with self._plate_cache_lock:
            now = time.time()
            if plate_number in self.recently_logged_plates:
                last_log_time = self.recently_logged_plates[plate_number]
                if (now - last_log_time) < self.sheets_antiduplicate_delay_s:
                    self._logger.info(
                        f"Дублікатний запис для НЗ '{plate_number}'. Минуло менше {self.sheets_antiduplicate_delay_s} с.")
                    return True
            # Якщо не дублікат, або час минув, оновлюємо/додаємо запис
            self.recently_logged_plates[plate_number] = now
            self.recently_logged_plates = {
                p: t for p, t in self.recently_logged_plates.items()
                if (now - t) < (self.sheets_antiduplicate_delay_s * 10) # Зберігати, наприклад, 10х інтервал
            }
            return False

    def _capture_and_recognize_plate(self, camera_controller, camera_type: str, image_save_dir: str) -> str | None:
        """Захоплює зображення та розпізнає номерний знак."""
        if not camera_controller or not camera_controller.is_initialized_successfully:
            self._logger.error(f"Камера '{camera_type}' не доступна для захоплення зображення.")
            return None

        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
        image_filename = os.path.join(image_save_dir, f"{camera_type}_{timestamp_str}.jpg")

        # Змінено: спочатку capture_array, потім, якщо потрібно, capture_file
        image_bgr_array = camera_controller.capture_array()

        if image_bgr_array is None:
            self._logger.error(f"Не вдалося захопити масив зображення з камери '{camera_type}'.")
            # Спробуємо захопити у файл як запасний варіант
            if not camera_controller.capture_image(image_filename):
                self._logger.error(f"Не вдалося зберегти зображення у файл з камери '{camera_type}'.")
                return None
            return None

        if self.cv_processor:  # Зберігаємо зображення, яке піде в CV, якщо save_intermediate_steps буде True
            # CVProcessor сам збереже оригінал, якщо save_intermediate_steps=True
            pass

        if self.cv_processor:
            # Передаємо save_path_prefix для збереження проміжних кроків CV
            # save_intermediate_steps вирішується всередині get_plate_number_from_image або передається як параметр
            plate_text = self.cv_processor.get_plate_number_from_image(
                image_bgr_array,
                camera_type=camera_type,
                save_intermediate_steps=True,  # Встановіть True для відладки CV
                save_path_prefix=os.path.join(CAPTURED_IMAGES_BASE_PATH, "cv_debug", camera_type)
            )
            if plate_text:
                self._logger.info(f"Розпізнано номерний знак '{plate_text}' на камері '{camera_type}'.")
                # Збереження успішного фото, якщо потрібно окремо від CV відладки
                # (cv_processor може це робити сам, якщо save_intermediate_steps=True)
                # cv2.imwrite(image_filename, image_bgr_array) # Якщо хочемо зберегти саме цей кадр
            else:
                self._logger.warning(f"Номерний знак не розпізнано на камері '{camera_type}'.")
                # Зберігаємо зображення, де не вдалося розпізнати, для аналізу
                failed_image_filename = os.path.join(image_save_dir, f"{camera_type}_failed_ocr_{timestamp_str}.jpg")
                cv2.imwrite(failed_image_filename, image_bgr_array)
                self._logger.info(f"Збережено зображення з невдалим розпізнаванням: {failed_image_filename}")

            return plate_text
        return None

    def _wait_for_vehicle_to_pass(self, gate_side_name: str) -> bool:
        """
        Очікує, поки автомобіль повністю проїде зону воріт,
        використовуючи ультразвуковий датчик.
        """
        self._logger.info(f"Очікування проїзду автомобіля через ворота ({gate_side_name})...")
        if self.sensor_manager and self.sensor_manager.ultrasonic_sensor_entry:
            if self.sensor_manager.ultrasonic_sensor_entry.wait_for_clear_after_pass(
                    timeout=self.passage_confirmation_timeout_s
            ):
                self._logger.info(f"Автомобіль проїхав зону воріт ({gate_side_name}).")
                return True
            else:
                self._logger.warning(f"Тайм-аут очікування проїзду автомобіля ({gate_side_name}). "
                                     "Можливо, автомобіль зупинився у проїзді.")
                return False
        else:
            self._logger.error("Ультразвуковий датчик для контролю проїзду недоступний.")
            return False  # Не можемо підтвердити проїзд

    def _handle_gate_closing_interruption(self, camera_type_for_check: str):
        """
        Перевіряє наявність нового авто під час активного таймера закриття воріт
        і перериває закриття, якщо авто виявлено.
        Повертає True, якщо закриття було перервано.
        """
        if self.gate_controller._auto_close_timer and \
                self.gate_controller._auto_close_timer.is_alive():

            vehicle_detected_by_cv = False
            if self.cv_processor:
                # Потрібно отримати поточний кадр з камери camera_type_for_check
                cam_to_check = self.camera_entry if camera_type_for_check == "entry" else self.camera_exit
                if cam_to_check and cam_to_check.is_initialized_successfully:
                    current_frame = cam_to_check.capture_array()
                    if current_frame is not None:
                        detections = self.cv_processor.detect_vehicle_in_frame(current_frame, camera_type_for_check)
                        if detections:  # Якщо список не порожній
                            vehicle_detected_by_cv = True
                            self._logger.info(
                                f"CV ВИЯВИВ НОВЕ АВТО ({camera_type_for_check}) під час таймера закриття!")

            vehicle_detected_by_ultrasonic = False
            ultrasonic_to_check = None
            if camera_type_for_check == "entry" and self.sensor_manager.ultrasonic_sensor_entry:
                ultrasonic_to_check = self.sensor_manager.ultrasonic_sensor_entry
            elif camera_type_for_check == "exit" and self.sensor_manager.ultrasonic_sensor_exit:  # Якщо є окремий
                ultrasonic_to_check = self.sensor_manager.ultrasonic_sensor_exit
            # Якщо немає окремого УЗД для виїзду, можна не перевіряти УЗД для виїзду тут

            if ultrasonic_to_check and ultrasonic_to_check.is_vehicle_approaching():
                vehicle_detected_by_ultrasonic = True
                self._logger.info(f"УЗД ВИЯВИВ НОВЕ АВТО ({camera_type_for_check}) під час таймера закриття!")

            if vehicle_detected_by_cv or vehicle_detected_by_ultrasonic:
                self._logger.warning(
                    f"НОВИЙ АВТОМОБІЛЬ ({camera_type_for_check}) ВИЯВЛЕНО ПІД ЧАС АКТИВНОГО ТАЙМЕРА ЗАКРИТТЯ!")
                self.gate_controller.interrupt_closing_procedure()
                return True  # Закриття було перервано, потрібно обробити нове авто
        return False

    def entry_scenario_loop(self):
        """Основний цикл для обробки сценарію В'ЇЗДУ."""
        self._logger.info("Запуск циклу обробки В'ЇЗДУ...")
        while self.is_running and (self.shutdown_event is None or not self.shutdown_event.is_set()):
            current_poll_interval = self.poll_interval_idle_s
            if self.gate_controller._auto_close_timer and self.gate_controller._auto_close_timer.is_alive():
                current_poll_interval = self.poll_interval_gate_closing_s  # Частіше перевіряємо, якщо ворота закриваються
                if self._handle_gate_closing_interruption("entry"):
                    self._logger.info("В'їзд: Закриття перервано новим авто. Повторний цикл детекції.")
                    time.sleep(0.1)  # Коротка пауза перед негайним повтором
                    continue  # Негайно перевіряємо нове авто

            self._logger.debug("В'їзд: Очікування автомобіля...")

            # 1. Детекція автомобіля (пріоритет CV, потім УЗД)
            vehicle_approaching = False
            initial_detection_frame = None

            if self.camera_entry and self.camera_entry.is_initialized_successfully and self.cv_processor:
                frame_for_detection = self.camera_entry.capture_array()
                if frame_for_detection is not None:
                    if self.cv_processor.detect_vehicle_in_frame(frame_for_detection, camera_type="entry"):
                        self._logger.info("В'їзд: CV зафіксував автомобіль.")
                        vehicle_approaching = True
                        initial_detection_frame = frame_for_detection

            if not vehicle_approaching and self.sensor_manager.ultrasonic_sensor_entry:
                if self.sensor_manager.ultrasonic_sensor_entry.is_vehicle_approaching():  # Неблокуюча перевірка
                    self._logger.info("В'їзд: УЗД зафіксував наближення.")
                    vehicle_approaching = True

            if vehicle_approaching:
                self._logger.info("В'їзд: Автомобіль виявлено. Початок обробки.")

                # Якщо initial_detection_frame не було отримано (напр. тільки УЗД спрацював), робимо новий кадр
                if initial_detection_frame is None and self.camera_entry:
                    initial_detection_frame = self.camera_entry.capture_array()

                if initial_detection_frame is None:
                    self._logger.error("В'їзд: Не вдалося отримати кадр для розпізнавання НЗ.")
                    time.sleep(current_poll_interval)
                    continue

                # Розпізнавання НЗ (використовує initial_detection_frame)
                plate_text = self.cv_processor.get_plate_number_from_image(
                    initial_detection_frame, camera_type="entry",
                    save_intermediate_steps=True,  # Увімкніть для відладки
                    save_path_prefix=os.path.join(CAPTURED_IMAGES_BASE_PATH, "cv_debug", "entry")
                )

                if plate_text:
                    if not self._is_duplicate_log(plate_text):
                        if self.sheet_handler.find_vehicle_and_update_entry_time(plate_text):
                            self._logger.info(f"В'їзд: Автомобіль '{plate_text}' АВТОРИЗОВАНО. Відкриття воріт.")
                            if self.gate_controller.open_gate():  # Перевірка, чи команда була успішною
                                if self._wait_for_vehicle_to_pass("в'їзду"):
                                    self.gate_controller.start_auto_close_timer()
                                else:
                                    # Авто не проїхало, можливо, потрібно закрити ворота або інша логіка
                                    self._logger.warning(
                                        "В'їзд: Авто не підтвердило проїзд після відкриття. Таймер закриття не запущено.")
                                    # Можливо, варто спробувати закрити ворота, якщо вони відкрилися
                                    # self.gate_controller.close_gate()
                        else:
                            self._logger.info(f"В'їзд: Автомобіль '{plate_text}' НЕ АВТОРИЗОВАНО.")
                            self.sheet_handler.add_unauthorized_attempt(plate_text)
                else:
                    self._logger.warning("В'їзд: Номерний знак не розпізнано.")
                    # Зберігаємо зображення, де не вдалося розпізнати (якщо це ще не зроблено в _capture_and_recognize_plate)
                    # Ця логіка тепер всередині _capture_and_recognize_plate / get_plate_number_from_image

            time.sleep(current_poll_interval)
        self._logger.info("Цикл обробки В'ЇЗДУ завершено.")

    def exit_scenario_loop(self):
        """Основний цикл для обробки сценарію ВИЇЗДУ."""
        self._logger.info("Запуск циклу обробки ВИЇЗДУ...")

        # Визначаємо, який УЗД використовувати для виїзду
        ultrasonic_exit_sensor = self.sensor_manager.ultrasonic_sensor_exit \
            if hasattr(self.sensor_manager, 'ultrasonic_sensor_exit') and self.sensor_manager.ultrasonic_sensor_exit \
            else self.sensor_manager.ultrasonic_sensor_entry  # Fallback на датчик в'їзду

        while self.is_running and (self.shutdown_event is None or not self.shutdown_event.is_set()):
            current_poll_interval = self.poll_interval_idle_s
            if self.gate_controller._auto_close_timer and self.gate_controller._auto_close_timer.is_alive():
                current_poll_interval = self.poll_interval_gate_closing_s
                if self._handle_gate_closing_interruption("exit"):
                    self._logger.info("Виїзд: Закриття перервано новим авто. Повторний цикл детекції.")
                    time.sleep(0.1)
                    continue

            self._logger.debug("Виїзд: Очікування автомобіля...")

            vehicle_approaching_exit = False
            initial_detection_frame_exit = None

            if self.camera_exit and self.camera_exit.is_initialized_successfully and self.cv_processor:
                frame_for_detection = self.camera_exit.capture_array()
                if frame_for_detection is not None:
                    if self.cv_processor.detect_vehicle_in_frame(frame_for_detection, camera_type="exit"):
                        self._logger.info("Виїзд: CV зафіксував автомобіль.")
                        vehicle_approaching_exit = True
                        initial_detection_frame_exit = frame_for_detection

            if not vehicle_approaching_exit and ultrasonic_exit_sensor:
                if ultrasonic_exit_sensor.is_vehicle_approaching():
                    self._logger.info("Виїзд: УЗД зафіксував наближення.")
                    vehicle_approaching_exit = True

            if vehicle_approaching_exit:
                self._logger.info("Виїзд: Автомобіль виявлено. Негайне відкриття воріт.")
                if self.gate_controller.open_gate():
                    # Робимо фото ПІСЛЯ відкриття або паралельно
                    if initial_detection_frame_exit is None and self.camera_exit:
                        initial_detection_frame_exit = self.camera_exit.capture_array()

                    if initial_detection_frame_exit is None:
                        self._logger.error(
                            "Виїзд: Не вдалося отримати кадр для розпізнавання НЗ після відкриття воріт.")
                    else:
                        plate_text = self.cv_processor.get_plate_number_from_image(
                            initial_detection_frame_exit, camera_type="exit",
                            save_intermediate_steps=True,  # Увімкніть для відладки CV
                            save_path_prefix=os.path.join(CAPTURED_IMAGES_BASE_PATH, "cv_debug", "exit")
                        )
                        if plate_text:
                            if not self._is_duplicate_log(plate_text):
                                self.sheet_handler.log_vehicle_exit(plate_text)
                        else:
                            self._logger.warning("Виїзд: Номерний знак не розпізнано, але виїзд дозволено.")

                    if self._wait_for_vehicle_to_pass("виїзду"):
                        self.gate_controller.start_auto_close_timer()
                    else:
                        self._logger.warning(
                            "Виїзд: Авто не підтвердило проїзд після відкриття. Таймер закриття не запущено.")

            time.sleep(current_poll_interval)
        self._logger.info("Цикл обробки ВИЇЗДУ завершено.")

    def start(self, shutdown_event_main):
        """Запускає основні цикли обробки в окремих потоках."""
        if self.is_running:
            self._logger.warning("VehicleEventHandler вже запущено.")
            return

        self._logger.info("Запуск VehicleEventHandler...")
        self.is_running = True
        self.shutdown_event = shutdown_event_main

        self.entry_thread = threading.Thread(target=self.entry_scenario_loop, name="EntryScenarioThread")
        self.exit_thread = threading.Thread(target=self.exit_scenario_loop, name="ExitScenarioThread")

        self.entry_thread.daemon = True
        self.exit_thread.daemon = True

        self.entry_thread.start()
        self.exit_thread.start()
        self._logger.info("Потоки обробки в'їзду та виїзду запущено.")

    def stop(self):
        """Сигналізує про зупинку обробки."""
        if not self.is_running:
            self._logger.info("VehicleEventHandler вже зупинено або не було запущено.")
            return

        self._logger.info("Зупинка VehicleEventHandler...")
        self.is_running = False  # Для виходу з циклів while


        self._logger.info("VehicleEventHandler отримав сигнал на зупинку.")


# В кінці файлу core/vehicle_event_handler.py

if __name__ == '__main__':
    # Налаштування логування для тестування (якщо ще не налаштовано глобально)
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s',
            handlers=[logging.StreamHandler()]
        )

    logger_main_test = logging.getLogger("EventHandlerTest")


    # --- Спрощені Мок-класи ---
    class MockCamera:
        def __init__(self, name="mock_cam"):
            self.name = name
            self.is_initialized_successfully = True
            self._logger = logging.getLogger(f"MockCamera.{name}")
            self._logger.info(f"Мок-камера '{name}' створена.")
            self.image_counter = 0

        def capture_array(self):
            self.image_counter += 1
            self._logger.info(f"[{self.name}] capture_array() викликано (кадр {self.image_counter})")
            # Повертаємо фіктивне зображення, щоб CVProcessor міг його "обробити"
            dummy_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(dummy_frame, f"Frame {self.image_counter}", (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1,
                        (255, 255, 255), 2)
            return dummy_frame

        def capture_image(self, filename):  # Для сумісності, якщо CVProcessor її викликає
            self._logger.info(f"[{self.name}] capture_image({filename}) викликано")
            # Можна імітувати збереження, якщо потрібно
            return filename

        def close(self): self._logger.info(f"[{self.name}] close() викликано")


    class MockUltrasonicSensor:
        def __init__(self, name="mock_us"):
            self.name = name
            self._logger = logging.getLogger(f"MockUltrasonic.{name}")
            self.should_detect_approach = False
            self.should_confirm_pass = True  # Чи має wait_for_clear_after_pass повернути True

        def is_vehicle_approaching(self, threshold_m=None):
            # self._logger.debug(f"[{self.name}] is_vehicle_approaching() -> {self.should_detect_approach}")
            return self.should_detect_approach

        def wait_for_clear_after_pass(self, timeout=None):
            self._logger.info(
                f"[{self.name}] wait_for_clear_after_pass(timeout={timeout}) -> {self.should_confirm_pass}")
            if self.should_confirm_pass:
                time.sleep(0.1)  # Імітація невеликої затримки
            return self.should_confirm_pass

        def cleanup(self): self._logger.info(f"[{self.name}] cleanup() викликано")


    class MockSensorManager:
        def __init__(self):
            self.ultrasonic_sensor_entry = MockUltrasonicSensor("US_Entry")

            self.ultrasonic_sensor_exit = self.ultrasonic_sensor_entry

            self._logger = logging.getLogger("MockSensorManager")
            self._logger.info("Мок-менеджер датчиків створено.")

        def cleanup(self): self._logger.info("MockSensorManager cleanup() викликано.")


    class MockSheetHandler:
        def __init__(self): self._logger = logging.getLogger("MockSheetHandler")

        def find_vehicle_and_update_entry_time(self, plate_number):
            self._logger.info(f"SHEETS: find_vehicle_and_update_entry_time для '{plate_number}'")
            # Імітуємо: нехай деякі номери будуть авторизовані
            if "AA1234AA" in plate_number: return True
            return False

        def add_unauthorized_attempt(self, plate_number):
            self._logger.info(f"SHEETS: add_unauthorized_attempt для '{plate_number}'")

        def log_vehicle_exit(self, plate_number):
            self._logger.info(f"SHEETS: log_vehicle_exit для '{plate_number}'")


    class MockCVProcessor:
        def __init__(self):
            self._logger = logging.getLogger("MockCVProcessor")
            self.simulated_plate = "AA1234AA"  # Номер, який буде "розпізнано"
            self.should_detect_vehicle = False
            self.should_recognize_plate = True

        def detect_vehicle_in_frame(self, image_bgr, camera_type):
            self._logger.info(f"CV: detect_vehicle_in_frame (камера: {camera_type}) -> {self.should_detect_vehicle}")
            if self.should_detect_vehicle:
                # Повертаємо список з однією фіктивною рамкою
                return [(10, 10, 100, 100, 0.9, "car")]
            return None

        def get_plate_number_from_image(self, image_bgr, camera_type, save_intermediate_steps=False,
                                        save_path_prefix=""):
            self._logger.info(f"CV: get_plate_number_from_image (камера: {camera_type})")
            if self.should_recognize_plate:
                self._logger.info(f"CV: Імітація розпізнавання -> '{self.simulated_plate}'")
                return self.simulated_plate
            self._logger.info("CV: Імітація НЕ розпізнавання НЗ")
            return None


    class MockGateController:
        def __init__(self):
            self._logger = logging.getLogger("MockGateController")
            self._auto_close_timer = None
            self._is_timer_active_mock = False  # Для імітації активного таймера

        def open_gate(self):
            self._logger.info("GATE: open_gate() викликано")
            return True

        def close_gate(self):
            self._logger.info("GATE: close_gate() викликано")
            return True

        def start_auto_close_timer(self, timeout_s=None):
            self._logger.info(f"GATE: start_auto_close_timer(timeout={timeout_s}) викликано")
            self._is_timer_active_mock = True

            # Імітуємо, що таймер колись завершиться
            def _clear_timer_flag():
                time.sleep(timeout_s if timeout_s else 1)
                self._is_timer_active_mock = False
                self._logger.info("GATE: (Мок) Таймер авто-закриття завершився")

            # Видаляємо попередній мок-таймер, якщо він ще існує
            if self._auto_close_timer and self._auto_close_timer.is_alive():
                self._auto_close_timer.cancel()

            self._auto_close_timer = threading.Timer(0.1, _clear_timer_flag)  # Швидкий запуск для тесту
            self._auto_close_timer.daemon = True
            self._auto_close_timer.start()

        def interrupt_closing_procedure(self):
            self._logger.info("GATE: interrupt_closing_procedure() викликано")
            if self._auto_close_timer and self._auto_close_timer.is_alive():
                self._auto_close_timer.cancel()
            self._is_timer_active_mock = False

        def get_current_gate_state(self):
            return "UNKNOWN"  # Спрощено

        # Додамо властивість для імітації _auto_close_timer.is_alive()
        @property
        def is_auto_close_timer_active(self):
            # self._logger.debug(f"GATE: Перевірка is_auto_close_timer_active -> {self._is_timer_active_mock}")
            return self._is_timer_active_mock

        def cleanup(self):
            self._logger.info("GATE: cleanup() викликано")


    logger_main_test.info("--- Початок тестування VehicleEventHandler ---")

    # Створюємо мок-об'єкти
    mock_cam_entry = MockCamera("EntryCam")
    mock_cam_exit = MockCamera("ExitCam")
    mock_sensors = MockSensorManager()
    mock_sheets = MockSheetHandler()
    mock_cv = MockCVProcessor()
    mock_gate = MockGateController()

    # Конфігурація для VehicleEventHandler
    test_config = {
        "sheets_antiduplicate_delay_s": 5,  # Короткий для тесту
        "passage_confirmation_timeout_s": 3,
        "poll_interval_idle_s": 0.5,
        "poll_interval_gate_closing_s": 0.2
    }

    # Ініціалізуємо обробник
    event_handler = VehicleEventHandler(
        camera_entry=mock_cam_entry,
        camera_exit=mock_cam_exit,
        sensor_manager=mock_sensors,
        sheet_handler=mock_sheets,
        cv_processor=mock_cv,
        gate_controller=mock_gate,
        config=test_config
    )

    # Створюємо подію для завершення
    test_shutdown_event = threading.Event()
    event_handler.start(test_shutdown_event)

    try:
        # --- Симуляція сценарію В'ЇЗДУ ---
        logger_main_test.info("\n>>> СИМУЛЯЦІЯ: Авторизований в'їзд <<<")
        mock_cv.should_detect_vehicle = True  # CV "бачить" авто
        mock_cv.simulated_plate = "AA1234AA"  # Авторизований номер
        mock_cv.should_recognize_plate = True
        mock_sensors.ultrasonic_sensor_entry.should_detect_approach = True  # УЗД також "бачить"

        time.sleep(2)  # Даємо час на обробку

        # Перевірка переривання закриття
        logger_main_test.info("\n>>> СИМУЛЯЦІЯ: Переривання закриття новим авто на В'ЇЗДІ <<<")

        if mock_gate.is_auto_close_timer_active:  # Якщо попередній проїзд запустив таймер
            logger_main_test.info("Таймер закриття активний. Симулюємо нове авто...")
            mock_cv.simulated_plate = "BB5678BB"  # Нове авто
            mock_sensors.ultrasonic_sensor_entry.should_detect_approach = True
            mock_cv.should_detect_vehicle = True  # CV бачить нове авто
            # event_handler має сам перервати закриття і почати обробку нового авто
            time.sleep(1)  # Даємо час на реакцію
        else:
            logger_main_test.info("Таймер закриття не активний для тесту переривання, "
                                  "або логіка _is_timer_active_mock потребує доопрацювання.")

        # --- Симуляція сценарію ВИЇЗДУ ---
        logger_main_test.info("\n>>> СИМУЛЯЦІЯ: Виїзд <<<")
        mock_cv.should_detect_vehicle = False  # Вимикаємо детекцію на в'їзді
        mock_sensors.ultrasonic_sensor_entry.should_detect_approach = False

        # Імітуємо наближення на виїзді
        mock_cv.simulated_plate = "CC9012CC"
        time.sleep(5)  # Загальний час роботи для тестів

    except KeyboardInterrupt:
        logger_main_test.info("Тест перервано користувачем.")
    finally:
        logger_main_test.info("Завершення тестів VehicleEventHandler...")
        test_shutdown_event.set()  # Сигнал для зупинки потоків
        event_handler.stop()  # Виклик методу stop

        # Даємо потокам час на завершення
        if hasattr(event_handler, 'entry_thread') and event_handler.entry_thread.is_alive():
            event_handler.entry_thread.join(timeout=2)
        if hasattr(event_handler, 'exit_thread') and event_handler.exit_thread.is_alive():
            event_handler.exit_thread.join(timeout=2)

        logger_main_test.info("Тестування завершено.")