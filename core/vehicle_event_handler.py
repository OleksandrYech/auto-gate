# core/vehicle_event_handler.py
import logging
import time
import threading
from typing import Optional, Dict, Any
from datetime import datetime

from .camera_manager import CameraController
from .sensors_manager import SensorManager
from .sheet_handler import SheetHandler
from .cv_processor import CVProcessor
from .gate_controller import GateController

try:
    from bot.telegram_notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None

logger = logging.getLogger(__name__)


class VehicleEventHandler:
    def __init__(self,
                 camera_entry: Optional[CameraController],
                 camera_exit: Optional[CameraController],
                 sensor_manager: SensorManager,
                 sheet_handler: SheetHandler,
                 cv_processor: CVProcessor,
                 gate_controller: GateController,
                 config: Dict[str, Any],
                 notifier: Optional[TelegramNotifier] = None):

        self.camera_entry = camera_entry
        self.camera_exit = camera_exit
        self.sensor_manager = sensor_manager
        self.sheet_handler = sheet_handler
        self.cv_processor = cv_processor
        self.gate_controller = gate_controller
        self.notifier = notifier
        self.config = config

        # --- НОВА ЛОГІКА: Словник для відстеження часу останньої детекції ---
        self.last_detection_times: Dict[str, float] = {}
        # Час у секундах, протягом якого повторні детекції ігноруються
        self.cooldown_period_s = config.get("sheets_antiduplicate_delay_s", 60)
        # -----------------------------------------------------------------

        self.is_running = False
        self.shutdown_event = threading.Event()
        self.system_busy = threading.Lock()

        self.entry_thread: Optional[threading.Thread] = None
        self.exit_thread: Optional[threading.Thread] = None

    def _process_vehicle_cycle(self, cam_type: str):
        """Цикл керування воротами з використанням геркона для відкриття та закриття."""
        logger.info(f"[{cam_type.upper()}] Початок циклу. Команда на відкриття воріт.")
        self.gate_controller.open_gate()

        if not self.sensor_manager.reed_switch.wait_for_open(timeout=self.config['reed_open_timeout_s']):
            logger.error(f"[{cam_type.upper()}] Ворота не відкрилися (немає сигналу від геркона). Цикл перервано.")
            return

        logger.info(f"[{cam_type.upper()}] Ворота відкрито. Очікування проїзду.")
        
        passage_timeout = self.config['passage_timeout_s']
        time.sleep(passage_timeout)
        logger.info(f"[{cam_type.upper()}] Час на проїзд ({passage_timeout}с) вичерпано. Початок закриття.")

        self.gate_controller.close_gate()
        
        gate_travel_time = self.config['gate_travel_time_s']
        logger.info(f"[{cam_type.upper()}] Очікування, поки ворота почнуть зачинятися (сигнал від геркона)...")
        
        if self.sensor_manager.reed_switch.wait_for_closed(timeout=gate_travel_time):
            logger.info(f"[{cam_type.upper()}] Геркон розімкнувся. Ворота зачиняються або вже зачинені.")
        else:
            logger.warning(f"[{cam_type.upper()}] Не отримано сигнал про закриття від геркона за {gate_travel_time}с.")
        
        logger.info(f"[{cam_type.upper()}] Повний цикл завершено.")

    def handle_request(self, cam_type: str, plate_text: str, photo_path: Optional[str] = None):
        if self.system_busy.locked():
            logger.warning(f"[{cam_type.upper()}] Система зайнята. Запит для '{plate_text}' ігнорується.")
            return

        with self.system_busy:
            if cam_type == 'entry':
                is_authorized = self.sheet_handler.find_vehicle_and_update_entry_time(plate_text)
                status = "Авторизовано" if is_authorized else "НЕ Авторизовано"

                if self.notifier and photo_path:
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    self.notifier.send_notification(photo_path, plate_text, timestamp, status)
                
                if is_authorized:
                    logger.info(f"[ENTRY] Номер '{plate_text}' авторизовано.")
                    self._process_vehicle_cycle(cam_type)
                else:
                    logger.info(f"[ENTRY] Номер '{plate_text}' НЕ авторизовано.")
                    self.sheet_handler.add_unauthorized_attempt(plate_text)

            elif cam_type == 'exit':
                logger.info(f"[EXIT] Автомобіль на виїзд. Відкриття воріт.")
                self.sheet_handler.log_vehicle_exit(plate_text or "UNKNOWN")
                self._process_vehicle_cycle(cam_type)

    def _polling_loop(self, camera: Optional[CameraController], cam_type: str):
        """Цикл, що постійно опитує камеру та фільтрує повторні детекції."""
        while not self.shutdown_event.is_set():
            if self.system_busy.locked():
                time.sleep(self.config.get('poll_interval_idle_s', 1.0))
                continue

            if camera and camera.is_initialized_successfully:
                try:
                    frame = camera.capture_array()
                    if frame is None:
                        time.sleep(self.config.get('poll_interval_idle_s', 1.0))
                        continue

                    if self.cv_processor.detect_vehicle_in_frame(frame, camera_type=cam_type):
                        logger.debug(f"[{cam_type.upper()}] Виявлено рух. Запуск розпізнавання номера...")
                        
                        plate, photo_path = self.cv_processor.get_plate_number_from_image(
                            frame, cam_type, save_intermediate_steps=True, save_path_prefix="captures"
                        )
                        
                        if plate:
                            # --- ОНОВЛЕНА ЛОГІКА: Перевірка "періоду спокою" ---
                            current_time = time.monotonic()
                            last_seen = self.last_detection_times.get(plate)
                            
                            if last_seen and (current_time - last_seen < self.cooldown_period_s):
                                logger.info(f"Номер '{plate}' розпізнано повторно. Ігноруємо, оскільки не минуло {self.cooldown_period_s}с.")
                                # Пропускаємо обробку, але не оновлюємо час, щоб "період спокою" не продовжувався нескінченно
                                continue
                            
                            # Якщо номер бачимо вперше або "період спокою" минув - обробляємо
                            self.last_detection_times[plate] = current_time
                            logger.info(f"Нова детекція для номера '{plate}'. Передача в обробник.")
                            self.handle_request(cam_type, plate, photo_path)
                            # ----------------------------------------------------

                except Exception as e:
                    logger.error(f"Помилка в циклі опитування камери '{cam_type}': {e}", exc_info=True)
                    time.sleep(5)

            time.sleep(self.config.get('poll_interval_idle_s', 1.0))

    def start(self, shutdown_event: threading.Event):
        # ... (код без змін)
        if self.is_running:
            return
        self.is_running = True
        self.shutdown_event = shutdown_event
        if self.camera_entry:
            self.entry_thread = threading.Thread(target=self._polling_loop, args=(self.camera_entry, 'entry'))
            self.entry_thread.start()
        if self.camera_exit:
            self.exit_thread = threading.Thread(target=self._polling_loop, args=(self.camera_exit, 'exit'))
            self.exit_thread.start()
        logger.info("Потоки обробки в'їзду та виїзду запущено.")

    def stop(self):
        # ... (код без змін)
        self.is_running = False
        if self.entry_thread:
            self.entry_thread.join()
        if self.exit_thread:
            self.exit_thread.join()
        logger.info("Потоки обробки зупинено.")
