# core/vehicle_event_handler.py
import logging
import time
import threading
from typing import Optional, Dict, Any, Tuple
from datetime import datetime

# Припускаємо, що ці модулі існують у проєкті
from .camera_manager import CameraController
from .sensors_manager import SensorManager
from .sheet_handler import SheetHandler
from .cv_processor import CVProcessor
from .gate_controller import GateController

# Опціональний імпорт для сповіщень
try:
    from bot.telegram_notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None

logger = logging.getLogger(__name__)


class VehicleEventHandler:
    """
    Головний клас, що керує логікою обробки подій, пов'язаних з автомобілями.
    """
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

        self.is_running = False
        self.shutdown_event = threading.Event()
        self.system_busy = threading.Lock()

        self.entry_thread: Optional[threading.Thread] = None
        self.exit_thread: Optional[threading.Thread] = None

    def _process_vehicle_cycle(self, cam_type: str):
        """Спрощений цикл керування воротами (відкриття, очікування, закриття)."""
        logger.info(f"[{cam_type.upper()}] Початок циклу. Команда на відкриття воріт.")
        self.gate_controller.open_gate()

        if not self.sensor_manager.reed_switch.wait_for_open(timeout=self.config['reed_open_timeout_s']):
            logger.error(f"[{cam_type.upper()}] Ворота не відкрилися (немає сигналу від геркона). Цикл перервано.")
            return

        logger.info(f"[{cam_type.upper()}] Ворота відкрито. Очікування проїзду.")

        # Використовуємо єдиний тайм-аут для проїзду
        passage_timeout = self.config['passage_timeout_s']
        time.sleep(passage_timeout)
        logger.info(f"[{cam_type.upper()}] Час на проїзд ({passage_timeout}с) вичерпано. Початок закриття.")

        self.gate_controller.close_gate()

        # --- СПРОЩЕНА ЛОГІКА ---
        # Замість циклу "гарячого очікування" тепер проста затримка
        gate_travel_time = self.config['gate_travel_time_s']
        logger.info(f"[{cam_type.upper()}] Очікування {gate_travel_time}с, поки ворота зачиняться...")
        time.sleep(gate_travel_time)
        # ------------------------

        logger.info(f"[{cam_type.upper()}] Час ходу воріт минув. Вважаємо ворота закритими.")
        logger.info(f"[{cam_type.upper()}] Повний цикл завершено.")

    def handle_request(self, cam_type: str, plate_text: str, photo_path: Optional[str] = None):
        """
        Обробляє один повний випадок розпізнавання: авторизація, сповіщення, керування воротами.
        """
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
        """Цикл, що постійно опитує камеру на наявність автомобіля."""
        while not self.shutdown_event.is_set():
            if not self.system_busy.locked() and camera and camera.is_initialized_successfully:
                try:
                    frame = camera.capture_array()
                    if frame is not None:
                        if self.cv_processor.detect_vehicle_in_frame(frame, camera_type=cam_type):
                            logger.info(f"[{cam_type.upper()}] Виявлено рух. Запуск розпізнавання номера...")

                            plate, photo_path = self.cv_processor.get_plate_number_from_image(
                                frame, cam_type, save_intermediate_steps=True, save_path_prefix="captures"
                            )

                            if plate:
                                self.handle_request(cam_type, plate, photo_path)
                except Exception as e:
                    logger.error(f"Помилка в циклі опитування камери '{cam_type}': {e}", exc_info=True)
                    time.sleep(5)

            time.sleep(self.config.get('poll_interval_idle_s', 1.0))

    def start(self, shutdown_event: threading.Event):
        """Запускає потоки моніторингу для камер в'їзду та виїзду."""
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
        """Зупиняє потоки обробки."""
        self.is_running = False
        if self.entry_thread:
            self.entry_thread.join()
        if self.exit_thread:
            self.exit_thread.join()
        logger.info("Потоки обробки зупинено.")
