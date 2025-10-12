# core/vehicle_event_handler.py
import logging
import time
import threading
from enum import Enum, auto
from typing import Optional, Dict, Any
from datetime import datetime

from .camera_manager import CameraController
from .sensors_manager import SensorManager
from .sheet_handler import SheetHandler
from .cv_processor import CVProcessor
from .gate_controller import GateController
from .settings_manager import SettingsManager

try:
    from bot.telegram_notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None

logger = logging.getLogger(__name__)


class SystemState(Enum):
    IDLE = auto()
    PROCESSING = auto()
    OPENING_GATE = auto()
    WAITING_FOR_ROI_CLEAR = auto()
    CLOSING_TIMER_ACTIVE = auto()


class VehicleEventHandler:
    def __init__(self,
                 camera_entry: Optional[CameraController],
                 camera_exit: Optional[CameraController],
                 sensor_manager: SensorManager,
                 sheet_handler: SheetHandler,
                 cv_processor: CVProcessor,
                 gate_controller: GateController,
                 config: Dict[str, Any],
                 notifier: Optional[TelegramNotifier] = None,
                 settings_manager: Optional[SettingsManager] = None):

        self.camera_entry = camera_entry
        self.camera_exit = camera_exit
        self.sensor_manager = sensor_manager
        self.sheet_handler = sheet_handler
        self.cv_processor = cv_processor
        self.gate_controller = gate_controller
        self.notifier = notifier
        self.settings_manager = settings_manager
        self.config = config
        self.last_detection_times: Dict[str, float] = {}
        self.cooldown_period_s = config.get("sheets_antiduplicate_delay_s", 60)
        self.state = SystemState.IDLE
        self.state_lock = threading.Lock()
        self.last_roi_clear_time: Optional[float] = None
        self.is_running = False
        self.shutdown_event = threading.Event()
        self.entry_thread: Optional[threading.Thread] = None

    def _set_state(self, new_state: SystemState):
        with self.state_lock:
            if self.state != new_state:
                logger.info(f"Зміна стану: {self.state.name} -> {new_state.name}")
                self.state = new_state

    def _polling_loop(self, camera: Optional[CameraController], cam_type: str):
        while not self.shutdown_event.is_set():
            time.sleep(self.config.get('poll_interval_idle_s', 1.0))
            if not (camera and camera.is_initialized_successfully):
                logger.error(f"Камера '{cam_type}' недоступна.")
                time.sleep(10)
                continue

            try:
                frame = camera.capture_array()
                if frame is None: continue
                vehicle_in_roi = bool(self.cv_processor.detect_vehicle_in_frame(frame, camera_type=cam_type))

                if self.state == SystemState.IDLE and vehicle_in_roi and cam_type == 'entry':
                    self._set_state(SystemState.PROCESSING)
                    plate, photo_path = self.cv_processor.get_plate_number_from_image(
                        frame, cam_type, save_intermediate_steps=True, save_path_prefix="captures"
                    )
                    if not plate:
                        self._set_state(SystemState.IDLE)
                        continue

                    current_time = time.monotonic()
                    if current_time - self.last_detection_times.get(plate, 0) < self.cooldown_period_s:
                        logger.info(f"[ENTRY] Номер '{plate}' розпізнано повторно. Ігноруємо.")
                        self._set_state(SystemState.IDLE)
                        continue

                    self.last_detection_times[plate] = current_time
                    is_authorized = self.sheet_handler.find_vehicle_and_update_entry_time(plate)
                    status = "Авторизовано" if is_authorized else "НЕ Авторизовано"

                    if self.settings_manager and self.settings_manager.are_notifications_enabled():
                        if self.notifier and photo_path:
                            # --- Переконуємося, що викликаємо правильний метод ---
                            self.notifier.send_notification_to_authorized(
                                photo_path,
                                plate,
                                datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
                                status
                            )

                    if is_authorized:
                        logger.info(f"[ENTRY] Номер '{plate}' авторизовано. Відчиняємо ворота.")
                        self._set_state(SystemState.OPENING_GATE)
                        self.gate_controller.open_gate()
                        time.sleep(self.config['gate_travel_time_s'])
                        self._set_state(SystemState.WAITING_FOR_ROI_CLEAR)
                    else:
                        logger.warning(f"[ENTRY] Номер '{plate}' НЕ авторизовано.")
                        self.sheet_handler.add_unauthorized_attempt(plate)
                        self._set_state(SystemState.IDLE)

            except Exception as e:
                logger.error(f"Критична помилка в циклі опитування камери '{cam_type}': {e}", exc_info=True)
                self._set_state(SystemState.IDLE)
                time.sleep(5)

    def start(self, shutdown_event: threading.Event):
        if self.is_running: return
        self.is_running = True
        self.shutdown_event = shutdown_event
        if self.camera_entry:
            self.entry_thread = threading.Thread(target=self._polling_loop, args=(self.camera_entry, 'entry'),
                                                 name="EntryThread")
            self.entry_thread.start()
        logger.info("Потоки обробки подій запущено.")

    def stop(self):
        if not self.is_running: return
        self.is_running = False
        if self.entry_thread and self.entry_thread.is_alive(): self.entry_thread.join()
        logger.info("Потоки обробки подій зупинено.")