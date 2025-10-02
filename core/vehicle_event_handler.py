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
    """Перелік можливих станів системи для керування логікою."""
    IDLE = auto()  # Система вільна і чекає на події
    PROCESSING = auto()  # Виконується розпізнавання (для в'їзду)
    OPENING_GATE = auto()  # Ворота відчиняються
    WAITING_FOR_ROI_CLEAR = auto()  # Ворота відчинені, чекаємо, поки авто покине зону
    CLOSING_TIMER_ACTIVE = auto()  # Зона вільна, йде відлік часу до закриття


class VehicleEventHandler:
    """
    Керує логікою обробки подій на основі станів системи.
    Реалізує безпечне закриття воріт.
    """
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

        # --- НОВА ЛОГІКА СТАНІВ ---
        self.state = SystemState.IDLE
        self.state_lock = threading.Lock() # Замок для потокобезпечної зміни стану
        self.last_roi_clear_time: Optional[float] = None # Час, коли ROI стала вільною
        self.active_thread: Optional[threading.Thread] = None # Потік для паралельних завдань
        # --- КІНЕЦЬ НОВОЇ ЛОГІКИ ---

        self.is_running = False
        self.shutdown_event = threading.Event()
        self.entry_thread: Optional[threading.Thread] = None
        self.exit_thread: Optional[threading.Thread] = None


    def _set_state(self, new_state: SystemState):
        """Потокобезпечно змінює стан системи."""
        with self.state_lock:
            if self.state != new_state:
                logger.info(f"Зміна стану: {self.state.name} -> {new_state.name}")
                self.state = new_state

    def _recognize_and_log_exit(self, frame: Any, cam_type: str):
        """Паралельна функція для розпізнавання та логування авто на виїзді."""
        plate, _ = self.cv_processor.get_plate_number_from_image(
            frame, cam_type, save_intermediate_steps=True, save_path_prefix="captures"
        )
        plate_text = plate or "UNKNOWN"
        logger.info(f"[{cam_type.upper()}] Розпізнано номер на виїзд: '{plate_text}'.")
        self.sheet_handler.log_vehicle_exit(plate_text)

    def _polling_loop(self, camera: Optional[CameraController], cam_type: str):
        """
        Основний цикл, що керує всією логікою на основі станів.
        """
        while not self.shutdown_event.is_set():
            time.sleep(self.config.get('poll_interval_idle_s', 1.0))

            if not (camera and camera.is_initialized_successfully):
                logger.error(f"Камера '{cam_type}' недоступна. Потік призупинено.")
                time.sleep(10)
                continue

            try:
                # 1. Перевіряємо наявність авто в ROI
                frame = camera.capture_array()
                if frame is None: continue

                vehicle_in_roi = bool(self.cv_processor.detect_vehicle_in_frame(frame, camera_type=cam_type))

                # --- ЛОГІКА КЕРУВАННЯ СТАНАМИ ---

                # Стан: Система вільна
                if self.state == SystemState.IDLE and vehicle_in_roi:

                    # --- ЛОГІКА ДЛЯ ВИЇЗДУ ---
                    if cam_type == 'exit':
                        self._set_state(SystemState.OPENING_GATE)
                        # Негайно відчиняємо ворота
                        self.gate_controller.open_gate()
                        logger.info("[EXIT] Авто на виїзді. Ворота відчиняються. Розпізнавання в фоні.")
                        # В окремому потоці розпізнаємо та логуємо, щоб не блокувати
                        self.active_thread = threading.Thread(target=self._recognize_and_log_exit, args=(frame, cam_type))
                        self.active_thread.start()
                        # Очікуємо, доки ворота відчиняться (за таймером)
                        time.sleep(self.config['gate_travel_time_s'])
                        self._set_state(SystemState.WAITING_FOR_ROI_CLEAR)

                    # --- ЛОГІКА ДЛЯ В'ЇЗДУ ---
                    elif cam_type == 'entry':
                        self._set_state(SystemState.PROCESSING)
                        plate, photo_path = self.cv_processor.get_plate_number_from_image(
                            frame, cam_type, save_intermediate_steps=True, save_path_prefix="captures"
                        )

                        if not plate: # Якщо номер не розпізнано, повертаємось у режим очікування
                            self._set_state(SystemState.IDLE)
                            continue

                        # Перевірка на дублікати
                        current_time = time.monotonic()
                        if current_time - self.last_detection_times.get(plate, 0) < self.cooldown_period_s:
                            logger.info(f"[ENTRY] Номер '{plate}' розпізнано повторно. Ігноруємо.")
                            self._set_state(SystemState.IDLE)
                            continue

                        self.last_detection_times[plate] = current_time

                        # Перевірка авторизації
                        is_authorized = self.sheet_handler.find_vehicle_and_update_entry_time(plate)
                        status = "Авторизовано" if is_authorized else "НЕ Авторизовано"

                        if self.settings_manager and self.settings_manager.are_notifications_enabled():
                            if self.notifier and photo_path:
                                self.notifier.send_notification(photo_path, plate, datetime.now().strftime("%Y-%m-%d %H:%M:%S"), status)

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

                # Стан: Чекаємо, доки авто покине зону
                elif self.state == SystemState.WAITING_FOR_ROI_CLEAR:
                    if not vehicle_in_roi:
                        logger.info("Зона ROI стала вільною. Запускаємо таймер на закриття.")
                        self.last_roi_clear_time = time.monotonic()
                        self._set_state(SystemState.CLOSING_TIMER_ACTIVE)

                # Стан: Йде відлік до закриття
                elif self.state == SystemState.CLOSING_TIMER_ACTIVE:
                    if vehicle_in_roi:
                        # --- ЗАХИСТ! ---
                        logger.warning("Авто з'явилося в ROI під час відліку. Таймер скинуто!")
                        self.last_roi_clear_time = None
                        self._set_state(SystemState.WAITING_FOR_ROI_CLEAR)
                    else:
                        if self.last_roi_clear_time and (time.monotonic() - self.last_roi_clear_time > self.config['passage_timeout_s']):
                            logger.info("Таймер вичерпано. Команда на закриття воріт.")
                            self.gate_controller.close_gate()
                            self.last_roi_clear_time = None
                            self._set_state(SystemState.IDLE)

            except Exception as e:
                logger.error(f"Критична помилка в циклі опитування камери '{cam_type}': {e}", exc_info=True)
                self._set_state(SystemState.IDLE) # Скидаємо стан у разі помилки
                time.sleep(5)

    def start(self, shutdown_event: threading.Event):
        """Запускає потоки для обробки в'їзду та виїзду."""
        if self.is_running:
            return
        logger.info("Запуск потоків обробки подій...")
        self.is_running = True
        self.shutdown_event = shutdown_event

        # Запускаємо окремий потік для кожної камери
        # Вони будуть конкурувати за зміну стану системи, що коректно обробляється
        if self.camera_entry:
            self.entry_thread = threading.Thread(target=self._polling_loop, args=(self.camera_entry, 'entry'), name="EntryThread")
            self.entry_thread.start()

        if self.camera_exit:
            self.exit_thread = threading.Thread(target=self._polling_loop, args=(self.camera_exit, 'exit'), name="ExitThread")
            self.exit_thread.start()

        logger.info("Потоки обробки в'їзду та виїзду успішно запущено.")

    def stop(self):
        """Зупиняє потоки обробки."""
        if not self.is_running:
            return
        logger.info("Зупинка потоків обробки подій...")
        self.is_running = False

        if self.entry_thread and self.entry_thread.is_alive():
            self.entry_thread.join()
        if self.exit_thread and self.exit_thread.is_alive():
            self.exit_thread.join()
        if self.active_thread and self.active_thread.is_alive():
             self.active_thread.join()

        logger.info("Потоки обробки успішно зупинено.")
