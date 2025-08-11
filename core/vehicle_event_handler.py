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
# --- ЗМІНА ---
# Додаємо імпорт нового менеджера налаштувань
from .settings_manager import SettingsManager
# --- КІНЕЦЬ ЗМІНИ ---

try:
    from bot.telegram_notifier import TelegramNotifier
except ImportError:
    TelegramNotifier = None

logger = logging.getLogger(__name__)


class VehicleEventHandler:
    """
    Головний клас, що керує логікою обробки подій, пов'язаних з автомобілями.
    "Диригент" для всіх core-компонентів.
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
                 # --- ЗМІНА ---
                 # Додаємо менеджер налаштувань як опціональний аргумент
                 settings_manager: Optional[SettingsManager] = None
                 # --- КІНЕЦЬ ЗМІНИ ---
                 ):

        self.camera_entry = camera_entry
        self.camera_exit = camera_exit
        self.sensor_manager = sensor_manager
        self.sheet_handler = sheet_handler
        self.cv_processor = cv_processor
        self.gate_controller = gate_controller
        self.notifier = notifier
        # --- ЗМІНА ---
        # Зберігаємо екземпляр менеджера налаштувань
        self.settings_manager = settings_manager
        # --- КІНЕЦЬ ЗМІНИ ---
        self.config = config

        # Словник для відстеження часу останньої детекції номера, щоб уникнути дублікатів
        self.last_detection_times: Dict[str, float] = {}
        self.cooldown_period_s = config.get("sheets_antiduplicate_delay_s", 60)

        self.is_running = False
        self.shutdown_event = threading.Event()
        self.system_busy = threading.Lock() # Блокування, щоб система не обробляла кілька авто одночасно

        self.entry_thread: Optional[threading.Thread] = None
        self.exit_thread: Optional[threading.Thread] = None

    def _process_vehicle_cycle(self, cam_type: str):
        """
        Повний цикл керування воротами: відкриття, очікування проїзду, закриття.
        Використовує геркон для контролю стану воріт.
        """
        logger.info(f"[{cam_type.upper()}] Початок циклу. Команда на відкриття воріт.")
        self.gate_controller.open_gate()

        # Очікуємо, доки геркон не зафіксує, що ворота повністю відкриті
        if not self.sensor_manager.reed_switch.wait_for_open(timeout=self.config['reed_open_timeout_s']):
            logger.error(f"[{cam_type.upper()}] Ворота не відкрилися (немає сигналу від геркона). Цикл перервано.")
            return

        logger.info(f"[{cam_type.upper()}] Ворота відкрито. Очікування проїзду ({self.config['passage_timeout_s']}с).")
        time.sleep(self.config['passage_timeout_s'])

        logger.info(f"[{cam_type.upper()}] Час на проїзд вичерпано. Початок закриття.")
        self.gate_controller.close_gate()

        # Очікуємо, доки геркон не розімкнеться (ворота почали рух із повністю відкритого стану)
        if self.sensor_manager.reed_switch.wait_for_closed(timeout=self.config['gate_travel_time_s']):
            logger.info(f"[{cam_type.upper()}] Геркон розімкнувся. Ворота зачиняються або вже зачинені.")
        else:
            logger.warning(f"[{cam_type.upper()}] Не отримано сигнал про початок закриття від геркона.")

        logger.info(f"[{cam_type.upper()}] Повний цикл роботи з воротами завершено.")

    def handle_request(self, cam_type: str, plate_text: str, photo_path: Optional[str] = None):
        """
        Обробляє запит на проїзд: перевіряє авторизацію, надсилає сповіщення, керує воротами.
        """
        if self.system_busy.locked():
            logger.warning(f"[{cam_type.upper()}] Система зайнята. Запит для '{plate_text}' ігнорується.")
            return

        with self.system_busy:
            if cam_type == 'entry':
                is_authorized = self.sheet_handler.find_vehicle_and_update_entry_time(plate_text)
                status = "Авторизовано" if is_authorized else "НЕ Авторизовано"

                # --- ЗМІНА ---
                # Перевіряємо, чи увімкнені сповіщення, перед їх надсиланням
                if self.settings_manager and self.settings_manager.are_notifications_enabled():
                    if self.notifier and photo_path:
                        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                        self.notifier.send_notification(photo_path, plate_text, timestamp, status)
                else:
                    logger.info("Сповіщення в Telegram вимкнені користувачем, надсилання пропущено.")
                # --- КІНЕЦЬ ЗМІНИ ---

                if is_authorized:
                    logger.info(f"[ENTRY] Номер '{plate_text}' авторизовано. Запуск циклу воріт.")
                    self._process_vehicle_cycle(cam_type)
                else:
                    logger.warning(f"[ENTRY] Номер '{plate_text}' НЕ авторизовано. Запис у лог неавторизованих.")
                    self.sheet_handler.add_unauthorized_attempt(plate_text)

            elif cam_type == 'exit':
                logger.info(f"[EXIT] Автомобіль на виїзд ('{plate_text}'). Запуск циклу воріт.")
                self.sheet_handler.log_vehicle_exit(plate_text or "UNKNOWN")
                self._process_vehicle_cycle(cam_type)

    def _polling_loop(self, camera: Optional[CameraController], cam_type: str):
        """
        Основний цикл, що в окремому потоці опитує камеру,
        шукає автомобілі та запускає їх обробку.
        """
        while not self.shutdown_event.is_set():
            if self.system_busy.locked():
                time.sleep(self.config.get('poll_interval_idle_s', 1.0))
                continue

            if not (camera and camera.is_initialized_successfully):
                logger.error(f"Камера '{cam_type}' недоступна. Потік буде призупинено на 10 секунд.")
                time.sleep(10)
                continue

            try:
                frame = camera.capture_array()
                if frame is None:
                    time.sleep(self.config.get('poll_interval_idle_s', 1.0))
                    continue

                # Спочатку шукаємо тільки автомобіль, це швидка операція
                if self.cv_processor.detect_vehicle_in_frame(frame, camera_type=cam_type):
                    logger.debug(f"[{cam_type.upper()}] Виявлено рух у ROI. Запуск повного розпізнавання...")

                    # Якщо авто є, запускаємо повний (і повільний) процес розпізнавання
                    plate, photo_path = self.cv_processor.get_plate_number_from_image(
                        frame, cam_type, save_intermediate_steps=True, save_path_prefix="captures"
                    )

                    if plate:
                        current_time = time.monotonic()
                        last_seen = self.last_detection_times.get(plate)

                        # Перевіряємо, чи не бачили ми цей же номер нещодавно
                        if last_seen and (current_time - last_seen < self.cooldown_period_s):
                            logger.info(f"Номер '{plate}' розпізнано повторно протягом {self.cooldown_period_s}с. Ігноруємо.")
                            continue

                        # Якщо номер новий, оновлюємо час і передаємо на обробку
                        self.last_detection_times[plate] = current_time
                        logger.info(f"Нова детекція для номера '{plate}'. Передача в обробник.")
                        self.handle_request(cam_type, plate, photo_path)

            except Exception as e:
                logger.error(f"Помилка в циклі опитування камери '{cam_type}': {e}", exc_info=True)
                time.sleep(5) # Пауза перед наступною спробою у разі помилки

            time.sleep(self.config.get('poll_interval_idle_s', 1.0))

    def start(self, shutdown_event: threading.Event):
        """Запускає потоки для обробки в'їзду та виїзду."""
        if self.is_running:
            return
        logger.info("Запуск потоків обробки подій...")
        self.is_running = True
        self.shutdown_event = shutdown_event

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

        logger.info("Потоки обробки успішно зупинено.")
