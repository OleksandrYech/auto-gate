# main.py
import logging
import os
import signal
import sys
import threading
import time
from typing import Dict, Any

from utils.logger_config import setup_global_logging
from core.camera_manager import CameraManager
from core.sensors_manager import SensorManager
from core.sheet_handler import SheetHandler
from core.gate_controller import GateController
from core.cv_processor import CVProcessor
from core.vehicle_event_handler import VehicleEventHandler

# --- Нова конфігурація ---
VEH_CONFIG: Dict[str, Any] = {
    "sheets_antiduplicate_delay_s": 60,  # Анти-дублікат для логів
    "passage_timeout_s": 20,  # Час на проїзд після відкриття
    "interrupted_passage_timeout_s": 30,  # Збільшений час після переривання
    "gate_travel_time_s": 15,  # Час повного ходу воріт для "гарячого очікування"
    "reed_open_timeout_s": 15,  # Таймаут очікування сигналу від геркона
    "poll_interval_idle_s": 1.0  # Інтервал перевірки камер у режимі простою
}

# --- Інші константи ---
REED_SWITCH_PIN: int = 22
OPEN_RELAY_PIN: int = 17
CLOSE_RELAY_PIN: int = 27


# ... (решта констант: шляхи до моделей, конфіги камер і т.д. залишаються без змін)

# (Код signal_handler та main_application залишається майже без змін,
#  змінюються лише виклики конструкторів)

def main_application():
    logger = logging.getLogger(__name__)
    # ... (ініціалізація cam_manager, sheet_hndl, cv_proc залишається такою ж)

    try:
        # ... (ініціалізація камер, Google Sheets, CVProcessor)

        logger.info("Ініціалізація SensorManager...")
        sensor_mgr = SensorManager(reed_pin=REED_SWITCH_PIN)
        if not sensor_mgr.reed_switch:
            logger.critical("Не вдалося ініціалізувати геркон. Завершення.")
            return

        logger.info("Ініціалізація GateController...")
        gate_ctrl = GateController(
            open_relay_pin=OPEN_RELAY_PIN,
            close_relay_pin=CLOSE_RELAY_PIN,
            relay_pulse_duration_s=0.5
        )
        if not gate_ctrl.relays_initialized:
            logger.critical("Реле в GateController не ініціалізовано. Завершення.")
            return

        logger.info("Ініціалізація VehicleEventHandler...")
        vehicle_event_hndl = VehicleEventHandler(
            camera_entry=cam_manager.get_entry_camera(),
            camera_exit=cam_manager.get_exit_camera(),
            sensor_manager=sensor_mgr,
            sheet_handler=sheet_hndl,
            cv_processor=cv_proc,
            gate_controller=gate_ctrl,
            config=VEH_CONFIG
        )

        logger.info("Запуск основних потоків обробки...")
        shutdown_event = threading.Event()

        # Обробник сигналів
        def signal_handler(sig, frame):
            logger.warning(f"Отримано сигнал {signal.Signals(sig).name}. Завершення роботи...")
            shutdown_event.set()

        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)

        vehicle_event_hndl.start(shutdown_event)

        logger.info("Система запущена. Натисніть Ctrl+C для завершення.")
        shutdown_event.wait()  # Очікуємо на сигнал завершення

    except Exception as e:
        logger.critical(f"Неперехоплена помилка в main_application: {e}", exc_info=True)
    finally:
        logger.info("Початок процедури коректного завершення роботи...")
        if 'vehicle_event_hndl' in locals() and vehicle_event_hndl:
            vehicle_event_hndl.stop()
        if 'gate_ctrl' in locals() and gate_ctrl:
            gate_ctrl.cleanup()
        if 'sensor_mgr' in locals() and sensor_mgr:
            sensor_mgr.cleanup()
        if 'cam_manager' in locals() and cam_manager:
            cam_manager.close_all_cameras()
        logger.info("Система завершила роботу.")


if __name__ == "__main__":
    setup_global_logging()  # Припускаємо, що цей файл існує
    main_application()