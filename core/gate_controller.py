# core/gate_controller.py
import time
import logging
import threading
from typing import Optional

try:
    from gpiozero import OutputDevice

    GPIOZERO_AVAILABLE = True
except ImportError:
    GPIOZERO_AVAILABLE = False


    # Мок-клас для тестування на ПК
    class OutputDevice:
        def __init__(self, pin, active_high=True, initial_value=False):
            self.pin = pin;
            self._active_high = active_high;
            self._value = initial_value
            self._is_active = False
            self.logger = logging.getLogger(f"MockOutputDevice.Pin{pin}")
            self.logger.info(f"Мок OutputDevice створено для піна {pin}")

        def on(self): self._is_active = True; self.logger.info(f"Мок pin {self.pin} -> УВІМКНЕНО")

        def off(self): self._is_active = False; self.logger.info(f"Мок pin {self.pin} -> ВИМКНЕНО")

        @property
        def is_active(self): return self._is_active

        def close(self): self.logger.info(f"Мок pin {self.pin} закрито.")

logger = logging.getLogger(__name__)

DEFAULT_OPEN_RELAY_PIN = 17
DEFAULT_CLOSE_RELAY_PIN = 27
DEFAULT_RELAY_PULSE_DURATION_S = 0.5
DEFAULT_AUTO_CLOSE_TIMEOUT_S = 30
DEFAULT_CLOSING_OBSTRUCTION_THRESHOLD_M = 0.3
DEFAULT_REED_CONFIRMATION_TIMEOUT_S = 5


class GateController:
    def __init__(self,
                 sensor_manager_instance,
                 open_relay_pin: int = DEFAULT_OPEN_RELAY_PIN,
                 close_relay_pin: int = DEFAULT_CLOSE_RELAY_PIN,
                 relay_pulse_duration_s: float = DEFAULT_RELAY_PULSE_DURATION_S,
                 auto_close_timeout_s: float = DEFAULT_AUTO_CLOSE_TIMEOUT_S,
                 closing_obstruction_threshold_m: float = DEFAULT_CLOSING_OBSTRUCTION_THRESHOLD_M,
                 reed_confirmation_timeout_s: float = DEFAULT_REED_CONFIRMATION_TIMEOUT_S):

        self.sensor_manager = sensor_manager_instance
        self.open_relay_pin = open_relay_pin
        self.close_relay_pin = close_relay_pin
        self.relay_pulse_duration_s = relay_pulse_duration_s
        self.default_auto_close_timeout_s = auto_close_timeout_s
        self.closing_obstruction_threshold_m = closing_obstruction_threshold_m
        self.reed_confirmation_timeout_s = reed_confirmation_timeout_s

        self._logger = logging.getLogger(f"{__name__}.GateController_O{open_relay_pin}C{close_relay_pin}")

        self.relays_initialized = False
        self.open_relay: Optional[OutputDevice] = None
        self.close_relay: Optional[OutputDevice] = None

        if GPIOZERO_AVAILABLE:
            try:
                # Ініціалізація з active_high=False та initial_value=True для low-level реле
                self.open_relay = OutputDevice(self.open_relay_pin, active_high=False, initial_value=True)
                self.open_relay.off()  # Явно встановлюємо HIGH (неактивний стан)
                self._logger.info(
                    f"Реле OPEN на GPIO{self.open_relay_pin} (active_low) ініціалізовано та вимкнено. is_active: {self.open_relay.is_active}")

                self.close_relay = OutputDevice(self.close_relay_pin, active_high=False, initial_value=True)
                self.close_relay.off()  # Явно встановлюємо HIGH (неактивний стан)
                self._logger.info(
                    f"Реле CLOSE на GPIO{self.close_relay_pin} (active_low) ініціалізовано та вимкнено. is_active: {self.close_relay.is_active}")

                self.relays_initialized = True
            except Exception as e:
                self._logger.critical(f"Критична помилка ініціалізації реле gpiozero: {e}", exc_info=True)
        else:  # Використання мок-об'єктів, якщо gpiozero недоступний
            self._logger.warning("Бібліотека gpiozero недоступна. Використовуються мок-об'єкти для реле.")
            self.open_relay = OutputDevice(self.open_relay_pin, active_high=False, initial_value=True)
            self.open_relay.off()
            self.close_relay = OutputDevice(self.close_relay_pin, active_high=False, initial_value=True)
            self.close_relay.off()
            self.relays_initialized = True

        self._auto_close_timer: Optional[threading.Timer] = None
        self._close_procedure_interrupted: bool = False
        self._lock = threading.Lock()

        self._commanded_state: str = "IDLE"
        self.gate_state: str = self._determine_state_from_reed_switch()

        self._logger.info(f"GateController ініціалізовано. Початковий стан (геркон): {self.gate_state}")
        # ... (решта логів з __init__)

    def _determine_state_from_reed_switch(self) -> str:
        # ... (без змін від попередньої версії) ...
        if not self.sensor_manager or not hasattr(self.sensor_manager, 'reed_switch') or \
                not self.sensor_manager.reed_switch or \
                (GPIOZERO_AVAILABLE and getattr(self.sensor_manager.reed_switch, '_device', None) is None):
            self._logger.warning("Геркон недоступний або не ініціалізований. Стан воріт невідомий.")
            return "UNKNOWN_SENSOR_UNAVAILABLE"
        try:
            if self.sensor_manager.reed_switch.are_gates_open:
                return "OPEN"
            elif self.sensor_manager.reed_switch.are_gates_closed:
                return "CLOSED"
            else:
                return "BETWEEN_STATES"
        except Exception as e:
            self._logger.error(f"Помилка читання стану геркона: {e}", exc_info=True)
            return "UNKNOWN_SENSOR_READ_ERROR"

    def _activate_relay_pulse(self, relay_device: Optional[OutputDevice], action_name: str):
        if not self.relays_initialized or not relay_device:
            self._logger.error(f"Реле для '{action_name}' не ініціалізовано або відсутнє. Дію скасовано.")
            return

        pin_number_str = str(getattr(relay_device, 'pin', 'N/A'))
        if GPIOZERO_AVAILABLE and hasattr(relay_device.pin, 'number'):  # Для реального gpiozero
            pin_number_str = str(relay_device.pin.number)

        self._logger.debug(f"Подача імпульсу на реле '{action_name}' (GPIO{pin_number_str}) "
                           f"тривалістю: {self.relay_pulse_duration_s} с.")
        try:
            relay_device.on()  # Активує реле (пін LOW для active_high=False)
            time.sleep(self.relay_pulse_duration_s)
        except Exception as e:
            self._logger.error(f"Помилка під час активації або очікування реле '{action_name}': {e}", exc_info=True)
        finally:
            try:
                # Завжди намагаємося вимкнути реле після імпульсу
                relay_device.off()  # Деактивує реле (пін HIGH)
                self._logger.info(f"Реле '{action_name}' (GPIO{pin_number_str}) команда OFF надіслана після імпульсу.")
            except Exception as e_off:
                self._logger.error(f"Помилка під час деактивації реле '{action_name}' у блоці finally: {e_off}",
                                   exc_info=True)

    def open_gate(self) -> bool:
        if not self.relays_initialized:
            self._logger.error("Неможливо відкрити ворота: реле не ініціалізовано.")
            return False
        current_actual_state = self._determine_state_from_reed_switch()
        if current_actual_state == "OPEN":
            self._logger.info("Ворота вже відкриті (геркон). Команда ВІДКРИТИ ігнорується.")
            self._commanded_state = "IDLE";
            self.gate_state = "OPEN"
            return True
        with self._lock:
            if self._auto_close_timer and self._auto_close_timer.is_alive():
                self._logger.info("Команда ВІДКРИТИ: скасування активного таймера авто-закриття.")
                self._auto_close_timer.cancel();
                self._auto_close_timer = None
            self._close_procedure_interrupted = False
        self._logger.info("КОМАНДА: Відкриття воріт.");
        self._activate_relay_pulse(self.open_relay, "OPEN")
        self._commanded_state = "OPENING";
        self.gate_state = "OPENING"
        if self.reed_confirmation_timeout_s > 0 and self.sensor_manager and \
                hasattr(self.sensor_manager, 'reed_switch') and self.sensor_manager.reed_switch and \
                ((GPIOZERO_AVAILABLE and getattr(self.sensor_manager.reed_switch, '_device', None) is not None) or \
                 (not GPIOZERO_AVAILABLE and self.sensor_manager.reed_switch)):  # Перевірка для моків теж
            self._logger.debug(
                f"Очікування підтвердження відкриття від геркона ({self.reed_confirmation_timeout_s}с)...")
            if self.sensor_manager.reed_switch.wait_for_open(timeout=self.reed_confirmation_timeout_s):
                if self.sensor_manager.reed_switch.are_gates_open:
                    self.gate_state = "OPEN";
                    self._commanded_state = "IDLE"
                    self._logger.info("Ворота підтверджено відкритими (геркон, внутрішня перевірка GateController).")
                else:
                    self.gate_state = self._determine_state_from_reed_switch()
                    self._logger.warning("wait_for_open успішний, але are_gates_open - ні. Стан: " + self.gate_state)
            else:
                self.gate_state = "MOVEMENT_TIMEOUT_OPEN"
                self._logger.warning(
                    f"Таймаут ({self.reed_confirmation_timeout_s}с) очікування відкриття воріт (геркон).")
        return True

    def is_passage_clear_for_closing(self) -> bool:
        # ... (без змін від попередньої версії) ...
        passage_sensor = getattr(self.sensor_manager, 'ultrasonic_sensor_entry', None)
        if not passage_sensor or (GPIOZERO_AVAILABLE and getattr(passage_sensor, '_sensor', None) is None):
            self._logger.warning("УЗД для перевірки проїзду недоступний. З метою безпеки, проїзд НЕ вільний.")
            return False
        try:
            distance = passage_sensor.get_distance()
        except Exception as e:
            self._logger.error(f"Помилка отримання даних з УЗД: {e}", exc_info=True); return False
        if distance is None: self._logger.warning(
            "УЗД повернув None. З метою безпеки, проїзд НЕ вільний."); return False
        if distance == float('inf'): self._logger.debug(
            f"Проїзд вільний для закриття. Відстань: > max_dist."); return True
        if distance < self.closing_obstruction_threshold_m:
            self._logger.info(
                f"Проїзд ЗАБЛОКОВАНО. Перешкода на {distance:.2f}м (поріг: {self.closing_obstruction_threshold_m}м).")
            return False
        else:
            self._logger.debug(f"Проїзд вільний для закриття. Відстань: {distance:.2f}м."); return True

    def close_gate(self) -> bool:
        if not self.relays_initialized:
            self._logger.error("Неможливо закрити ворота: реле не ініціалізовано.");
            return False
        current_actual_state = self._determine_state_from_reed_switch()
        if current_actual_state == "CLOSED":
            self._logger.info("Ворота вже закриті (геркон). Команда ЗАКРИТИ ігнорується.")
            self._commanded_state = "IDLE";
            self.gate_state = "CLOSED"
            return True
        with self._lock:
            if self._close_procedure_interrupted:
                self._logger.info("Процедуру закриття було перервано. Ворота НЕ закриватимуться.")
                self._close_procedure_interrupted = False
                self.gate_state = self._determine_state_from_reed_switch();
                return False
            if not self.is_passage_clear_for_closing():
                self._logger.warning("КОМАНДУ ВІДХИЛЕНО: Неможливо закрити ворота, проїзд заблоковано.")
                self.gate_state = "OBSTRUCTED";
                return False
        self._logger.info("КОМАНДА: Закриття воріт (проїзд вільний).")
        self._activate_relay_pulse(self.close_relay, "CLOSE")
        self._commanded_state = "CLOSING";
        self.gate_state = "CLOSING"
        if self.reed_confirmation_timeout_s > 0 and self.sensor_manager and \
                hasattr(self.sensor_manager, 'reed_switch') and self.sensor_manager.reed_switch and \
                ((GPIOZERO_AVAILABLE and getattr(self.sensor_manager.reed_switch, '_device', None) is not None) or \
                 (not GPIOZERO_AVAILABLE and self.sensor_manager.reed_switch)):
            self._logger.debug(
                f"Очікування підтвердження закриття від геркона ({self.reed_confirmation_timeout_s}с)...")
            if self.sensor_manager.reed_switch.wait_for_close(timeout=self.reed_confirmation_timeout_s):
                if self.sensor_manager.reed_switch.are_gates_closed:
                    self.gate_state = "CLOSED";
                    self._commanded_state = "IDLE"
                    self._logger.info("Ворота підтверджено закритими (геркон, внутрішня перевірка GateController).")
                else:
                    self.gate_state = self._determine_state_from_reed_switch()
                    self._logger.warning("wait_for_close успішний, але are_gates_closed - ні. Стан: " + self.gate_state)
            else:
                self.gate_state = "MOVEMENT_TIMEOUT_CLOSE"
                self._logger.warning(
                    f"Таймаут ({self.reed_confirmation_timeout_s}с) очікування закриття воріт (геркон).")
        return True

    def _auto_close_gate_callback(self):
        # ... (без змін) ...
        self._logger.info("Таймер авто-закриття спрацював. Спроба закрити ворота.")
        self.close_gate()

    def start_auto_close_timer(self, timeout_s: Optional[float] = None):
        # ... (без змін) ...
        with self._lock:
            if self._auto_close_timer and self._auto_close_timer.is_alive():
                self._logger.debug("Скасування існуючого таймера авто-закриття.")
                self._auto_close_timer.cancel()
            self._close_procedure_interrupted = False
            effective_timeout = timeout_s if timeout_s is not None else self.default_auto_close_timeout_s
            if effective_timeout <= 0:
                self._logger.info("Тайм-аут авто-закриття <= 0. Таймер не запускається.")
                return
            self._logger.info(f"Запуск таймера авто-закриття на {effective_timeout} секунд.")
            self._auto_close_timer = threading.Timer(effective_timeout, self._auto_close_gate_callback)
            self._auto_close_timer.daemon = True
            self._auto_close_timer.start()

    def cancel_auto_close_timer(self):
        # ... (без змін) ...
        with self._lock:
            if self._auto_close_timer and self._auto_close_timer.is_alive():
                self._logger.info("Таймер авто-закриття скасовано вручну.")
                self._auto_close_timer.cancel()
                self._auto_close_timer = None
            else:
                self._logger.debug("Немає активного таймера авто-закриття для скасування.")

    def interrupt_closing_procedure(self):
        # ... (без змін) ...
        with self._lock:
            if not self._close_procedure_interrupted:
                self._logger.info("Отримано сигнал ПЕРЕРИВАННЯ для процедури закриття воріт.")
            self._close_procedure_interrupted = True
            if self._auto_close_timer and self._auto_close_timer.is_alive():
                self._logger.info("Скасування активного таймера авто-закриття через переривання.")
                self._auto_close_timer.cancel()
                self._auto_close_timer = None
            self.gate_state = self._determine_state_from_reed_switch()
            self._commanded_state = "IDLE"

    def get_current_gate_state(self) -> str:
        # ... (без змін) ...
        reed_state = self._determine_state_from_reed_switch()
        if reed_state in ["OPEN", "CLOSED"]:
            self.gate_state = reed_state
            if (self.gate_state == "OPEN" and self._commanded_state == "OPENING") or \
                    (self.gate_state == "CLOSED" and self._commanded_state == "CLOSING"):
                self._commanded_state = "IDLE"
            return self.gate_state
        if self.gate_state == "OBSTRUCTED": return "OBSTRUCTED"
        if self._commanded_state in ["OPENING", "CLOSING"]: return self._commanded_state
        if reed_state == "BETWEEN_STATES": return "BETWEEN_STATES"
        return self.gate_state

    def cleanup(self):
        # ... (без змін) ...
        self._logger.info("Очищення ресурсів GateController...")
        self.cancel_auto_close_timer()
        if self.relays_initialized:
            if hasattr(self, 'open_relay') and self.open_relay:
                try:
                    self.open_relay.off(); self.open_relay.close()  # Гарантоване вимкнення перед закриттям
                except Exception as e:
                    self._logger.error(f"Помилка при закритті open_relay: {e}")
            if hasattr(self, 'close_relay') and self.close_relay:
                try:
                    self.close_relay.off(); self.close_relay.close()
                except Exception as e:
                    self._logger.error(f"Помилка при закритті close_relay: {e}")
        self._logger.info("Очищення ресурсів GateController завершено.")

    def __del__(self):
        self.cleanup()


# --- Блок для тестування модуля ---
if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s',
            handlers=[logging.StreamHandler()]
        )
    from unittest.mock import MagicMock

    logger_test = logging.getLogger(f"{__name__}_Test")
    logger_test.info("--- Тестування модуля gate_controller.py (з моками) ---")



    class MockGpioDevice:
        def __init__(self, pin, pull_up=None):
            self.pin = pin
            self._value = 1 if pull_up else 0
            self.is_active_state = False
            if pull_up and self._value == 0: self.is_active_state = True
            self.logger = logging.getLogger(f"MockGpioDevice.Pin{pin}")
            self.logger.info(
                f"Мок GPIO пристрою на піні {pin} створено. pull_up={pull_up}, value={self._value}, is_active={self.is_active_state}")

        @property
        def is_active(self):
            return self.is_active_state

        def _update_active_state(self):
            self.is_active_state = (self._value == 0)

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, new_val):
            self._value = new_val; self._update_active_state()

        def close(self):
            self.logger.debug(f"MockGpioDevice pin {self.pin} closed")

        def wait_for_active(self, timeout):
            self.logger.debug(f"Mock: wait_for_active on pin {self.pin}")
            if self.is_active_state: return True
            if timeout: time.sleep(min(0.1, timeout / 2))
            return self.is_active_state

        def wait_for_inactive(self, timeout):
            self.logger.debug(f"Mock: wait_for_inactive on pin {self.pin}")
            if not self.is_active_state: return True
            if timeout: time.sleep(min(0.1, timeout / 2))
            return not self.is_active_state


    class MockReedSwitch:
        def __init__(self, pin_number=22, name="MockReed"):
            self.name = name
            self._device = MockGpioDevice(pin_number, pull_up=True)
            self._logger = logging.getLogger(f"MockSensor.{self.name}")
            self.set_gates_closed()

        @property
        def are_gates_open(self):
            return self._device.is_active if self._device else None

        @property
        def are_gates_closed(self):
            return not self._device.is_active if self._device else None

        def wait_for_open(self, timeout=None):
            return self._device.wait_for_active(timeout) if self._device else False

        def wait_for_close(self, timeout=None):
            return self._device.wait_for_inactive(timeout) if self._device else False

        def set_gates_open(self):
            if self._device: self._device.value = 0; self._logger.info(f"Мок '{self.name}': ВСТАНОВЛЕНО ВІДКРИТО")

        def set_gates_closed(self):
            if self._device: self._device.value = 1; self._logger.info(f"Мок '{self.name}': ВСТАНОВЛЕНО ЗАКРИТО")


    class MockUltrasonicSensor:
        def __init__(self, name="MockUS"):
            self.name = name
            self._sensor = True
            self.distance = 2.0
            self._logger = logging.getLogger(f"MockSensor.{self.name}")

        def get_distance(self): self._logger.debug(
            f"Мок '{self.name}': повертає відстань {self.distance}м"); return self.distance

        def set_distance(self, dist): self.distance = dist; self._logger.info(
            f"Мок '{self.name}': встановлено відстань {dist}м")


    mock_sensor_manager = MagicMock()
    mock_sensor_manager.reed_switch = MockReedSwitch(
        pin_number=DEFAULT_REED_CONFIRMATION_TIMEOUT_S)  # Використовуємо константу як пін для мока
    mock_sensor_manager.ultrasonic_sensor_entry = MockUltrasonicSensor()

    gate_ctrl = GateController(
        sensor_manager_instance=mock_sensor_manager,
        open_relay_pin=20, close_relay_pin=21,
        auto_close_timeout_s=2,
        reed_confirmation_timeout_s=0.5  # Коротший таймаут для швидких тестів
    )

    if not gate_ctrl.relays_initialized:
        logger_test.error("Реле не були ініціалізовані, тестування неможливе.")
    else:
        logger_test.info(f"Початковий стан воріт: {gate_ctrl.get_current_gate_state()}")

        logger_test.info("\nТест: Відкриття воріт...")


        def simulate_open_reed():
            time.sleep(0.1); mock_sensor_manager.reed_switch.set_gates_open()


        threading.Thread(target=simulate_open_reed).start()
        gate_ctrl.open_gate()
        time.sleep(gate_ctrl.reed_confirmation_timeout_s + 0.2)
        logger_test.info(f"Стан після команди відкрити: {gate_ctrl.get_current_gate_state()}")

        logger_test.info("\nТест: Закриття воріт (проїзд вільний)...")
        mock_sensor_manager.ultrasonic_sensor_entry.set_distance(2.0)


        def simulate_close_reed():
            time.sleep(0.1); mock_sensor_manager.reed_switch.set_gates_closed()


        threading.Thread(target=simulate_close_reed).start()
        gate_ctrl.close_gate()
        time.sleep(gate_ctrl.reed_confirmation_timeout_s + 0.2)
        logger_test.info(f"Стан після команди закрити: {gate_ctrl.get_current_gate_state()}")

        logger_test.info("\nТест: Спроба закрити (проїзд заблоковано 0.1м)...")
        mock_sensor_manager.reed_switch.set_gates_open()
        gate_ctrl.gate_state = "OPEN";
        gate_ctrl._commanded_state = "IDLE"
        mock_sensor_manager.ultrasonic_sensor_entry.set_distance(0.1)
        gate_ctrl.close_gate()
        logger_test.info(f"Стан після спроби закрити з перешкодою: {gate_ctrl.get_current_gate_state()}")

        logger_test.info("\nТест: Авто-закриття...")
        mock_sensor_manager.reed_switch.set_gates_open()
        gate_ctrl.gate_state = "OPEN";
        gate_ctrl._commanded_state = "IDLE"
        mock_sensor_manager.ultrasonic_sensor_entry.set_distance(2.0)
        gate_ctrl.start_auto_close_timer()
        logger_test.info("Таймер авто-закриття запущено. Очікування...")


        def delayed_reed_close_for_auto_timer():
            time.sleep(gate_ctrl.default_auto_close_timeout_s + 0.1)  # Після спрацювання таймера
            mock_sensor_manager.reed_switch.set_gates_closed()


        threading.Thread(target=delayed_reed_close_for_auto_timer).start()
        time.sleep(gate_ctrl.default_auto_close_timeout_s + gate_ctrl.reed_confirmation_timeout_s + 0.5)
        logger_test.info(f"Стан після авто-закриття: {gate_ctrl.get_current_gate_state()}")

        gate_ctrl.cleanup()
    logger_test.info("--- Тестування модуля gate_controller.py завершено ---")
