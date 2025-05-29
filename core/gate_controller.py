# core/gate_controller.py
import time
import logging
import threading
from typing import Optional  # Додано для Optional

try:
    from gpiozero import OutputDevice

    GPIOZERO_AVAILABLE = True
except ImportError:
    GPIOZERO_AVAILABLE = False


    class OutputDevice:
        def __init__(self, pin, active_high=True, initial_value=False):
            self.pin = pin
            self._active_high = active_high
            self._value = initial_value
            self._is_active = False
            self.logger = logging.getLogger(f"MockOutputDevice.Pin{pin}")
            self.logger.info(f"Мок OutputDevice створено для піна {pin}")

        def on(self):
            self._is_active = True
            self.logger.debug(f"Мок OutputDevice pin {self.pin} -> ON")

        def off(self):
            self._is_active = False
            self.logger.debug(f"Мок OutputDevice pin {self.pin} -> OFF")

        @property
        def is_active(self):
            return self._is_active

        def close(self):
            self.logger.debug(f"Мок OutputDevice pin {self.pin} closed")

logger = logging.getLogger(__name__)

# --- Конфігурація за замовчуванням ---
DEFAULT_OPEN_RELAY_PIN = 17
DEFAULT_CLOSE_RELAY_PIN = 27
DEFAULT_RELAY_PULSE_DURATION_S = 0.5
DEFAULT_AUTO_CLOSE_TIMEOUT_S = 30  # Буде перевизначено VehicleEventHandler на 4с
DEFAULT_CLOSING_OBSTRUCTION_THRESHOLD_M = 0.3
DEFAULT_REED_CONFIRMATION_TIMEOUT_S = 5


class GateController:
    """
    Керує логікою відкриття/закриття воріт, безпекою та автоматичним закриттям.
    Використовує геркон для визначення стану та ультразвуковий датчик для безпеки.
    """

    def __init__(self,
                 sensor_manager_instance,
                 open_relay_pin: int = DEFAULT_OPEN_RELAY_PIN,
                 close_relay_pin: int = DEFAULT_CLOSE_RELAY_PIN,
                 relay_pulse_duration_s: float = DEFAULT_RELAY_PULSE_DURATION_S,
                 auto_close_timeout_s: float = DEFAULT_AUTO_CLOSE_TIMEOUT_S,
                 # Цей таймаут використовується для внутрішнього таймера
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
        if GPIOZERO_AVAILABLE:
            try:
                self.open_relay = OutputDevice(self.open_relay_pin, active_high=False, initial_value=True)
                self.close_relay = OutputDevice(self.close_relay_pin, active_high=False, initial_value=True)
                self._logger.info(f"Реле ВІДКРИТТЯ на GPIO{self.open_relay_pin} (active_low).")
                self._logger.info(f"Реле ЗАКРИТТЯ на GPIO{self.close_relay_pin} (active_low).")
                self.relays_initialized = True
            except Exception as e:
                self._logger.critical(f"Критична помилка ініціалізації реле gpiozero: {e}", exc_info=True)
        else:
            self._logger.warning("Бібліотека gpiozero недоступна. Використовуються мок-об'єкти для реле.")
            # Створюємо мок-реле, якщо gpiozero не встановлено (для тестування на ПК)
            self.open_relay = OutputDevice(self.open_relay_pin, active_high=False, initial_value=True)
            self.close_relay = OutputDevice(self.close_relay_pin, active_high=False, initial_value=True)
            self.relays_initialized = True  # Імітуємо успішну ініціалізацію моків

        self._auto_close_timer: Optional[threading.Timer] = None
        self._close_procedure_interrupted: bool = False
        self._lock = threading.Lock()

        self._commanded_state: str = "IDLE"
        self.gate_state: str = self._determine_state_from_reed_switch()

        self._logger.info(f"GateController ініціалізовано. Початковий стан (геркон): {self.gate_state}")
        self._logger.info(
            f"  Тривалість імпульсу: {self.relay_pulse_duration_s}с, Стандартний таймаут авто-закриття: {self.default_auto_close_timeout_s}с")
        self._logger.info(
            f"  Поріг перешкоди: {self.closing_obstruction_threshold_m}м, Таймаут геркона: {self.reed_confirmation_timeout_s}с")

    def _determine_state_from_reed_switch(self) -> str:
        if not self.sensor_manager or not hasattr(self.sensor_manager, 'reed_switch') or \
                not self.sensor_manager.reed_switch or \
                (
                        GPIOZERO_AVAILABLE and self.sensor_manager.reed_switch._device is None):  # Для реального gpiozero перевіряємо _device
            self._logger.warning("Геркон недоступний або не ініціалізований. Стан воріт невідомий.")
            return "UNKNOWN_SENSOR_UNAVAILABLE"
        try:
            # Для мок-об'єкта або реального
            if self.sensor_manager.reed_switch.are_gates_open:
                return "OPEN"
            elif self.sensor_manager.reed_switch.are_gates_closed:
                return "CLOSED"
            else:
                return "BETWEEN_STATES"
        except Exception as e:
            self._logger.error(f"Помилка читання стану геркона: {e}", exc_info=True)
            return "UNKNOWN_SENSOR_READ_ERROR"

    def _activate_relay_pulse(self, relay_device: OutputDevice, action_name: str):
        if not self.relays_initialized or not relay_device:
            self._logger.error(f"Реле для '{action_name}' не ініціалізовано. Дію скасовано.")
            return

        pin_number_str = str(getattr(relay_device, 'pin', 'N/A'))  # Для моків може не бути .pin.number
        if GPIOZERO_AVAILABLE and hasattr(relay_device.pin, 'number'):
            pin_number_str = str(relay_device.pin.number)

        self._logger.debug(f"Активація реле '{action_name}' на піні GPIO{pin_number_str} "
                           f"з тривалістю імпульсу: {self.relay_pulse_duration_s} с.")
        try:
            relay_device.on()
            time.sleep(self.relay_pulse_duration_s)
        except Exception as e:
            self._logger.error(f"Помилка під час увімкнення або очікування реле '{action_name}': {e}", exc_info=True)
        finally:
            try:
                if relay_device.is_active:  # Перевіряємо, чи реле все ще активне
                    relay_device.off()
                    self._logger.info(f"Реле '{action_name}' (GPIO{pin_number_str}) деактивовано.")
            except Exception as e_off:
                self._logger.error(f"Помилка під час вимкнення реле '{action_name}': {e_off}", exc_info=True)

    def open_gate(self) -> bool:
        if not self.relays_initialized:
            self._logger.error("Неможливо відкрити ворота: реле не ініціалізовано.")
            return False

        current_actual_state = self._determine_state_from_reed_switch()
        if current_actual_state == "OPEN":
            self._logger.info("Ворота вже відкриті (геркон). Команда ВІДКРИТИ ігнорується.")
            self._commanded_state = "IDLE"
            self.gate_state = "OPEN"
            return True

        with self._lock:
            if self._auto_close_timer and self._auto_close_timer.is_alive():
                self._logger.info("Команда ВІДКРИТИ: скасування активного таймера авто-закриття.")
                self._auto_close_timer.cancel()
                self._auto_close_timer = None
            self._close_procedure_interrupted = False

        self._logger.info("КОМАНДА: Відкриття воріт.")
        self._activate_relay_pulse(self.open_relay, "OPEN")
        self._commanded_state = "OPENING"
        self.gate_state = "OPENING"

        # Вбудована коротка перевірка герконом (не для retry логіки VehicleEventHandler)
        if self.reed_confirmation_timeout_s > 0 and self.sensor_manager and \
                hasattr(self.sensor_manager, 'reed_switch') and self.sensor_manager.reed_switch and \
                (
                        GPIOZERO_AVAILABLE and self.sensor_manager.reed_switch._device is not None or not GPIOZERO_AVAILABLE):  # Додана перевірка для моків
            self._logger.debug(
                f"Очікування підтвердження відкриття від геркона ({self.reed_confirmation_timeout_s}с)...")
            if self.sensor_manager.reed_switch.wait_for_open(timeout=self.reed_confirmation_timeout_s):
                if self.sensor_manager.reed_switch.are_gates_open:  # Додаткова перевірка після wait_for_open
                    self.gate_state = "OPEN"
                    self._commanded_state = "IDLE"
                    self._logger.info("Ворота підтверджено відкритими (геркон, внутрішня перевірка GateController).")
                else:  # wait_for_open повернув True, але are_gates_open - False (малоймовірно)
                    self.gate_state = self._determine_state_from_reed_switch()  # Перевірити актуальний стан
                    self._logger.warning("wait_for_open успішний, але are_gates_open - ні. Стан: " + self.gate_state)

            else:  # wait_for_open повернув False (таймаут)
                self.gate_state = "MOVEMENT_TIMEOUT_OPEN"
                self._logger.warning(
                    f"Таймаут ({self.reed_confirmation_timeout_s}с) очікування відкриття воріт (геркон).")
        return True

    def is_passage_clear_for_closing(self) -> bool:
        passage_sensor = getattr(self.sensor_manager, 'ultrasonic_sensor_entry', None)
        if not passage_sensor or (GPIOZERO_AVAILABLE and passage_sensor._sensor is None):
            self._logger.warning("УЗД для перевірки проїзду недоступний. З метою безпеки, проїзд НЕ вільний.")
            return False
        try:
            distance = passage_sensor.get_distance()
        except Exception as e:
            self._logger.error(f"Помилка отримання даних з УЗД: {e}", exc_info=True)
            return False

        if distance is None:
            self._logger.warning("УЗД повернув None. З метою безпеки, проїзд НЕ вільний.")
            return False
        if distance == float('inf'):
            self._logger.debug(f"Проїзд вільний для закриття. Відстань: > max_dist.")
            return True

        if distance < self.closing_obstruction_threshold_m:
            self._logger.info(
                f"Проїзд ЗАБЛОКОВАНО. Перешкода на {distance:.2f}м (поріг: {self.closing_obstruction_threshold_m}м).")
            return False
        else:
            self._logger.debug(f"Проїзд вільний для закриття. Відстань: {distance:.2f}м.")
            return True

    def close_gate(self) -> bool:
        if not self.relays_initialized:
            self._logger.error("Неможливо закрити ворота: реле не ініціалізовано.")
            return False

        current_actual_state = self._determine_state_from_reed_switch()
        if current_actual_state == "CLOSED":
            self._logger.info("Ворота вже закриті (геркон). Команда ЗАКРИТИ ігнорується.")
            self._commanded_state = "IDLE"
            self.gate_state = "CLOSED"
            return True

        with self._lock:
            if self._close_procedure_interrupted:
                self._logger.info("Процедуру закриття було перервано. Ворота НЕ закриватимуться.")
                self._close_procedure_interrupted = False
                self.gate_state = self._determine_state_from_reed_switch()
                return False

            if not self.is_passage_clear_for_closing():
                self._logger.warning("КОМАНДУ ВІДХИЛЕНО: Неможливо закрити ворота, проїзд заблоковано.")
                self.gate_state = "OBSTRUCTED"
                return False

        self._logger.info("КОМАНДА: Закриття воріт (проїзд вільний).")
        self._activate_relay_pulse(self.close_relay, "CLOSE")
        self._commanded_state = "CLOSING"
        self.gate_state = "CLOSING"

        if self.reed_confirmation_timeout_s > 0 and self.sensor_manager and \
                hasattr(self.sensor_manager, 'reed_switch') and self.sensor_manager.reed_switch and \
                (GPIOZERO_AVAILABLE and self.sensor_manager.reed_switch._device is not None or not GPIOZERO_AVAILABLE):
            self._logger.debug(
                f"Очікування підтвердження закриття від геркона ({self.reed_confirmation_timeout_s}с)...")
            if self.sensor_manager.reed_switch.wait_for_close(timeout=self.reed_confirmation_timeout_s):
                if self.sensor_manager.reed_switch.are_gates_closed:
                    self.gate_state = "CLOSED"
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
        self._logger.info("Таймер авто-закриття спрацював. Спроба закрити ворота.")
        self.close_gate()

    def start_auto_close_timer(self, timeout_s: Optional[float] = None):
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
        with self._lock:
            if self._auto_close_timer and self._auto_close_timer.is_alive():
                self._logger.info("Таймер авто-закриття скасовано вручну.")
                self._auto_close_timer.cancel()
                self._auto_close_timer = None
            else:
                self._logger.debug("Немає активного таймера авто-закриття для скасування.")

    def interrupt_closing_procedure(self):
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
        self._logger.info("Очищення ресурсів GateController...")
        self.cancel_auto_close_timer()
        if self.relays_initialized:  # Виконуємо close тільки якщо реле були ініціалізовані
            if hasattr(self, 'open_relay') and self.open_relay:
                try:
                    self.open_relay.close()
                except Exception as e:
                    self._logger.error(f"Помилка при закритті open_relay: {e}")
            if hasattr(self, 'close_relay') and self.close_relay:
                try:
                    self.close_relay.close()
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

    logger_test = logging.getLogger(f"{__name__}_Test")
    logger_test.info("--- Тестування модуля gate_controller.py (з моками) ---")


    class MockGpioDevice:  # Простий мок для gpiozero.DigitalInputDevice
        def __init__(self, pin, pull_up=None):
            self.pin = pin
            # Початковий стан: якщо pull_up=True, геркон розімкнений (ворота закриті) -> HIGH (1)
            self._value = 1 if pull_up else 0
            self.is_active_state = False  # Для is_active (LOW -> True)
            if pull_up and self._value == 0: self.is_active_state = True
            if not pull_up and self._value == 1: self.is_active_state = True  # (для active_low без pull_up) - не наш випадок

            self.logger = logging.getLogger(f"MockGpioDevice.Pin{pin}")
            self.logger.info(
                f"Мок GPIO пристрою на піні {pin} створено. pull_up={pull_up}, value={self._value}, is_active={self.is_active_state}")

        @property
        def is_active(self):
            return self.is_active_state

        def _update_active_state(self):  # pull_up=True
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
            if self.is_active_state: return True  # Якщо вже активний
            # Імітація очікування - для тесту потрібно буде змінювати value ззовні
            if timeout: time.sleep(min(0.1, timeout / 2))  # Коротка пауза
            return self.is_active_state

        def wait_for_inactive(self, timeout):
            self.logger.debug(f"Mock: wait_for_inactive on pin {self.pin}")
            if not self.is_active_state: return True  # Якщо вже неактивний
            if timeout: time.sleep(min(0.1, timeout / 2))
            return not self.is_active_state


    class MockReedSwitch:  # Адаптований мок
        def __init__(self, pin_number=22, name="MockReed"):
            self.name = name
            self._device = MockGpioDevice(pin_number, pull_up=True)
            self._logger = logging.getLogger(f"MockSensor.{self.name}")
            self.set_gates_closed()  # Початковий стан

        @property
        def are_gates_open(self): return self._device.is_active

        @property
        def are_gates_closed(self): return not self._device.is_active

        def wait_for_open(self, timeout=None): return self._device.wait_for_active(timeout)

        def wait_for_close(self, timeout=None): return self._device.wait_for_inactive(timeout)

        def set_gates_open(self): self._device.value = 0; self._logger.info(f"Мок '{self.name}': ВСТАНОВЛЕНО ВІДКРИТО")

        def set_gates_closed(self): self._device.value = 1; self._logger.info(f"Мок '{self.name}': ВСТАНОВЛЕНО ЗАКРИТО")


    class MockUltrasonicSensor:  # Адаптований мок
        def __init__(self, name="MockUS"):
            self.name = name
            self._sensor = True  # Імітуємо, що датчик ініціалізовано
            self.distance = 2.0  # Початково проїзд вільний
            self._logger = logging.getLogger(f"MockSensor.{self.name}")

        def get_distance(self): self._logger.debug(
            f"Мок '{self.name}': повертає відстань {self.distance}м"); return self.distance

        def set_distance(self, dist): self.distance = dist; self._logger.info(
            f"Мок '{self.name}': встановлено відстань {dist}м")


    mock_sensor_manager = MagicMock()  # Використовуємо MagicMock для гнучкості
    mock_sensor_manager.reed_switch = MockReedSwitch()
    mock_sensor_manager.ultrasonic_sensor_entry = MockUltrasonicSensor()

    # --- Тести ---
    gate_ctrl = GateController(
        sensor_manager_instance=mock_sensor_manager,
        open_relay_pin=20, close_relay_pin=21,
        auto_close_timeout_s=2,  # Короткий для тесту
        reed_confirmation_timeout_s=0.5
    )

    if not gate_ctrl.relays_initialized:
        logger_test.error("Реле не ініціалізовані, тестування неможливе.")
    else:
        logger_test.info(f"Початковий стан воріт: {gate_ctrl.get_current_gate_state()}")  # CLOSED

        logger_test.info("\nТест: Відкриття воріт...")


        # Імітуємо, що геркон спрацює під час очікування в open_gate
        def simulate_open_reed():
            time.sleep(0.1); mock_sensor_manager.reed_switch.set_gates_open()


        threading.Thread(target=simulate_open_reed).start()
        gate_ctrl.open_gate()
        time.sleep(gate_ctrl.reed_confirmation_timeout_s + 0.2)  # Дати час на wait_for_open
        logger_test.info(f"Стан після команди відкрити: {gate_ctrl.get_current_gate_state()}")  # Має бути OPEN

        logger_test.info("\nТест: Закриття воріт (проїзд вільний)...")
        mock_sensor_manager.ultrasonic_sensor_entry.set_distance(2.0)  # Вільний


        def simulate_close_reed():
            time.sleep(0.1); mock_sensor_manager.reed_switch.set_gates_closed()


        threading.Thread(target=simulate_close_reed).start()
        gate_ctrl.close_gate()
        time.sleep(gate_ctrl.reed_confirmation_timeout_s + 0.2)
        logger_test.info(f"Стан після команди закрити: {gate_ctrl.get_current_gate_state()}")  # Має бути CLOSED

        logger_test.info("\nТест: Спроба закрити (проїзд заблоковано 0.1м)...")
        mock_sensor_manager.reed_switch.set_gates_open()  # "Відкриваємо" ворота
        gate_ctrl.gate_state = "OPEN";
        gate_ctrl._commanded_state = "IDLE"
        mock_sensor_manager.ultrasonic_sensor_entry.set_distance(0.1)  # Перешкода!
        gate_ctrl.close_gate()
        logger_test.info(
            f"Стан після спроби закрити з перешкодою: {gate_ctrl.get_current_gate_state()}")  # Має бути OBSTRUCTED

        logger_test.info("\nТест: Авто-закриття...")
        mock_sensor_manager.reed_switch.set_gates_open()  # Ворота відкриті
        gate_ctrl.gate_state = "OPEN";
        gate_ctrl._commanded_state = "IDLE"
        mock_sensor_manager.ultrasonic_sensor_entry.set_distance(2.0)  # Проїзд вільний

        gate_ctrl.start_auto_close_timer()  # Таймер на 2с
        logger_test.info("Таймер авто-закриття запущено. Очікування 2.5с...")


        # Імітуємо, що геркон спрацює після команди закриття від таймера
        def delayed_reed_close_for_auto_timer():
            time.sleep(2.1)  # Після спрацювання таймера (2с) + невеликий запас
            mock_sensor_manager.reed_switch.set_gates_closed()


        threading.Thread(target=delayed_reed_close_for_auto_timer).start()
        time.sleep(2.5 + gate_ctrl.reed_confirmation_timeout_s)
        logger_test.info(f"Стан після авто-закриття: {gate_ctrl.get_current_gate_state()}")  # Має бути CLOSED

        gate_ctrl.cleanup()
    logger_test.info("--- Тестування модуля gate_controller.py завершено ---")