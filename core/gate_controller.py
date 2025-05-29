# core/gate_controller.py
import time
import logging
import threading
from gpiozero import OutputDevice  # Для керування реле

logger = logging.getLogger(__name__)

# --- Конфігурація за замовчуванням ---
DEFAULT_OPEN_RELAY_PIN = 17
DEFAULT_CLOSE_RELAY_PIN = 27
DEFAULT_RELAY_PULSE_DURATION_S = 0.5
DEFAULT_AUTO_CLOSE_TIMEOUT_S = 30
DEFAULT_CLOSING_OBSTRUCTION_THRESHOLD_M = 0.3
# Новий параметр: час очікування підтвердження від геркона (0 - не чекати)
DEFAULT_REED_CONFIRMATION_TIMEOUT_S = 5  # секунд


class GateController:
    """
    Керує логікою відкриття/закриття воріт, безпекою та автоматичним закриттям,
    використовуючи геркон для визначення стану та ультразвуковий датчик для безпеки.
    """

    def __init__(self,
                 sensor_manager_instance,  # Екземпляр SensorManager
                 open_relay_pin=DEFAULT_OPEN_RELAY_PIN,
                 close_relay_pin=DEFAULT_CLOSE_RELAY_PIN,
                 relay_pulse_duration_s=DEFAULT_RELAY_PULSE_DURATION_S,
                 auto_close_timeout_s=DEFAULT_AUTO_CLOSE_TIMEOUT_S,
                 closing_obstruction_threshold_m=DEFAULT_CLOSING_OBSTRUCTION_THRESHOLD_M,
                 reed_confirmation_timeout_s=DEFAULT_REED_CONFIRMATION_TIMEOUT_S):
        self.sensor_manager = sensor_manager_instance
        self.open_relay_pin = open_relay_pin
        self.close_relay_pin = close_relay_pin
        self.relay_pulse_duration_s = relay_pulse_duration_s
        self.auto_close_timeout_s = auto_close_timeout_s
        self.closing_obstruction_threshold_m = closing_obstruction_threshold_m
        self.reed_confirmation_timeout_s = reed_confirmation_timeout_s

        self._logger = logging.getLogger(f"{__name__}.GateController_PinsO{open_relay_pin}C{close_relay_pin}")

        try:
            self.open_relay = OutputDevice(self.open_relay_pin, active_high=False, initial_value=True)
            self.close_relay = OutputDevice(self.close_relay_pin, active_high=False, initial_value=True)
            self._logger.info(f"Реле ВІДКРИТТЯ налаштовано на GPIO{self.open_relay_pin} (active_low).")
            self._logger.info(f"Реле ЗАКРИТТЯ налаштовано на GPIO{self.close_relay_pin} (active_low).")
        except Exception as e:
            self._logger.error(f"Помилка ініціалізації реле: {e}", exc_info=True)
            raise RuntimeError(f"Не вдалося ініціалізувати реле: {e}")

        self._auto_close_timer = None
        self._close_procedure_interrupted = False
        self._lock = threading.Lock()

        # Внутрішні стани, що відображають останню відому дію або намір
        self._commanded_state = "UNKNOWN"  # 'OPENING', 'CLOSING', or 'IDLE'

        # Ініціалізація початкового стану за допомогою геркона
        self.gate_state = self._determine_state_from_reed_switch()
        self._logger.info(f"GateController ініціалізовано. Початковий стан воріт (геркон): {self.gate_state}")
        self._logger.info(f"  Тривалість імпульсу реле: {self.relay_pulse_duration_s} с")
        self._logger.info(f"  Тайм-аут авто-закриття: {self.auto_close_timeout_s} с")
        self._logger.info(f"  Поріг блокування при закритті: {self.closing_obstruction_threshold_m} м")
        self._logger.info(f"  Тайм-аут підтвердження герконом: {self.reed_confirmation_timeout_s} с")

    def _determine_state_from_reed_switch(self) -> str:
        """Визначає поточний стан воріт на основі даних геркона."""
        if not self.sensor_manager or not hasattr(self.sensor_manager, 'reed_switch'):  # Змінено на reed_switch
            self._logger.warning("Геркон не знайдено в sensor_manager. Стан невідомий.")
            return "UNKNOWN_SENSOR_UNAVAILABLE"

        reed = self.sensor_manager.reed_switch  # Змінено на reed_switch
        if reed._device is None:  # Перевірка, чи сам геркон ініціалізований
            self._logger.warning("Пристрій геркона не ініціалізовано. Стан невідомий.")
            return "UNKNOWN_SENSOR_DEVICE_ERROR"
        try:
            if reed.are_gates_open:  #
                return "OPEN"
            elif reed.are_gates_closed:  #
                return "CLOSED"
            else:
                # Якщо геркон не показує ні відкрито, ні закрито (наприклад, проміжний стан або несправність)
                return "BETWEEN_STATES"  # Або "PARTIALLY_OPEN"
        except Exception as e:
            self._logger.error(f"Помилка читання стану геркона: {e}", exc_info=True)
            return "UNKNOWN_SENSOR_READ_ERROR"

    def _activate_relay_pulse(self, relay_device: OutputDevice, action_name: str):
        """Активує вказане реле на встановлену тривалість імпульсу."""
        if not relay_device:
            self._logger.error(f"Реле для дії '{action_name}' не ініціалізовано. Дію скасовано.")
            return
        self._logger.debug(f"Активація реле '{action_name}' на піні GPIO{relay_device.pin.number}...")
        try:
            relay_device.on()
            time.sleep(self.relay_pulse_duration_s)
            relay_device.off()
            self._logger.info(f"Реле '{action_name}' (GPIO{relay_device.pin.number}) було активовано імпульсом.")
        except Exception as e:
            self._logger.error(f"Помилка під час активації реле '{action_name}': {e}", exc_info=True)

    def open_gate(self):
        """Надсилає команду на відкриття воріт та оновлює очікуваний стан."""
        current_actual_state = self._determine_state_from_reed_switch()
        if current_actual_state == "OPEN":
            self._logger.info("Ворота вже відкриті (за даними геркона). Команда ВІДКРИТИ ігнорується.")
            self._commanded_state = "IDLE"
            self.gate_state = "OPEN"
            return True  # Вважаємо успішним, бо мета досягнута

        with self._lock:
            if self._auto_close_timer and self._auto_close_timer.is_alive():
                self._logger.info(
                    "Команда ВІДКРИТИ отримана під час активного таймера авто-закриття. Скасування таймера.")
                self._auto_close_timer.cancel()
            self._close_procedure_interrupted = False

        self._logger.info("КОМАНДА: Відкриття воріт.")
        self._activate_relay_pulse(self.open_relay, "OPEN")
        self._commanded_state = "OPENING"
        self.gate_state = "OPENING"  # Тимчасовий стан до підтвердження герконом

        if self.reed_confirmation_timeout_s > 0 and hasattr(self.sensor_manager, 'reed_switch'):
            self._logger.debug(
                f"Очікування підтвердження відкриття від геркона ({self.reed_confirmation_timeout_s} с)...")
            try:
                self.sensor_manager.reed_switch.wait_for_open(timeout=self.reed_confirmation_timeout_s)  #
                if self.sensor_manager.reed_switch.are_gates_open:  #
                    self.gate_state = "OPEN"
                    self._commanded_state = "IDLE"
                    self._logger.info("Ворота підтверджено відкритими (геркон).")
                else:  # Тайм-аут або не відкрилися
                    self.gate_state = "MOVEMENT_TIMEOUT_OPEN"
                    self._logger.warning("Тайм-аут очікування відкриття воріт (геркон) або не вдалося відкрити.")
            except Exception as e:  # gpiozero може кидати винятки при тайм-ауті або помилці
                self._logger.error(f"Помилка під час очікування відкриття герконом: {e}")
                self.gate_state = self._determine_state_from_reed_switch()  # Перевірити поточний стан
        else:
            # Якщо не чекаємо підтвердження, стан залишиться "OPENING"
            # get_current_gate_state оновить його при наступному виклику
            pass
        return True

    def is_passage_clear_for_closing(self) -> bool:
        """Перевіряє, чи вільний проїзд за допомогою ультразвукового датчика."""
        # Використовуємо ultrasonic_sensor_entry, як зазначено у вашому sensors.py
        # та як було реалізовано раніше.
        if not self.sensor_manager or not hasattr(self.sensor_manager, 'ultrasonic_sensor_entry'):
            self._logger.warning("Ультразвуковий датчик 'ultrasonic_sensor_entry' не знайдено в sensor_manager. "
                                 "З метою безпеки припускаємо, що проїзд ЗАБЛОКОВАНО.")
            return False

        ultrasonic_sensor = self.sensor_manager.ultrasonic_sensor_entry
        if ultrasonic_sensor._sensor is None:  # Перевірка, чи сам датчик ініціалізований
            self._logger.warning(
                "Пристрій ультразвукового датчика не ініціалізовано. Припускаємо, що проїзд ЗАБЛОКОВАНО.")
            return False

        try:
            distance = ultrasonic_sensor.get_distance()  #
        except Exception as e:
            self._logger.error(f"Помилка отримання даних з ультразвукового датчика: {e}", exc_info=True)
            return False

        if distance is None:
            self._logger.warning("Ультразвуковий датчик повернув None. Припускаємо, що проїзд ЗАБЛОКОВАНО.")
            return False
        if distance == float('inf'):
            self._logger.debug(f"Проїзд вільний для закриття. Відстань: > max_dist ({distance} м).")
            return True

        if distance < self.closing_obstruction_threshold_m:
            self._logger.info(f"Проїзд ЗАБЛОКОВАНО для закриття. Виявлено перешкоду на відстані {distance:.2f} м "
                              f"(поріг: {self.closing_obstruction_threshold_m} м).")
            return False
        else:
            self._logger.debug(f"Проїзд вільний для закриття. Відстань: {distance:.2f} м.")
            return True

    def close_gate(self):
        """Надсилає команду на закриття воріт та оновлює очікуваний стан."""
        current_actual_state = self._determine_state_from_reed_switch()
        if current_actual_state == "CLOSED":
            self._logger.info("Ворота вже закриті (за даними геркона). Команда ЗАКРИТИ ігнорується.")
            self._commanded_state = "IDLE"
            self.gate_state = "CLOSED"
            return True

        with self._lock:
            if self._close_procedure_interrupted:
                self._logger.info("Процедуру закриття було перервано. Ворота НЕ закриватимуться.")
                self._close_procedure_interrupted = False
                self.gate_state = self._determine_state_from_reed_switch()  # Оновити актуальний стан
                return False

            if not self.is_passage_clear_for_closing():
                self._logger.warning(
                    "КОМАНДУ ВІДХИЛЕНО: Неможливо закрити ворота, проїзд заблоковано або помилка датчика.")
                self.gate_state = "OBSTRUCTED"
                return False

        self._logger.info("КОМАНДА: Закриття воріт (проїзд вільний).")
        self._activate_relay_pulse(self.close_relay, "CLOSE")
        self._commanded_state = "CLOSING"
        self.gate_state = "CLOSING"  # Тимчасовий стан

        if self.reed_confirmation_timeout_s > 0 and hasattr(self.sensor_manager, 'reed_switch'):
            self._logger.debug(
                f"Очікування підтвердження закриття від геркона ({self.reed_confirmation_timeout_s} с)...")
            try:
                self.sensor_manager.reed_switch.wait_for_close(timeout=self.reed_confirmation_timeout_s)  #
                if self.sensor_manager.reed_switch.are_gates_closed:  #
                    self.gate_state = "CLOSED"
                    self._commanded_state = "IDLE"
                    self._logger.info("Ворота підтверджено закритими (геркон).")
                else:  # Тайм-аут або не закрилися
                    self.gate_state = "MOVEMENT_TIMEOUT_CLOSE"
                    self._logger.warning("Тайм-аут очікування закриття воріт (геркон) або не вдалося закрити.")
            except Exception as e:
                self._logger.error(f"Помилка під час очікування закриття герконом: {e}")
                self.gate_state = self._determine_state_from_reed_switch()
        else:
            # Стан залишиться "CLOSING", get_current_gate_state оновить його
            pass
        return True

    def _auto_close_gate_callback(self):
        self._logger.info("Таймер авто-закриття спрацював. Спроба закрити ворота.")
        self.close_gate()

    def start_auto_close_timer(self, timeout_s=None):
        with self._lock:
            if self._auto_close_timer and self._auto_close_timer.is_alive():
                self._logger.debug("Скасування існуючого таймера авто-закриття.")
                self._auto_close_timer.cancel()
            self._close_procedure_interrupted = False
            effective_timeout = timeout_s if timeout_s is not None else self.auto_close_timeout_s
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
            # Після переривання, стан воріт має бути визначений герконом
            self.gate_state = self._determine_state_from_reed_switch()
            self._commanded_state = "IDLE"  # Скасувати попередню команду на закриття

    def get_current_gate_state(self) -> str:
        """
        Повертає поточний стан воріт, пріоритезуючи дані геркона.
        Якщо геркон недоступний або в проміжному стані, враховує останню команду.
        """
        reed_state = self._determine_state_from_reed_switch()

        if reed_state in ["OPEN", "CLOSED"]:
            self.gate_state = reed_state
            if self._commanded_state not in ["IDLE", "UNKNOWN"]:  # Якщо була команда, а геркон вже в цілі
                if (reed_state == "OPEN" and self._commanded_state == "OPENING") or \
                        (reed_state == "CLOSED" and self._commanded_state == "CLOSING"):
                    self._commanded_state = "IDLE"  # Рух завершено
            return self.gate_state

        # Якщо геркон в проміжному стані (BETWEEN_STATES) або помилка датчика
        if self.gate_state == "OBSTRUCTED":  # Перевірка на перешкоду має вищий пріоритет
            return "OBSTRUCTED"
        if self._commanded_state == "OPENING":
            return "OPENING"
        if self._commanded_state == "CLOSING":
            return "CLOSING"
        if reed_state == "BETWEEN_STATES":  # Якщо немає активної команди, але геркон каже "між"
            return "BETWEEN_STATES"

        # Повертаємо останній відомий стан, якщо нічого іншого не визначено
        return self.gate_state

    def cleanup(self):
        self._logger.info("Очищення ресурсів GateController...")
        self.cancel_auto_close_timer()
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


# --- Приклад використання та тестування модуля ---
if __name__ == '__main__':
    logging.basicConfig(
        level=logging.DEBUG,
        format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s'
    )


    # --- Мок-об'єкти для SensorManager, ReedSwitch та UltrasonicSensor ---
    class MockGpioDevice:  # Простий мок для gpiozero.DigitalInputDevice
        def __init__(self, pin_number, pull_up=None, active_state=None, pin_factory=None):
            self.pin = pin_number
            self._value = 1 if pull_up else 0  # pull_up=True, геркон розімкнений (ворота закриті) -> HIGH (1)
            self.logger = logging.getLogger(f"MockGpioDevice.Pin{pin_number}")
            self.logger.info(f"Мок GPIO пристрою на піні {pin_number} створено. pull_up={pull_up}, value={self._value}")

        @property
        def value(self):
            return self._value

        @value.setter
        def value(self, new_value):
            self.logger.info(f"Мок GPIO пін {self.pin} встановлено в {new_value}")
            self._value = new_value

        @property
        def is_active(self):  # Для DigitalInputDevice, is_active == True коли value == 0 (LOW) якщо pull_up=True
            return self._value == 0

        def wait_for_active(self, timeout=None):
            self.logger.info(f"Мок: очікування ACTIVE на піні {self.pin} (таймаут: {timeout}с)")
            if self._value == 0: return
            if timeout: time.sleep(min(timeout, 0.1))  # Імітація очікування
            # Для тестування можна змінити _value або імітувати тайм-аут
            # raise GPIOZEROTimeout("Mock timeout")

        def wait_for_inactive(self, timeout=None):
            self.logger.info(f"Мок: очікування INACTIVE на піні {self.pin} (таймаут: {timeout}с)")
            if self._value == 1: return
            if timeout: time.sleep(min(timeout, 0.1))
            # raise GPIOZEROTimeout("Mock timeout")

        def close(self):
            self.logger.info(f"Мок GPIO пристрою на піні {self.pin} закрито.")


    class MockReedSwitch:
        def __init__(self, pin_number=22, name="MockReed"):
            self.name = name
            # Імітуємо DigitalInputDevice з pull_up=True
            self._device = MockGpioDevice(pin_number, pull_up=True)
            self._logger = logging.getLogger(f"MockSensor.{self.name}")
            self._logger.info(f"Мок-геркон '{self.name}' ініціалізовано на піні {pin_number}.")
            # Встановлюємо початковий стан: ворота закриті (геркон розімкнений, пін HIGH)
            self._device.value = 1

        @property
        def are_gates_open(self):  # Геркон замкнений (магніт близько) -> пін LOW -> _device.is_active == True
            return self._device.is_active if self._device else False

        @property
        def are_gates_closed(self):  # Геркон розімкнений (магніт далеко) -> пін HIGH -> _device.is_active == False
            return not self._device.is_active if self._device else False

        def wait_for_open(self, timeout=None):
            self._logger.info(f"'{self.name}' очікує відкриття (таймаут: {timeout}с)")
            self._device.wait_for_active(timeout)

        def wait_for_close(self, timeout=None):
            self._logger.info(f"'{self.name}' очікує закриття (таймаут: {timeout}с)")
            self._device.wait_for_inactive(timeout)

        # Допоміжні методи для тестування
        def set_gates_open(self):
            self._logger.info(f"'{self.name}' МОК: встановлено стан ВІДКРИТО (пін LOW).")
            self._device.value = 0  # LOW signal, contact closed

        def set_gates_closed(self):
            self._logger.info(f"'{self.name}' МОК: встановлено стан ЗАКРИТО (пін HIGH).")
            self._device.value = 1  # HIGH signal, contact open

        def set_pin_value(self, val):  # Для імітації проміжних станів або помилок
            self._device.value = val


    class MockUltrasonicSensor:
        def __init__(self, trigger_pin=23, echo_pin=24, name="MockUltra"):  #
            self.name = name
            self._sensor = True  # Імітація того, що датчик ініціалізовано
            self._distance_m = 3.0
            self._logger = logging.getLogger(f"MockSensor.{self.name}")
            self._logger.info(f"Мок-УЗД '{self.name}' ініціалізовано (Trig:{trigger_pin}, Echo:{echo_pin}).")

        def get_distance(self):  #
            self._logger.debug(f"'{self.name}' повертає відстань: {self._distance_m} м")
            return self._distance_m

        def set_distance(self, dist_m):
            self._logger.info(f"'{self.name}' МОК: встановлює відстань: {dist_m} м")
            self._distance_m = dist_m


    class MockSensorManager:
        def __init__(self):
            self.reed_switch = MockReedSwitch()  # Змінено на reed_switch
            self.ultrasonic_sensor_entry = MockUltrasonicSensor()  #
            self._logger = logging.getLogger("MockSensorManager")
            self._logger.info("Мок-менеджер датчиків ініціалізовано.")


    logger.info("--- Тестування модуля gate_controller.py з герконом ---")

    TEST_OPEN_PIN_NUM = 20
    TEST_CLOSE_PIN_NUM = 21
    TEST_PULSE_DURATION = 0.2
    TEST_AUTO_CLOSE_S = 3
    TEST_OBSTRUCTION_M = 0.3
    TEST_REED_TIMEOUT_S = 1  # Короткий таймаут для тестів

    mock_sensor_mgr = MockSensorManager()
    gate_ctrl_instance = None

    try:
        logger.info("\n--- Тест 1: Ініціалізація GateController ---")
        # Початковий стан геркона: ЗАКРИТО
        mock_sensor_mgr.reed_switch.set_gates_closed()
        gate_ctrl_instance = GateController(
            sensor_manager_instance=mock_sensor_mgr,
            open_relay_pin=TEST_OPEN_PIN_NUM,
            close_relay_pin=TEST_CLOSE_PIN_NUM,
            relay_pulse_duration_s=TEST_PULSE_DURATION,
            auto_close_timeout_s=TEST_AUTO_CLOSE_S,
            closing_obstruction_threshold_m=TEST_OBSTRUCTION_M,
            reed_confirmation_timeout_s=TEST_REED_TIMEOUT_S
        )
        logger.info(f"GateController ініціалізовано. Початковий стан: {gate_ctrl_instance.get_current_gate_state()}")
        assert gate_ctrl_instance.get_current_gate_state() == "CLOSED"

        logger.info("\n--- Тест 2: Відкриття воріт (з підтвердженням герконом) ---")


        # Імітуємо, що геркон спрацює під час очікування
        # У реальному житті це відбудеться після фактичного руху воріт
        def simulate_reed_open_after_command():
            time.sleep(TEST_REED_TIMEOUT_S / 2)  # Геркон спрацьовує швидше за таймаут
            mock_sensor_mgr.reed_switch.set_gates_open()


        threading.Thread(target=simulate_reed_open_after_command).start()
        gate_ctrl_instance.open_gate()
        logger.info(
            f"Стан воріт після команди ВІДКРИТИ та очікування геркона: {gate_ctrl_instance.get_current_gate_state()}")
        assert gate_ctrl_instance.get_current_gate_state() == "OPEN"

        logger.info("\n--- Тест 2.1: Спроба відкрити вже відкриті ворота ---")
        gate_ctrl_instance.open_gate()  # Геркон вже показує "OPEN"
        logger.info(f"Стан воріт: {gate_ctrl_instance.get_current_gate_state()}")
        assert gate_ctrl_instance.get_current_gate_state() == "OPEN"

        logger.info("\n--- Тест 3: Закриття воріт (з підтвердженням герконом, проїзд вільний) ---")
        mock_sensor_mgr.ultrasonic_sensor_entry.set_distance(2.0)  # Проїзд вільний


        def simulate_reed_close_after_command():
            time.sleep(TEST_REED_TIMEOUT_S / 2)
            mock_sensor_mgr.reed_switch.set_gates_closed()


        threading.Thread(target=simulate_reed_close_after_command).start()
        gate_ctrl_instance.close_gate()
        logger.info(
            f"Стан воріт після команди ЗАКРИТИ та очікування геркона: {gate_ctrl_instance.get_current_gate_state()}")
        assert gate_ctrl_instance.get_current_gate_state() == "CLOSED"

        logger.info("\n--- Тест 3.1: Спроба закрити вже закриті ворота ---")
        gate_ctrl_instance.close_gate()  # Геркон вже показує "CLOSED"
        logger.info(f"Стан воріт: {gate_ctrl_instance.get_current_gate_state()}")
        assert gate_ctrl_instance.get_current_gate_state() == "CLOSED"

        logger.info("\n--- Тест 4: Відкриття воріт (геркон не спрацьовує - таймаут) ---")
        mock_sensor_mgr.reed_switch.set_gates_closed()  # Починаємо з закритих
        # Геркон не змінюватиме стан, щоб імітувати таймаут
        gate_ctrl_instance.open_gate()
        logger.info(
            f"Стан воріт після команди ВІДКРИТИ та таймауту геркона: {gate_ctrl_instance.get_current_gate_state()}")
        assert gate_ctrl_instance.get_current_gate_state() == "MOVEMENT_TIMEOUT_OPEN"
        # Перевірка, що get_current_gate_state поверне поточний стан геркона, якщо він не змінився
        current_state_via_getter = gate_ctrl_instance.get_current_gate_state()
        logger.info(f"Стан через get_current_gate_state: {current_state_via_getter}")
        assert current_state_via_getter == "CLOSED"  # Бо геркон так і залишився CLOSED

        logger.info("\n--- Тест 5: Авто-закриття (ворота відкриті, проїзд вільний, геркон спрацює) ---")
        mock_sensor_mgr.reed_switch.set_gates_open()  # Починаємо з відкритих
        gate_ctrl_instance.gate_state = "OPEN"  # Встановлюємо актуальний стан
        gate_ctrl_instance._commanded_state = "IDLE"
        mock_sensor_mgr.ultrasonic_sensor_entry.set_distance(2.0)


        def simulate_reed_close_for_auto():
            time.sleep(TEST_AUTO_CLOSE_S + TEST_REED_TIMEOUT_S / 2)  # Спрацює після того, як таймер викличе close_gate
            mock_sensor_mgr.reed_switch.set_gates_closed()


        threading.Thread(target=simulate_reed_close_for_auto).start()
        gate_ctrl_instance.start_auto_close_timer()
        logger.info(f"Таймер авто-закриття запущено. Очікуємо {TEST_AUTO_CLOSE_S + TEST_REED_TIMEOUT_S + 0.5} с...")
        time.sleep(TEST_AUTO_CLOSE_S + TEST_REED_TIMEOUT_S + 0.5)
        logger.info(f"Стан воріт після авто-закриття: {gate_ctrl_instance.get_current_gate_state()}")
        assert gate_ctrl_instance.get_current_gate_state() == "CLOSED"

        logger.info("\n--- Тест 6: Перевірка стану 'BETWEEN_STATES' ---")
        mock_sensor_mgr.reed_switch.set_pin_value(0.5)  # Імітація невизначеного стану геркона (не 0 і не 1)
        gate_ctrl_instance._commanded_state = "IDLE"  # Немає активної команди
        logger.info(f"Стан воріт при невизначеному герконі: {gate_ctrl_instance.get_current_gate_state()}")
        assert gate_ctrl_instance.get_current_gate_state() == "BETWEEN_STATES"

        # Повертаємо геркон в нормальний стан для наступних тестів
        mock_sensor_mgr.reed_switch.set_gates_closed()


    except Exception as e:
        logger.error(f"Під час тестування сталася помилка: {e}", exc_info=True)
    finally:
        if gate_ctrl_instance:
            gate_ctrl_instance.cleanup()
            logger.info("GateController.cleanup() викликано.")

    logger.info("\n--- Тестування модуля gate_controller.py з герконом завершено ---")