# core/gate_controller.py
import time
import logging
import threading
from gpiozero import OutputDevice  # Для керування реле

logger = logging.getLogger(__name__)

# --- Конфігурація за замовчуванням ---
DEFAULT_OPEN_RELAY_PIN = 17  # Приклад: GPIO17 (BCM нумерація)
DEFAULT_CLOSE_RELAY_PIN = 27  # Приклад: GPIO27
DEFAULT_RELAY_PULSE_DURATION_S = 0.5  # 0.5 секунди (імітація натискання кнопки)
DEFAULT_AUTO_CLOSE_TIMEOUT_S = 30  # 30 секунд для автоматичного закриття
DEFAULT_CLOSING_OBSTRUCTION_THRESHOLD_M = 0.3  # Поріг для УЗД перед закриттям
DEFAULT_REED_CONFIRMATION_TIMEOUT_S = 5  # Секунд очікування підтвердження від геркона


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
                 closing_obstruction_threshold_m: float = DEFAULT_CLOSING_OBSTRUCTION_THRESHOLD_M,
                 reed_confirmation_timeout_s: float = DEFAULT_REED_CONFIRMATION_TIMEOUT_S):

        self.sensor_manager = sensor_manager_instance
        self.open_relay_pin = open_relay_pin
        self.close_relay_pin = close_relay_pin
        self.relay_pulse_duration_s = relay_pulse_duration_s
        self.auto_close_timeout_s = auto_close_timeout_s
        self.closing_obstruction_threshold_m = closing_obstruction_threshold_m
        self.reed_confirmation_timeout_s = reed_confirmation_timeout_s

        # Логгер для екземпляра
        self._logger = logging.getLogger(f"{__name__}.GateController_O{open_relay_pin}C{close_relay_pin}")

        # Ініціалізація реле (для low-level реле active_high=False)
        try:
            self.open_relay = OutputDevice(self.open_relay_pin, active_high=False, initial_value=True)
            self.close_relay = OutputDevice(self.close_relay_pin, active_high=False, initial_value=True)
            self._logger.info(f"Реле ВІДКРИТТЯ на GPIO{self.open_relay_pin} (active_low).")
            self._logger.info(f"Реле ЗАКРИТТЯ на GPIO{self.close_relay_pin} (active_low).")
            self.relays_initialized = True
        except Exception as e:
            self._logger.critical(f"Критична помилка ініціалізації реле: {e}", exc_info=True)
            self.relays_initialized = False  # Важливо для перевірки перед використанням реле

        self._auto_close_timer: Optional[threading.Timer] = None
        self._close_procedure_interrupted: bool = False
        self._lock = threading.Lock()  # Для синхронізації доступу до таймера та прапорця переривання

        self._commanded_state: str = "IDLE"  # Останній намір: 'OPENING', 'CLOSING', 'IDLE'
        self.gate_state: str = self._determine_state_from_reed_switch()  # Поточний стан

        self._logger.info(f"GateController ініціалізовано. Початковий стан (геркон): {self.gate_state}")
        self._logger.info(
            f"  Тривалість імпульсу: {self.relay_pulse_duration_s}с, Авто-закриття: {self.auto_close_timeout_s}с")
        self._logger.info(
            f"  Поріг перешкоди: {self.closing_obstruction_threshold_m}м, Таймаут геркона: {self.reed_confirmation_timeout_s}с")

    def _determine_state_from_reed_switch(self) -> str:
        if not self.sensor_manager or not hasattr(self.sensor_manager, 'reed_switch') or \
                not self.sensor_manager.reed_switch or self.sensor_manager.reed_switch._device is None:
            self._logger.warning("Геркон недоступний або не ініціалізований. Стан воріт невідомий.")
            return "UNKNOWN_SENSOR_UNAVAILABLE"
        try:
            if self.sensor_manager.reed_switch.are_gates_open:
                return "OPEN"
            elif self.sensor_manager.reed_switch.are_gates_closed:
                return "CLOSED"
            else:
                return "BETWEEN_STATES"  # Геркон не в крайньому положенні
        except Exception as e:
            self._logger.error(f"Помилка читання стану геркона: {e}", exc_info=True)
            return "UNKNOWN_SENSOR_READ_ERROR"

    def _activate_relay_pulse(self, relay_device: OutputDevice, action_name: str):
        if not self.relays_initialized or not relay_device:
            self._logger.error(f"Реле для '{action_name}' не ініціалізовано. Дію скасовано.")
            return
        self._logger.debug(f"Активація реле '{action_name}' на піні GPIO{relay_device.pin.number}...")
        try:
            relay_device.on()  # Активує реле (LOW для active_high=False)
            time.sleep(self.relay_pulse_duration_s)
            relay_device.off()  # Деактивує реле (HIGH)
            self._logger.info(f"Реле '{action_name}' (GPIO{relay_device.pin.number}) активовано імпульсом.")
        except Exception as e:
            self._logger.error(f"Помилка під час активації реле '{action_name}': {e}", exc_info=True)

    def open_gate(self) -> bool:
        if not self.relays_initialized:
            self._logger.error("Неможливо відкрити ворота: реле не ініціалізовано.")
            return False

        current_actual_state = self._determine_state_from_reed_switch()
        if current_actual_state == "OPEN":
            self._logger.info("Ворота вже відкриті (геркон). Команда ВІДКРИТИ ігнорується.")
            self._commanded_state = "IDLE"  # Скидаємо намір, якщо мета досягнута
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

        if self.reed_confirmation_timeout_s > 0 and self.sensor_manager and \
                hasattr(self.sensor_manager, 'reed_switch') and self.sensor_manager.reed_switch and \
                self.sensor_manager.reed_switch._device is not None:
            self._logger.debug(
                f"Очікування підтвердження відкриття від геркона ({self.reed_confirmation_timeout_s}с)...")
            try:
                self.sensor_manager.reed_switch.wait_for_open(timeout=self.reed_confirmation_timeout_s)
                if self.sensor_manager.reed_switch.are_gates_open:
                    self.gate_state = "OPEN"
                    self._commanded_state = "IDLE"
                    self._logger.info("Ворота підтверджено відкритими (геркон).")
                else:
                    self.gate_state = "MOVEMENT_TIMEOUT_OPEN"
                    self._logger.warning("Тайм-аут очікування відкриття воріт (геркон) або не вдалося відкрити.")
            except Exception as e:
                self._logger.error(f"Помилка під час очікування відкриття герконом: {e}")
                self.gate_state = self._determine_state_from_reed_switch()
        return True

    def is_passage_clear_for_closing(self) -> bool:
        # Використовуємо УЗД, що контролює проріз воріт (припускаємо, це ultrasonic_sensor_entry)
        passage_sensor = getattr(self.sensor_manager, 'ultrasonic_sensor_entry', None)
        if not passage_sensor or passage_sensor._sensor is None:
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
        # float('inf') означає, що об'єкт дуже далеко або відсутній - тобто проїзд вільний
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
                self.sensor_manager.reed_switch._device is not None:
            self._logger.debug(
                f"Очікування підтвердження закриття від геркона ({self.reed_confirmation_timeout_s}с)...")
            try:
                self.sensor_manager.reed_switch.wait_for_close(timeout=self.reed_confirmation_timeout_s)
                if self.sensor_manager.reed_switch.are_gates_closed:
                    self.gate_state = "CLOSED"
                    self._commanded_state = "IDLE"
                    self._logger.info("Ворота підтверджено закритими (геркон).")
                else:
                    self.gate_state = "MOVEMENT_TIMEOUT_CLOSE"
                    self._logger.warning("Тайм-аут очікування закриття воріт (геркон) або не вдалося закрити.")
            except Exception as e:
                self._logger.error(f"Помилка під час очікування закриття герконом: {e}")
                self.gate_state = self._determine_state_from_reed_switch()
        return True

    def _auto_close_gate_callback(self):
        self._logger.info("Таймер авто-закриття спрацював. Спроба закрити ворота.")
        self.close_gate()  # Викличе is_passage_clear_for_closing та перевірку _close_procedure_interrupted

    def start_auto_close_timer(self, timeout_s: Optional[float] = None):
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
            self.gate_state = self._determine_state_from_reed_switch()
            self._commanded_state = "IDLE"

    def get_current_gate_state(self) -> str:
        # Пріоритет - актуальний стан з геркона
        reed_state = self._determine_state_from_reed_switch()
        if reed_state in ["OPEN", "CLOSED"]:
            self.gate_state = reed_state
            # Якщо геркон показує фінальний стан, а команда ще "в процесі", оновлюємо commanded_state
            if (self.gate_state == "OPEN" and self._commanded_state == "OPENING") or \
                    (self.gate_state == "CLOSED" and self._commanded_state == "CLOSING"):
                self._commanded_state = "IDLE"
            return self.gate_state

        # Якщо геркон не в крайньому положенні або помилка
        if self.gate_state == "OBSTRUCTED": return "OBSTRUCTED"  # Перешкода має вищий пріоритет
        if self._commanded_state in ["OPENING", "CLOSING"]: return self._commanded_state  # Відображаємо намір руху
        if reed_state == "BETWEEN_STATES": return "BETWEEN_STATES"  # Якщо немає команди, але ворота між станами

        return self.gate_state  # Повертаємо останній відомий/встановлений стан

    def cleanup(self):
        self._logger.info("Очищення ресурсів GateController...")
        self.cancel_auto_close_timer()
        if self.relays_initialized:
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
    from unittest.mock import MagicMock  # Потрібен для більш складних моків, якщо потрібно

    logger_test = logging.getLogger(f"{__name__}_Test")
    logger_test.info("--- Тестування модуля gate_controller.py (з моками) ---")


    # Мок для GPIO пристрою (для ReedSwitch)
    class MockGpioDevice:
        def __init__(self, pin, pull_up=None): self.pin = pin; self.is_active = False; self.value = 1

        def close(self): logger_test.debug(f"MockGpioDevice pin {self.pin} closed")

        def wait_for_active(self, timeout): logger_test.debug(f"Mock: wait_for_active on pin {self.pin}")

        def wait_for_inactive(self, timeout): logger_test.debug(f"Mock: wait_for_inactive on pin {self.pin}")


    # Мок для SensorManager та його компонентів
    mock_sensor_manager = MagicMock()
    mock_sensor_manager.reed_switch = MagicMock()
    mock_sensor_manager.reed_switch._device = MockGpioDevice(DEFAULT_REED_CONFIRMATION_TIMEOUT_S)  # фіктивний пін
    mock_sensor_manager.reed_switch.are_gates_open = False  # Початковий стан - закрито
    mock_sensor_manager.reed_switch.are_gates_closed = True

    mock_sensor_manager.ultrasonic_sensor_entry = MagicMock()
    mock_sensor_manager.ultrasonic_sensor_entry._sensor = True  # Імітуємо, що сенсор ініціалізовано
    mock_sensor_manager.ultrasonic_sensor_entry.get_distance.return_value = 2.0  # Проїзд вільний

    # Ініціалізація GateController з мок-сенсорами
    gate_ctrl = GateController(
        sensor_manager_instance=mock_sensor_manager,
        open_relay_pin=20,  # Використовуйте тестові піни
        close_relay_pin=21,
        auto_close_timeout_s=3,  # Короткий для тесту
        reed_confirmation_timeout_s=1
    )

    if not gate_ctrl.relays_initialized:
        logger_test.error("Реле не були ініціалізовані, тестування неможливе.")
    else:
        logger_test.info(f"Початковий стан воріт: {gate_ctrl.get_current_gate_state()}")

        logger_test.info("Тест: Відкриття воріт...")
        mock_sensor_manager.reed_switch.are_gates_open = True  # Імітуємо, що геркон спрацював
        mock_sensor_manager.reed_switch.are_gates_closed = False
        gate_ctrl.open_gate()
        logger_test.info(f"Стан після команди відкрити: {gate_ctrl.get_current_gate_state()}")  # Має бути OPEN

        time.sleep(0.5)

        logger_test.info("Тест: Закриття воріт (проїзд вільний)...")
        mock_sensor_manager.ultrasonic_sensor_entry.get_distance.return_value = 2.0  # Вільний
        mock_sensor_manager.reed_switch.are_gates_open = False
        mock_sensor_manager.reed_switch.are_gates_closed = True  # Імітуємо закриття
        gate_ctrl.close_gate()
        logger_test.info(f"Стан після команди закрити: {gate_ctrl.get_current_gate_state()}")  # Має бути CLOSED

        time.sleep(0.5)

        logger_test.info("Тест: Спроба закрити (проїзд заблоковано)...")
        gate_ctrl.gate_state = "OPEN"  # Припустимо, ворота відкриті
        mock_sensor_manager.reed_switch.are_gates_open = True
        mock_sensor_manager.reed_switch.are_gates_closed = False
        mock_sensor_manager.ultrasonic_sensor_entry.get_distance.return_value = 0.1  # Перешкода!
        gate_ctrl.close_gate()
        logger_test.info(
            f"Стан після спроби закрити з перешкодою: {gate_ctrl.get_current_gate_state()}")  # Має бути OBSTRUCTED

        time.sleep(0.5)

        logger_test.info("Тест: Авто-закриття...")
        gate_ctrl.gate_state = "OPEN"  # Ворота відкриті
        mock_sensor_manager.reed_switch.are_gates_open = True
        mock_sensor_manager.reed_switch.are_gates_closed = False
        mock_sensor_manager.ultrasonic_sensor_entry.get_distance.return_value = 2.0  # Проїзд знову вільний

        gate_ctrl.start_auto_close_timer()
        logger_test.info("Таймер авто-закриття запущено. Очікування...")
        time.sleep(3.5)  # Чекаємо довше за таймаут таймера + час на "спрацювання" геркона
        mock_sensor_manager.reed_switch.are_gates_open = False  # Імітуємо, що геркон спрацював на закриття
        mock_sensor_manager.reed_switch.are_gates_closed = True
        # Стан має оновитися на CLOSED або через callback таймера, або через наступний get_current_gate_state
        logger_test.info(f"Стан після авто-закриття: {gate_ctrl.get_current_gate_state()}")

        gate_ctrl.cleanup()

    logger_test.info("--- Тестування модуля gate_controller.py завершено ---")