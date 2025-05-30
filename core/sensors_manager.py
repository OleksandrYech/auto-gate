# core/sensors_manager.py
from gpiozero import DigitalInputDevice, DistanceSensor
from gpiozero.pins.pigpio import PiGPIOFactory 
import time
import logging
from typing import Optional 

logger = logging.getLogger(__name__)

try:
    factory = PiGPIOFactory()
    logger.info("Використовується PiGPIOFactory для керування пінами в sensors_manager.")
except Exception as e: 
    logger.warning(f"Не вдалося ініціалізувати PiGPIOFactory: {e}. "
                   "Буде використано фабрику пінів за замовчуванням. "
                   "Для DistanceSensor рекомендується pigpio.")
    factory = None

class ReedSwitch:
    def __init__(self, pin_number: int, name: str = "ReedSwitch"):
        self.name = name
        self.pin_number = pin_number
        self._logger = logging.getLogger(f"{__name__}.{self.name}_GPIO{self.pin_number}")
        self._device: Optional[DigitalInputDevice] = None
        try:
            self._device = DigitalInputDevice(pin_number, pull_up=True, pin_factory=factory)
            self._logger.info(f"Геркон '{self.name}' ініціалізовано на GPIO{self.pin_number}.")
            initial_state_msg = "ВІДКРИТІ (контакт замкнений)" if self._device.is_active else "ЗАКРИТІ (контакт розімкнений)"
            self._logger.info(f"Початковий стан '{self.name}': ВОРОТА {initial_state_msg}")
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати геркон '{self.name}' на GPIO{self.pin_number}: {e}", exc_info=True)

    @property
    def are_gates_open(self) -> Optional[bool]:
        if self._device is None: return None
        return self._device.is_active

    @property
    def are_gates_closed(self) -> Optional[bool]:
        if self._device is None: return None
        return not self._device.is_active

    def wait_for_open(self, timeout: Optional[float] = None) -> bool:
        if not self._device: self._logger.warning(f"'{self.name}': Датчик не ініціалізовано."); return False
        self._logger.debug(f"'{self.name}': Очікування відкриття (таймаут: {timeout}с)...")
        try:
            self._device.wait_for_active(timeout)
            self._logger.debug(f"'{self.name}': Виявлено відкриття.")
            return True
        except Exception: self._logger.debug(f"'{self.name}': Таймаут/помилка очікування відкриття."); return False

    def wait_for_close(self, timeout: Optional[float] = None) -> bool:
        if not self._device: self._logger.warning(f"'{self.name}': Датчик не ініціалізовано."); return False
        self._logger.debug(f"'{self.name}': Очікування закриття (таймаут: {timeout}с)...")
        try:
            self._device.wait_for_inactive(timeout)
            self._logger.debug(f"'{self.name}': Виявлено закриття.")
            return True
        except Exception: self._logger.debug(f"'{self.name}': Таймаут/помилка очікування закриття."); return False

    def cleanup(self):
        if self._device:
            try: self._device.close(); self._logger.info(f"Ресурси геркона '{self.name}' звільнено.")
            except Exception as e: self._logger.error(f"Помилка звільнення геркона '{self.name}': {e}", exc_info=True)
            self._device = None

class UltrasonicSensor:
    DEFAULT_THRESHOLD_VEHICLE_APPROACH = 1.0 
    DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR = 2.0 
    PASS_CONFIRMATION_TIME_S = 1.5
    DEFAULT_PASSAGE_OBSTRUCTION_THRESHOLD_M = 0.3 

    def __init__(self, trigger_pin: int, echo_pin: int, name: str = "UltrasonicSensor"):
        self.name = name
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self._logger = logging.getLogger(f"{__name__}.{self.name}_T{trigger_pin}E{echo_pin}")
        self._sensor: Optional[DistanceSensor] = None
        self.last_clear_time_monotonic: Optional[float] = None

        try:
            self._sensor = DistanceSensor(
                echo=echo_pin, trigger=trigger_pin, max_distance=4,
                pin_factory=factory, queue_len=3
            )
            self._logger.info(f"УЗД '{self.name}' ініціалізовано: Trig:GPIO{trigger_pin}, Echo:GPIO{echo_pin}.")
            time.sleep(0.5)
            initial_dist = self.get_distance()
            if initial_dist is not None:
                dist_str = f"{initial_dist:.2f} м" if initial_dist != float('inf') else "Поза діапазоном (> 4 м)"
                self._logger.info(f"'{self.name}': Початкова відстань: {dist_str}")
            else: self._logger.warning(f"'{self.name}': Не вдалося отримати початкову відстань.")
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати УЗД '{self.name}': {e}", exc_info=True)

    def get_distance(self) -> Optional[float]:
        if self._sensor is None: self._logger.warning(f"'{self.name}': УЗД не ініціалізовано."); return None
        try: return float(self._sensor.distance)
        except Exception as e: self._logger.warning(f"'{self.name}': Не вдалося прочитати відстань: {e}"); return float('inf')

    def is_vehicle_approaching(self, threshold_m: Optional[float] = None) -> bool:
        current_threshold = threshold_m if threshold_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_APPROACH
        dist = self.get_distance()
        if dist is None or dist == float('inf'): return False
        return dist < current_threshold

    def has_vehicle_passed(self, threshold_clear_m: Optional[float] = None, confirmation_time_s: Optional[float] = None) -> bool:
        current_threshold_clear = threshold_clear_m if threshold_clear_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR
        current_confirmation_time = confirmation_time_s if confirmation_time_s is not None else self.PASS_CONFIRMATION_TIME_S
        dist = self.get_distance()
        if dist is None: self.last_clear_time_monotonic = None; return False
        
        if dist > current_threshold_clear:
            if self.last_clear_time_monotonic is None:
                self.last_clear_time_monotonic = time.monotonic()
                self._logger.debug(f"'{self.name}': has_vehicle_passed - зона чиста (>{current_threshold_clear:.2f}м). Таймер підтвердження запущено.")
                return False 
            elif time.monotonic() - self.last_clear_time_monotonic >= current_confirmation_time:
                self._logger.debug(f"'{self.name}': has_vehicle_passed - проїзд підтверджено (чисто {current_confirmation_time:.1f}с).")
                self.last_clear_time_monotonic = None 
                return True
            return False
        else: 
            if self.last_clear_time_monotonic is not None:
                self._logger.debug(f"'{self.name}': has_vehicle_passed - об'єкт в зоні (<={current_threshold_clear:.2f}м). Таймер скинуто.")
            self.last_clear_time_monotonic = None
            return False

    def wait_for_approach(self, threshold_m: Optional[float] = None, timeout: Optional[float] = None) -> bool:
        if not self._sensor: self._logger.warning(f"'{self.name}': Датчик не ініціалізовано."); return False
        current_threshold = threshold_m if threshold_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_APPROACH
        self._logger.debug(f"'{self.name}': Очікування наближення (< {current_threshold:.2f}м, таймаут: {timeout}с)...")
        start_time = time.monotonic()
        while True:
            if self.is_vehicle_approaching(current_threshold): self._logger.debug(f"'{self.name}': Об'єкт наблизився."); return True
            if timeout is not None and (time.monotonic() - start_time) > timeout: self._logger.debug(f"'{self.name}': Таймаут наближення."); return False
            time.sleep(0.1)

    def wait_for_clear_after_pass(self, threshold_clear_m: Optional[float] = None, confirmation_s: Optional[float] = None, timeout: Optional[float] = None) -> bool:
        """Чекає, поки зона стане чистою з підтвердженням (логіка з sensors.py / ultrasonic_test.py)."""
        if not self._sensor: self._logger.warning(f"'{self.name}': Датчик не ініціалізовано."); return False
        # Параметри для has_vehicle_passed
        thresh_clear = threshold_clear_m if threshold_clear_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR
        confirm_s = confirmation_s if confirmation_s is not None else self.PASS_CONFIRMATION_TIME_S
        self._logger.debug(f"'{self.name}': Очікування проїзду та очищення зони (> {thresh_clear:.2f}м протягом {confirm_s:.1f}с, таймаут: {timeout}с)...")
        
        start_time_overall = time.monotonic()
        # Спочатку переконаємося, що об'єкт був або є (тобто зона не "завжди чиста")
        # Це може потребувати додаткової логіки або припущення, що цей метод викликається після детекції.
        # Для простоти, поточна реалізація has_vehicle_passed скидає таймер, якщо об'єкт близько.

        while True:
            if self.has_vehicle_passed(thresh_clear, confirm_s):
                self._logger.info(f"'{self.name}': Автомобіль проїхав, зона чиста (згідно has_vehicle_passed).")
                return True
            if timeout is not None and (time.monotonic() - start_time_overall) > timeout:
                self._logger.debug(f"'{self.name}': Таймаут очікування проїзду та очищення зони.")
                return False 
            time.sleep(0.1)

    def detect_object_in_passage(self, passage_threshold_m: Optional[float] = None) -> bool:
        current_threshold = passage_threshold_m if passage_threshold_m is not None else self.DEFAULT_PASSAGE_OBSTRUCTION_THRESHOLD_M
        dist = self.get_distance()
        if dist is None or dist == float('inf'): return False 
        return dist < current_threshold

    def wait_for_object_to_enter_passage(self, passage_threshold_m: Optional[float] = None, timeout_s: Optional[float] = None) -> bool:
        current_threshold = passage_threshold_m if passage_threshold_m is not None else self.DEFAULT_PASSAGE_OBSTRUCTION_THRESHOLD_M
        self._logger.debug(f"'{self.name}': Очікування ВХОДУ об'єкта в проїзд (< {current_threshold:.2f}м, таймаут: {timeout_s}с)...")
        start_time = time.monotonic()
        while True:
            if not self._sensor: self._logger.error(f"'{self.name}': Датчик не ініціалізовано."); return False
            if self.detect_object_in_passage(current_threshold):
                self._logger.info(f"'{self.name}': Об'єкт УВІЙШОВ у проїзд (відстань < {current_threshold:.2f}м).")
                return True
            if timeout_s is not None and (time.monotonic() - start_time) > timeout_s:
                self._logger.debug(f"'{self.name}': Таймаут очікування входу об'єкта в проїзд.")
                return False
            time.sleep(0.05) 

    def wait_for_object_to_clear_passage(self, passage_threshold_m: Optional[float] = None, timeout_s: Optional[float] = None) -> bool:
        current_threshold = passage_threshold_m if passage_threshold_m is not None else self.DEFAULT_PASSAGE_OBSTRUCTION_THRESHOLD_M
        self._logger.debug(f"'{self.name}': Очікування ВИХОДУ об'єкта з проїзду (> {current_threshold:.2f}м, таймаут: {timeout_s}с)...")
        start_time = time.monotonic()
        while True:
            if not self._sensor: self._logger.error(f"'{self.name}': Датчик не ініціалізовано."); return False
            is_object_present = self.detect_object_in_passage(current_threshold)
            if not is_object_present:
                dist_check = self.get_distance()
                if dist_check is not None and (dist_check > current_threshold or dist_check == float('inf')):
                    self._logger.info(f"'{self.name}': Об'єкт ПОКИНУВ проїзд (відстань > {current_threshold:.2f}м).")
                    return True
            if timeout_s is not None and (time.monotonic() - start_time) > timeout_s:
                self._logger.debug(f"'{self.name}': Таймаут очікування виходу об'єкта з проїзду.")
                return not self.detect_object_in_passage(current_threshold) 
            time.sleep(0.05)

    def cleanup(self):
        if self._sensor:
            try: self._sensor.close(); self._logger.info(f"Ресурси УЗД '{self.name}' звільнено.")
            except Exception as e: self._logger.error(f"Помилка звільнення УЗД '{self.name}': {e}", exc_info=True)
            self._sensor = None

class SensorManager:
    def __init__(self,
                 reed_pin: int,
                 ultrasonic_entry_trigger_pin: int,
                 ultrasonic_entry_echo_pin: int,
                 ultrasonic_exit_trigger_pin: Optional[int] = None,
                 ultrasonic_exit_echo_pin: Optional[int] = None,
                 reed_name: str = "GateReedSensor",
                 ultrasonic_entry_name: str = "EntryUltrasonicSensor",
                 ultrasonic_exit_name: str = "ExitUltrasonicSensor"):
        self._logger = logging.getLogger(f"{__name__}.SensorManager")
        self._logger.info("Ініціалізація SensorManager...")

        self.reed_switch: Optional[ReedSwitch] = None
        self.ultrasonic_sensor_entry: Optional[UltrasonicSensor] = None
        self.ultrasonic_sensor_exit: Optional[UltrasonicSensor] = None

        try: self.reed_switch = ReedSwitch(pin_number=reed_pin, name=reed_name) 
        except Exception as e: self._logger.error(f"Не вдалося ініціалізувати {reed_name}: {e}", exc_info=True)
        try:
            self.ultrasonic_sensor_entry = UltrasonicSensor(
                trigger_pin=ultrasonic_entry_trigger_pin,
                echo_pin=ultrasonic_entry_echo_pin, name=ultrasonic_entry_name) 
        except Exception as e: self._logger.error(f"Не вдалося ініціалізувати {ultrasonic_entry_name}: {e}", exc_info=True)

        if ultrasonic_exit_trigger_pin is not None and ultrasonic_exit_echo_pin is not None:
            try:
                self.ultrasonic_sensor_exit = UltrasonicSensor(
                    trigger_pin=ultrasonic_exit_trigger_pin,
                    echo_pin=ultrasonic_exit_echo_pin, name=ultrasonic_exit_name)
            except Exception as e: self._logger.error(f"Не вдалося ініціалізувати {ultrasonic_exit_name}: {e}", exc_info=True)
        else: self._logger.info("Піни для УЗД виїзду не надано, датчик виїзду не ініціалізовано.")
        self._logger.info("SensorManager ініціалізацію завершено (або спробовано).")

    def cleanup(self):
        self._logger.info("Очищення ресурсів SensorManager...")
        if self.reed_switch: self.reed_switch.cleanup()
        if self.ultrasonic_sensor_entry: self.ultrasonic_sensor_entry.cleanup()
        if self.ultrasonic_sensor_exit: self.ultrasonic_sensor_exit.cleanup()
        self._logger.info("Очищення ресурсів SensorManager завершено.")

    def __del__(self):
        self.cleanup()

if __name__ == '__main__':
    # ... (Тестовий блок з ultrasonic_test.py можна адаптувати сюди, 
    #      створюючи екземпляр SensorManager, а потім викликаючи методи 
    #      manager.ultrasonic_sensor_entry.назва_методу_для_тесту ) ...
    if not logging.getLogger().hasHandlers():
         logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s')
    logger.info("Тестування модуля sensors_manager.py (включаючи логіку УЗД з sensors.py)...")
    # ... (решта тестового коду з попередньої версії sensors_manager.py)
    TEST_REED_PIN = 22 
    TEST_US_ENTRY_TRIG = 23 
    TEST_US_ENTRY_ECHO = 24 
    
    PASSAGE_THRESHOLD_TEST = 0.3
    ENTER_TIMEOUT_TEST = 3
    CLEAR_TIMEOUT_TEST = 3

    try:
        manager = SensorManager(
            reed_pin=TEST_REED_PIN,
            ultrasonic_entry_trigger_pin=TEST_US_ENTRY_TRIG,
            ultrasonic_entry_echo_pin=TEST_US_ENTRY_ECHO
        )
        # ... (тести для геркона) ...
        if manager.ultrasonic_sensor_entry and (not hasattr(manager.ultrasonic_sensor_entry, '_sensor') or manager.ultrasonic_sensor_entry._sensor):
            us_sensor = manager.ultrasonic_sensor_entry
            logger.info(f"\n--- Тестування УЗД '{us_sensor.name}' ---")
            logger.info(f"  Поріг проїзду (detect_object_in_passage): {us_sensor.DEFAULT_PASSAGE_OBSTRUCTION_THRESHOLD_M}м")
            logger.info(f"  Поріг наближення (is_vehicle_approaching): {us_sensor.DEFAULT_THRESHOLD_VEHICLE_APPROACH}м")
            logger.info(f"  Поріг чистої зони (has_vehicle_passed): {us_sensor.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR}м")

            logger.info("\n  Тест wait_for_clear_after_pass (логіка з sensors.py):")
            logger.info("  Тримайте об'єкт близько (<1м), потім приберіть (>2м) на 1.5с. Таймаут тесту 10с.")
            input("  Натисніть Enter для старту тесту wait_for_clear_after_pass...")
            if us_sensor.wait_for_clear_after_pass(timeout=10):
                logger.info("    wait_for_clear_after_pass: УСПІХ - проїзд підтверджено.")
            else:
                logger.warning("    wait_for_clear_after_pass: ПОМИЛКА/ТАЙМАУТ.")
            
            logger.info(f"\n  Тест нових методів (поріг {PASSAGE_THRESHOLD_TEST}м):")
            logger.info(f"  Спочатку приберіть об'єкт, потім піднесіть (<{PASSAGE_THRESHOLD_TEST}м), потім знову приберіть.")
            input("  Натисніть Enter для старту тестів wait_for_object_to_enter/clear_passage...")
            
            logger.info("  Тест: wait_for_object_to_enter_passage...")
            if us_sensor.wait_for_object_to_enter_passage(PASSAGE_THRESHOLD_TEST, ENTER_TIMEOUT_TEST):
                logger.info("    УСПІХ: Об'єкт увійшов у проїзд.")
                logger.info("  Тест: wait_for_object_to_clear_passage...")
                if us_sensor.wait_for_object_to_clear_passage(PASSAGE_THRESHOLD_TEST, CLEAR_TIMEOUT_TEST):
                    logger.info("    УСПІХ: Об'єкт покинув проїзд.")
                else:
                    logger.warning("    ПОМИЛКА/ТАЙМАУТ: Об'єкт не покинув проїзд.")
            else:
                logger.warning("    ПОМИЛКА/ТАЙМАУТ: Об'єкт не увійшов у проїзд (для тесту clear).")
        manager.cleanup()
    except Exception as e:
        logger.error(f"Помилка під час тестування SensorManager: {e}", exc_info=True)
    logger.info("Тестування модуля sensors_manager.py завершено.")
