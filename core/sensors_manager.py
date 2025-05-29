# core/sensors_manager.py
from gpiozero import DigitalInputDevice, DistanceSensor
from gpiozero.pins.pigpio import PiGPIOFactory
import time
import logging
from typing import Optional  # Додано для Optional

# Налаштування логування для модуля sensors_manager
logger = logging.getLogger(__name__)

# --- PiGPIO Factory Setup ---
try:
    factory = PiGPIOFactory()
    logger.info("Використовується PiGPIOFactory для керування пінами в sensors_manager.")
except (OSError, NameError) as e:
    logger.warning(f"Демон pigpio не знайдено/запущено, або python3-pigpio не встановлено (помилка: {e}). "
                   "Повернення до фабрики пінів за замовчуванням. "
                   "Для DistanceSensor рекомендується pigpio для кращої стабільності.")
    factory = None


# --- Клас для Геркона (Reed Switch) ---
class ReedSwitch:
    def __init__(self, pin_number: int, name: str = "ReedSwitch"):
        self.name = name
        self.pin_number = pin_number
        self._logger = logging.getLogger(f"{__name__}.{self.name}_GPIO{self.pin_number}")
        self._device: Optional[DigitalInputDevice] = None
        try:
            self._device = DigitalInputDevice(pin_number, pull_up=True, pin_factory=factory)
            self._logger.info(f"Геркон '{self.name}' ініціалізовано на GPIO{self.pin_number}.")
            if self._device.is_active:
                self._logger.info(f"Початковий стан '{self.name}': ВОРОТА ВІДКРИТІ (контакт замкнений, пін LOW)")
            else:
                self._logger.info(f"Початковий стан '{self.name}': ВОРОТА ЗАКРИТІ (контакт розімкнений, пін HIGH)")
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати геркон '{self.name}' на GPIO{self.pin_number}: {e}",
                               exc_info=True)

    @property
    def are_gates_open(self) -> Optional[bool]:
        if self._device is None:
            self._logger.warning(f"'{self.name}': Спроба перевірити стан неініціалізованого геркона.")
            return None
        return self._device.is_active

    @property
    def are_gates_closed(self) -> Optional[bool]:
        if self._device is None:
            self._logger.warning(f"'{self.name}': Спроба перевірити стан неініціалізованого геркона.")
            return None
        return not self._device.is_active

    def wait_for_open(self, timeout: Optional[float] = None) -> bool:
        if self._device:
            self._logger.debug(f"'{self.name}': Очікування відкриття воріт (таймаут: {timeout}с)...")
            try:
                self._device.wait_for_active(timeout)
                self._logger.debug(f"'{self.name}': Виявлено відкриття воріт.")
                return True
            except Exception:  # Наприклад, gpiozero.exc.GPIOZeroTimeoutError
                self._logger.debug(f"'{self.name}': Таймаут або помилка очікування відкриття.")
                return False
        self._logger.warning(f"'{self.name}': Неможливо очікувати відкриття, датчик не ініціалізовано.")
        return False

    def wait_for_close(self, timeout: Optional[float] = None) -> bool:
        if self._device:
            self._logger.debug(f"'{self.name}': Очікування закриття воріт (таймаут: {timeout}с)...")
            try:
                self._device.wait_for_inactive(timeout)
                self._logger.debug(f"'{self.name}': Виявлено закриття воріт.")
                return True
            except Exception:
                self._logger.debug(f"'{self.name}': Таймаут або помилка очікування закриття.")
                return False
        self._logger.warning(f"'{self.name}': Неможливо очікувати закриття, датчик не ініціалізовано.")
        return False

    def cleanup(self):
        if self._device:
            try:
                self._device.close()
                self._logger.info(f"Ресурси геркона '{self.name}' (GPIO{self.pin_number}) звільнено.")
            except Exception as e:
                self._logger.error(f"Помилка під час звільнення ресурсів геркона '{self.name}': {e}", exc_info=True)
            self._device = None


# --- Клас для Ультразвукового датчика ---
class UltrasonicSensor:
    DEFAULT_PASSAGE_THRESHOLD_M = 0.3  # Поріг для визначення об'єкта в проїзді
    DEFAULT_APPROACH_THRESHOLD_M = 1.0
    DEFAULT_CLEAR_ZONE_THRESHOLD_M = 2.0
    DEFAULT_CLEAR_CONFIRMATION_S = 1.5

    def __init__(self, trigger_pin: int, echo_pin: int, name: str = "UltrasonicSensor"):
        self.name = name
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self._logger = logging.getLogger(f"{__name__}.{self.name}_T{trigger_pin}E{echo_pin}")
        self._sensor: Optional[DistanceSensor] = None
        self._last_clear_time_monotonic: Optional[float] = None  # Для старої логіки has_vehicle_passed

        try:
            self._sensor = DistanceSensor(
                echo=echo_pin, trigger=trigger_pin, max_distance=4,
                pin_factory=factory, queue_len=3
            )
            self._logger.info(f"УЗД '{self.name}' ініціалізовано на Trigger:GPIO{trigger_pin}, Echo:GPIO{echo_pin}.")
            time.sleep(0.5)
            current_dist = self.get_distance()
            if current_dist is not None:
                self._logger.info(f"'{self.name}': Початкова відстань: {current_dist:.2f} м")
            else:
                self._logger.warning(f"'{self.name}': Не вдалося отримати початкову відстань.")
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати УЗД '{self.name}': {e}", exc_info=True)

    def get_distance(self) -> Optional[float]:
        if self._sensor is None:
            self._logger.warning(f"'{self.name}': Спроба отримати відстань з неініціалізованого УЗД.")
            return None
        try:
            distance = self._sensor.distance
            return float(distance)
        except Exception as e:
            self._logger.warning(f"'{self.name}': Не вдалося прочитати відстань: {e}")
            return None  # Повертаємо None у разі помилки читання, щоб відрізнити від float('inf')

    def is_vehicle_approaching(self, threshold_m: Optional[float] = None) -> bool:
        """Загальна функція для перевірки наближення (використовує старий поріг)."""
        current_threshold = threshold_m if threshold_m is not None else self.DEFAULT_APPROACH_THRESHOLD_M
        dist = self.get_distance()
        if dist is None: return False
        return dist < current_threshold

    def detect_object_in_passage(self, passage_threshold_m: Optional[float] = None) -> bool:
        """Перевіряє, чи є об'єкт ЗАРАЗ у проїзді (ближче за 'passage_threshold_m')."""
        current_threshold = passage_threshold_m if passage_threshold_m is not None else self.DEFAULT_PASSAGE_THRESHOLD_M
        dist = self.get_distance()
        if dist is None:
            self._logger.debug(f"'{self.name}': detect_object_in_passage - get_distance() повернув None.")
            return False
            # Якщо float('inf'), об'єкта немає близько
        if dist == float('inf'):
            return False
        return dist < current_threshold

    def wait_for_object_to_enter_passage(self, passage_threshold_m: Optional[float] = None,
                                         timeout_s: Optional[float] = None) -> bool:
        """Очікує, поки об'єкт не увійде в зону проїзду."""
        current_threshold = passage_threshold_m if passage_threshold_m is not None else self.DEFAULT_PASSAGE_THRESHOLD_M
        self._logger.debug(
            f"'{self.name}': Очікування входу об'єкта в проїзд (< {current_threshold:.2f}м, таймаут: {timeout_s}с)...")
        start_time = time.monotonic()
        while True:
            if self._sensor is None: self._logger.error(f"'{self.name}': Датчик не ініціалізовано."); return False
            if self.detect_object_in_passage(current_threshold):
                self._logger.info(f"'{self.name}': Об'єкт увійшов у проїзд (відстань < {current_threshold:.2f}м).")
                return True
            if timeout_s is not None and (time.monotonic() - start_time) > timeout_s:
                self._logger.debug(f"'{self.name}': Таймаут очікування входу об'єкта в проїзд.")
                return False
            time.sleep(0.05)

    def wait_for_object_to_clear_passage(self, passage_threshold_m: Optional[float] = None,
                                         timeout_s: Optional[float] = None) -> bool:
        """Очікує, поки об'єкт не покине зону проїзду. Припускає, що об'єкт вже був у зоні."""
        current_threshold = passage_threshold_m if passage_threshold_m is not None else self.DEFAULT_PASSAGE_THRESHOLD_M
        self._logger.debug(
            f"'{self.name}': Очікування виходу об'єкта з проїзду (> {current_threshold:.2f}м, таймаут: {timeout_s}с)...")
        start_time = time.monotonic()
        while True:
            if self._sensor is None: self._logger.error(f"'{self.name}': Датчик не ініціалізовано."); return False

            # Перевіряємо, чи зона стала вільною
            is_object_present = self.detect_object_in_passage(current_threshold)
            if not is_object_present:
                # Щоб підтвердити, що зона дійсно вільна (а не помилка датчика), перевіримо відстань ще раз
                dist_check = self.get_distance()
                if dist_check is not None and (dist_check > current_threshold or dist_check == float('inf')):
                    self._logger.info(f"'{self.name}': Об'єкт покинув проїзд (відстань > {current_threshold:.2f}м).")
                    return True
                # Якщо get_distance()=None, це помилка читання, продовжуємо чекати або виходимо за таймаутом

            if timeout_s is not None and (time.monotonic() - start_time) > timeout_s:
                self._logger.debug(f"'{self.name}': Таймаут очікування виходу об'єкта з проїзду.")
                # Повертаємо True, якщо об'єкта немає після таймауту, False - якщо все ще є
                return not self.detect_object_in_passage(current_threshold)
            time.sleep(0.05)

    def cleanup(self):
        if self._sensor:
            try:
                self._sensor.close()
                self._logger.info(f"Ресурси УЗД '{self.name}' (T:{self.trigger_pin},E:{self.echo_pin}) звільнено.")
            except Exception as e:
                self._logger.error(f"Помилка під час звільнення ресурсів УЗД '{self.name}': {e}", exc_info=True)
            self._sensor = None


# --- Клас Менеджера Датчиків ---
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

        try:
            self.reed_switch = ReedSwitch(pin_number=reed_pin, name=reed_name)
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати {reed_name}: {e}", exc_info=True)
        try:
            self.ultrasonic_sensor_entry = UltrasonicSensor(
                trigger_pin=ultrasonic_entry_trigger_pin,
                echo_pin=ultrasonic_entry_echo_pin,
                name=ultrasonic_entry_name
            )
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати {ultrasonic_entry_name}: {e}", exc_info=True)

        if ultrasonic_exit_trigger_pin is not None and ultrasonic_exit_echo_pin is not None:
            try:
                self.ultrasonic_sensor_exit = UltrasonicSensor(
                    trigger_pin=ultrasonic_exit_trigger_pin,
                    echo_pin=ultrasonic_exit_echo_pin,
                    name=ultrasonic_exit_name
                )
            except Exception as e:
                self._logger.error(f"Не вдалося ініціалізувати {ultrasonic_exit_name}: {e}", exc_info=True)
        else:
            self._logger.info("Піни для УЗД виїзду не надано, датчик виїзду не ініціалізовано.")
        self._logger.info("SensorManager ініціалізацію завершено (або спробовано).")

    def cleanup(self):
        self._logger.info("Очищення ресурсів SensorManager...")
        if self.reed_switch: self.reed_switch.cleanup()
        if self.ultrasonic_sensor_entry: self.ultrasonic_sensor_entry.cleanup()
        if self.ultrasonic_sensor_exit: self.ultrasonic_sensor_exit.cleanup()
        self._logger.info("Очищення ресурсів SensorManager завершено.")

    def __del__(self):
        self.cleanup()


# --- Приклад використання (для тестування модуля окремо) ---
if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s')

    logger.info("Тестування модуля sensors_manager.py з новою логікою УЗД...")

    TEST_REED_PIN = 22
    TEST_US_ENTRY_TRIG = 23
    TEST_US_ENTRY_ECHO = 24

    try:
        manager = SensorManager(
            reed_pin=TEST_REED_PIN,
            ultrasonic_entry_trigger_pin=TEST_US_ENTRY_TRIG,
            ultrasonic_entry_echo_pin=TEST_US_ENTRY_ECHO
        )

        if manager.reed_switch and manager.reed_switch._device:
            logger.info("\nТестування геркона:")
            logger.info(f"  Ворота відкриті? {manager.reed_switch.are_gates_open}")
            logger.info(f"  Ворота закриті? {manager.reed_switch.are_gates_closed}")

        if manager.ultrasonic_sensor_entry and manager.ultrasonic_sensor_entry._sensor:
            us_sensor = manager.ultrasonic_sensor_entry
            passage_thresh = 0.3  # Тестовий поріг 0.3м

            logger.info(f"\nТестування УЗД '{us_sensor.name}' з порогом проїзду {passage_thresh}м:")

            # Імітація: зона спочатку вільна
            logger.info("  Сценарій 1: Об'єкт не входить у зону (таймаут)")
            # us_sensor.set_distance_mock(2.0) # Якщо є мок-метод для встановлення відстані
            result_enter = us_sensor.wait_for_object_to_enter_passage(passage_thresh, timeout_s=2)
            logger.info(f"    Результат очікування входу: {result_enter} (очікується False)")

            logger.info("\n  Сценарій 2: Об'єкт входить, потім виходить")
            # print("  Будь ласка, наблизьте об'єкт до датчика (<0.3м)...")
            # time.sleep(5)
            # if us_sensor.detect_object_in_passage(passage_thresh):
            #    logger.info("    Об'єкт увійшов.")
            #    print("  Тепер віддаліть об'єкт (>0.3м)...")
            #    time.sleep(5)
            #    if not us_sensor.detect_object_in_passage(passage_thresh):
            #        logger.info("    Об'єкт покинув зону.")
            #    else:
            #        logger.warning("    Об'єкт все ще в зоні після віддалення.")
            # else:
            #    logger.warning("    Об'єкт не увійшов у зону.")

            logger.info(
                "  Для повного тестування wait_for_object_to_enter/clear_passage потрібна імітація зміни відстані.")

        manager.cleanup()
    except Exception as e:
        logger.error(f"Помилка під час тестування SensorManager: {e}", exc_info=True)

    logger.info("Тестування модуля sensors_manager.py завершено.")