# core/sensors_manager.py
from gpiozero import DigitalInputDevice, DistanceSensor
from gpiozero.pins.pigpio import PiGPIOFactory  # Для більш стабільної роботи, особливо з DistanceSensor
import time
import logging

# Налаштування логування для модуля sensors_manager
logger = logging.getLogger(__name__)

# --- PiGPIO Factory Setup ---
try:
    factory = PiGPIOFactory()
    logger.info("Використовується PiGPIOFactory для керування пінами в sensors_manager.")
except (OSError, NameError) as e:
    logger.warning(f"Демон pigpio не знайдено, не запущено, або python3-pigpio не встановлено (помилка: {e}). "
                   "Повернення до фабрики пінів за замовчуванням. "
                   "Для DistanceSensor рекомендується pigpio для кращої стабільності.")
    factory = None  # gpiozero обере фабрику за замовчуванням


# --- Клас для Геркона (Reed Switch) ---
class ReedSwitch:
    """
    Клас для роботи з герконовим датчиком MC-38.
    Вказує, чи ворота відкриті або закриті.
    """

    def __init__(self, pin_number, name="ReedSwitch"):
        """
        Ініціалізація герконового датчика.

        Args:
            pin_number (int): Номер GPIO піна (BCM нумерація), до якого підключений датчик.
            name (str): Ім'я датчика для логування.
        """
        self.name = name
        self.pin_number = pin_number
        # Логгер для конкретного екземпляра датчика
        self._logger = logging.getLogger(f"{__name__}.{self.name}_GPIO{self.pin_number}")
        self._device = None  # Ініціалізуємо _device як None
        try:
            # pull_up=True: Коли геркон розімкнений (магніт далеко, ворота закриті), пін підтягнутий до HIGH.
            # Коли геркон замкнений (магніт близько, ворота відкриті), він замикає пін на GND, і пін стає LOW.
            # Отже, is_active (LOW) буде означати "ворота відкриті".
            self._device = DigitalInputDevice(pin_number, pull_up=True, pin_factory=factory)  #
            self._logger.info(f"Геркон '{self.name}' ініціалізовано на GPIO{self.pin_number}.")
            # Початкове зчитування стану
            if self._device.is_active:  # is_active == True, якщо пін LOW (геркон замкнений)
                self._logger.info(f"Початковий стан '{self.name}': ВОРОТА ВІДКРИТІ (контакт замкнений, пін LOW)")
            else:  # is_active == False, якщо пін HIGH (геркон розімкнений)
                self._logger.info(f"Початковий стан '{self.name}': ВОРОТА ЗАКРИТІ (контакт розімкнений, пін HIGH)")
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати геркон '{self.name}' на GPIO{self.pin_number}: {e}",
                               exc_info=True)
            self._device = None  # Залишаємо None у разі помилки

    @property
    def are_gates_open(self):
        """
        Перевіряє, чи ворота відкриті.

        Returns:
            bool: True, якщо ворота відкриті (геркон замкнений), False в іншому випадку.
                  None, якщо датчик не ініціалізований.
        """
        if self._device is None:
            self._logger.warning(f"'{self.name}': Спроба перевірити стан неініціалізованого геркона.")
            return None
        # Якщо pull_up=True, то:
        # Геркон замкнений (ворота відкриті) -> пін LOW -> self._device.value == 0 -> self._device.is_active == True
        # Геркон розімкнений (ворота закриті) -> пін HIGH -> self._device.value == 1 -> self._device.is_active == False
        return self._device.is_active  #

    @property
    def are_gates_closed(self):
        """
        Перевіряє, чи ворота закриті.

        Returns:
            bool: True, якщо ворота закриті (геркон розімкнений), False в іншому випадку.
                  None, якщо датчик не ініціалізований.
        """
        if self._device is None:
            self._logger.warning(f"'{self.name}': Спроба перевірити стан неініціалізованого геркона.")
            return None
        return not self._device.is_active  # Протилежно до are_gates_open

    def wait_for_open(self, timeout=None):
        """Чекає, поки ворота відкриються (геркон замкнеться)."""
        if self._device:
            self._logger.debug(f"'{self.name}': Очікування відкриття воріт (таймаут: {timeout}с)...")
            self._device.wait_for_active(timeout)  # is_active (LOW) -> ворота відкриті
            self._logger.debug(f"'{self.name}': Виявлено відкриття воріт.")
        else:
            self._logger.warning(f"'{self.name}': Неможливо очікувати відкриття, датчик не ініціалізовано.")

    def wait_for_close(self, timeout=None):
        """Чекає, поки ворота закриються (геркон розімкнеться)."""
        if self._device:
            self._logger.debug(f"'{self.name}': Очікування закриття воріт (таймаут: {timeout}с)...")
            self._device.wait_for_inactive(timeout)  # is_inactive (HIGH) -> ворота закриті
            self._logger.debug(f"'{self.name}': Виявлено закриття воріт.")
        else:
            self._logger.warning(f"'{self.name}': Неможливо очікувати закриття, датчик не ініціалізовано.")

    def cleanup(self):
        """Звільняє ресурси датчика."""
        if self._device:
            try:
                self._device.close()
                self._logger.info(f"Ресурси геркона '{self.name}' (GPIO{self.pin_number}) звільнено.")
            except Exception as e:
                self._logger.error(f"Помилка під час звільнення ресурсів геркона '{self.name}': {e}", exc_info=True)
            self._device = None


# --- Клас для Ультразвукового датчика ---
class UltrasonicSensor:
    """
    Клас для роботи з ультразвуковим датчиком відстані AJ-SPO4M (або сумісним HC-SR04).
    """
    # Відстані в метрах
    DEFAULT_THRESHOLD_VEHICLE_APPROACH = 1.0  # 1 метр для фіксації під'їзду
    DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR = 2.0  # 2 метри, щоб вважати, що авто проїхало і зона вільна
    PASS_CONFIRMATION_TIME_S = 1.5  # Секунди, протягом яких зона має бути вільною, щоб підтвердити проїзд

    def __init__(self, trigger_pin, echo_pin, name="UltrasonicSensor"):
        """
        Ініціалізація ультразвукового датчика.

        Args:
            trigger_pin (int): Номер GPIO піна (BCM) для Trigger.
            echo_pin (int): Номер GPIO піна (BCM) для Echo.
            name (str): Ім'я датчика для логування.
        """
        self.name = name
        self.trigger_pin = trigger_pin
        self.echo_pin = echo_pin
        self._logger = logging.getLogger(f"{__name__}.{self.name}_T{trigger_pin}E{echo_pin}")
        self.last_clear_time_monotonic = None  #
        self._sensor = None  # Ініціалізуємо _sensor як None
        try:
            self._sensor = DistanceSensor(
                echo=echo_pin,
                trigger=trigger_pin,
                max_distance=4,  # Макс. вимірювана відстань - 4 метри
                pin_factory=factory,  #
                queue_len=3  # Усереднення по 3 вимірам для стабільності
            )
            self._logger.info(f"УЗД '{self.name}' ініціалізовано на Trigger:GPIO{trigger_pin}, Echo:GPIO{echo_pin}.")
            time.sleep(0.5)  # Дати датчику трохи часу на стабілізацію
            self._logger.info(f"'{self.name}': Початкова відстань: {self.get_distance():.2f} м")
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати УЗД '{self.name}': {e}", exc_info=True)
            self._sensor = None

    def get_distance(self):
        """
        Повертає поточну виміряну відстань в метрах.

        Returns:
            float: Відстань в метрах, або float('inf') у разі помилки/відсутності об'єкта.
                   None, якщо датчик не ініціалізований.
        """
        if self._sensor is None:
            self._logger.warning(f"'{self.name}': Спроба отримати відстань з неініціалізованого УЗД.")
            return None  # Або float('inf') залежно від того, як обробляється помилка
        try:
            distance = self._sensor.distance  #
            return float(distance)  #
        except Exception as e:  # gpiozero може кинути виняток, якщо не може прочитати (напр. Timeout)
            self._logger.warning(f"'{self.name}': Не вдалося прочитати відстань: {e}")
            return float('inf')  # Повертаємо float('inf') у разі помилки читання

    def is_vehicle_approaching(self, threshold_m=None):
        """
        Перевіряє, чи під'їхав автомобіль (відстань менша за поріг).
        Args:
            threshold_m (float, optional): Порогова відстань. Використовує DEFAULT_THRESHOLD_VEHICLE_APPROACH якщо None.
        Returns:
            bool: True, якщо автомобіль під'їхав.
                  None, якщо датчик не ініціалізований.
        """
        if self._sensor is None: return None  #

        current_threshold = threshold_m if threshold_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_APPROACH  #
        dist = self.get_distance()
        if dist is None or dist == float('inf'):  # Помилка датчика або об'єкт занадто далеко
            return False  #
        return dist < current_threshold  #

    def has_vehicle_passed(self,
                           threshold_clear_m=None,
                           confirmation_time_s=None):
        """
        Перевіряє, чи автомобіль проїхав зону воріт.
        Логіка: відстань стала більшою за поріг `threshold_clear_m`
        протягом `confirmation_time_s` секунд.
        Returns:
            bool: True, якщо автомобіль проїхав.
                  None, якщо датчик не ініціалізований.
        """
        if self._sensor is None: return None  #

        current_threshold_clear = threshold_clear_m if threshold_clear_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR  #
        current_confirmation_time = confirmation_time_s if confirmation_time_s is not None else self.PASS_CONFIRMATION_TIME_S  #

        dist = self.get_distance()
        if dist is None or dist == float('inf'):  # Помилка датчика або об'єкт занадто далеко
            self.last_clear_time_monotonic = None  # Скидаємо таймер
            return False  # Не можна підтвердити проїзд, якщо не бачимо чистої зони

        if dist > current_threshold_clear:  #
            if self.last_clear_time_monotonic is None:  #
                self.last_clear_time_monotonic = time.monotonic()  #
                self._logger.debug(
                    f"'{self.name}': Виявлено чисту зону (відстань: {dist:.2f}м > {current_threshold_clear:.2f}м). Запуск таймера підтвердження.")
                return False  # Ще не підтверджено
            elif time.monotonic() - self.last_clear_time_monotonic >= current_confirmation_time:  #
                self._logger.debug(
                    f"'{self.name}': Проїзд автомобіля підтверджено: чиста зона утримувалася {time.monotonic() - self.last_clear_time_monotonic:.2f}с.")
                self.last_clear_time_monotonic = None  # Скидаємо для наступного разу
                return True  #
            else:
                return False  #
        else:  # Об'єкт все ще близько або з'явився знову, скидаємо таймер
            if self.last_clear_time_monotonic is not None:
                self._logger.debug(
                    f"'{self.name}': Виявлено об'єкт у зоні (відстань: {dist:.2f}м <= {current_threshold_clear:.2f}м). Скидання таймера підтвердження.")
            self.last_clear_time_monotonic = None  #
            return False  #

    def wait_for_approach(self, threshold_m=None, timeout=None):
        """Чекає, поки об'єкт не наблизиться на відстань threshold_m."""
        if self._sensor:
            current_threshold = threshold_m if threshold_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_APPROACH  #
            self._logger.debug(
                f"'{self.name}': Очікування наближення авто (< {current_threshold:.2f}м, таймаут: {timeout}с)...")
            start_time = time.monotonic()  #
            while True:
                if self.is_vehicle_approaching(current_threshold):  #
                    self._logger.debug(f"'{self.name}': Автомобіль наблизився.")
                    return True  #
                if timeout is not None and (time.monotonic() - start_time) > timeout:  #
                    self._logger.debug(f"'{self.name}': Таймаут очікування наближення авто.")
                    return False  #
                time.sleep(0.1)  # Невелика затримка, щоб не навантажувати CPU
        else:
            self._logger.warning(f"'{self.name}': Неможливо очікувати наближення, датчик не ініціалізовано.")
            return False

    def wait_for_clear_after_pass(self, threshold_m=None, confirmation_s=None, timeout=None):
        """Чекає, поки об'єкт не проїде (зона стане чистою з підтвердженням)."""
        if self._sensor:
            current_threshold_clear = threshold_m if threshold_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR  #
            current_confirmation_time = confirmation_s if confirmation_s is not None else self.PASS_CONFIRMATION_TIME_S  #
            self._logger.debug(
                f"'{self.name}': Очікування проїзду авто та очищення зони (> {current_threshold_clear:.2f}м протягом {current_confirmation_time:.1f}с, таймаут: {timeout}с)...")

            start_time_overall = time.monotonic()  #
            while True:
                if self.has_vehicle_passed(current_threshold_clear, current_confirmation_time):  #
                    self._logger.debug(f"'{self.name}': Автомобіль проїхав, зона чиста.")
                    return True  #
                if timeout is not None and (time.monotonic() - start_time_overall) > timeout:  #
                    self._logger.debug(f"'{self.name}': Таймаут очікування проїзду авто та очищення зони.")
                    return False  #
                time.sleep(0.1)  # Невелика затримка
        else:
            self._logger.warning(f"'{self.name}': Неможливо очікувати проїзд, датчик не ініціалізовано.")
            return False

    def cleanup(self):
        """Звільняє ресурси датчика."""
        if self._sensor:
            try:
                self._sensor.close()
                self._logger.info(f"Ресурси УЗД '{self.name}' (T:{self.trigger_pin},E:{self.echo_pin}) звільнено.")
            except Exception as e:
                self._logger.error(f"Помилка під час звільнення ресурсів УЗД '{self.name}': {e}", exc_info=True)
            self._sensor = None


# --- Клас Менеджера Датчиків ---
class SensorManager:
    """
    Клас для ініціалізації та керування всіма датчиками системи.
    """

    def __init__(self,
                 reed_pin: int,
                 ultrasonic_entry_trigger_pin: int,
                 ultrasonic_entry_echo_pin: int,
                 ultrasonic_exit_trigger_pin: int = None,  # Опціонально для виїзду
                 ultrasonic_exit_echo_pin: int = None,  # Опціонально для виїзду
                 reed_name: str = "GateReedSensor",
                 ultrasonic_entry_name: str = "EntryUltrasonicSensor",
                 ultrasonic_exit_name: str = "ExitUltrasonicSensor"):
        """
        Ініціалізує всі необхідні датчики.

        Args:
            reed_pin (int): GPIO пін для геркона.
            ultrasonic_entry_trigger_pin (int): Trigger пін для УЗД в'їзду.
            ultrasonic_entry_echo_pin (int): Echo пін для УЗД в'їзду.
            ultrasonic_exit_trigger_pin (int, optional): Trigger пін для УЗД виїзду.
            ultrasonic_exit_echo_pin (int, optional): Echo пін для УЗД виїзду.
            reed_name (str): Ім'я для геркона.
            ultrasonic_entry_name (str): Ім'я для УЗД в'їзду.
            ultrasonic_exit_name (str): Ім'я для УЗД виїзду.
        """
        self._logger = logging.getLogger(f"{__name__}.SensorManager")
        self._logger.info("Ініціалізація SensorManager...")

        self.reed_switch = None
        self.ultrasonic_sensor_entry = None
        self.ultrasonic_sensor_exit = None

        try:
            self.reed_switch = ReedSwitch(pin_number=reed_pin, name=reed_name)  #
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати {reed_name}: {e}", exc_info=True)
            # Можна прокинути помилку далі або продовжити без цього датчика

        try:
            self.ultrasonic_sensor_entry = UltrasonicSensor(
                trigger_pin=ultrasonic_entry_trigger_pin,
                echo_pin=ultrasonic_entry_echo_pin,
                name=ultrasonic_entry_name
            )  #
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
        """Звільняє ресурси всіх керованих датчиків."""
        self._logger.info("Очищення ресурсів SensorManager...")
        if self.reed_switch:
            self.reed_switch.cleanup()
        if self.ultrasonic_sensor_entry:
            self.ultrasonic_sensor_entry.cleanup()
        if self.ultrasonic_sensor_exit:
            self.ultrasonic_sensor_exit.cleanup()
        self._logger.info("Очищення ресурсів SensorManager завершено.")

    def __del__(self):
        self.cleanup()


# --- Приклад використання (для тестування модуля окремо) ---
if __name__ == '__main__':
    # Налаштування базового логування для тестування
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s')

    logger.info("Тестування модуля sensors_manager.py...")

    # Параметри пінів для тесту (змініть на ваші реальні піни для тестування на Raspberry Pi)
    TEST_REED_PIN = 22
    TEST_US_ENTRY_TRIG = 23
    TEST_US_ENTRY_ECHO = 24
    # TEST_US_EXIT_TRIG = 25
    # TEST_US_EXIT_ECHO = 26

    try:
        manager = SensorManager(
            reed_pin=TEST_REED_PIN,
            ultrasonic_entry_trigger_pin=TEST_US_ENTRY_TRIG,
            ultrasonic_entry_echo_pin=TEST_US_ENTRY_ECHO
            # ultrasonic_exit_trigger_pin=TEST_US_EXIT_TRIG,
            # ultrasonic_exit_echo_pin=TEST_US_EXIT_ECHO
        )

        if manager.reed_switch and manager.reed_switch._device:  # Перевірка, чи геркон ініціалізовано
            logger.info("\nТестування геркона через менеджер:")
            logger.info(f"  Ворота відкриті? {manager.reed_switch.are_gates_open}")  #
            logger.info(f"  Ворота закриті? {manager.reed_switch.are_gates_closed}")  #
            # Імітація зміни стану геркона (якщо це мок або реальний датчик)

        if manager.ultrasonic_sensor_entry and manager.ultrasonic_sensor_entry._sensor:  # Перевірка УЗД
            logger.info("\nТестування УЗД в'їзду через менеджер:")
            distance = manager.ultrasonic_sensor_entry.get_distance()  #
            logger.info(f"  Відстань (в'їзд): {distance:.2f} м")
            logger.info(
                f"  Авто наближається (в'їзд, поріг 1м)? {manager.ultrasonic_sensor_entry.is_vehicle_approaching(1.0)}")  #
            logger.info(
                f"  Авто проїхало (в'їзд, поріг зони 2м, підтвердження 1с)? {manager.ultrasonic_sensor_entry.has_vehicle_passed(2.0, 1.0)}")  #

        if manager.ultrasonic_sensor_exit and manager.ultrasonic_sensor_exit._sensor:
            logger.info("\nТестування УЗД виїзду через менеджер:")
            distance_exit = manager.ultrasonic_sensor_exit.get_distance()
            logger.info(f"  Відстань (виїзд): {distance_exit:.2f} м")

        # Важливо викликати cleanup для звільнення ресурсів gpiozero
        manager.cleanup()

    except Exception as e:
        logger.error(f"Помилка під час тестування SensorManager: {e}", exc_info=True)

    logger.info("Тестування модуля sensors_manager.py завершено.")