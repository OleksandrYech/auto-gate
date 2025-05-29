# sensors.py
from gpiozero import DigitalInputDevice, DistanceSensor
from gpiozero.pins.pigpio import PiGPIOFactory # Для більш стабільної роботи, особливо з DistanceSensor
import time
import logging

# Налаштування логування
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__) # Створюємо логгер для цього модуля

# Використання PiGPIOFactory може покращити стабільність, особливо для ШІМ/точних вимірювань
# Потрібно встановити: sudo apt install pigpiod python3-pigpio
# та запустити демон: sudo systemctl start pigpiod; sudo systemctl enable pigpiod
# Якщо ви не хочете використовувати pigpio, можна закоментувати наступні рядки
# і gpiozero буде використовувати RPi.GPIO або інший доступний pin factory.
# Однак для DistanceSensor це часто рекомендовано.
try:
    factory = PiGPIOFactory()
    logger.info("Using PiGPIOFactory for pin control.")
except (OSError, NameError): # NameError якщо PiGPIOFactory не імпортовано/встановлено
    logger.warning("pigpio daemon not found, not running, or python3-pigpio not installed. "
                   "Falling back to default pin factory. "
                   "For DistanceSensor, pigpio is recommended for better stability.")
    factory = None # gpiozero обере фабрику за замовчуванням

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
        self._logger = logging.getLogger(f"{__name__}.{self.name}") # Специфічний логгер для екземпляра
        try:
            # pull_up=True: Коли геркон розімкнений (магніт далеко, ворота закриті), пін підтягнутий до HIGH.
            # Коли геркон замкнений (магніт близько, ворота відкриті), він замикає пін на GND, і пін стає LOW.
            # Отже, is_active (LOW) буде означати "ворота відкриті".
            self._device = DigitalInputDevice(pin_number, pull_up=True, pin_factory=factory)
            self._logger.info(f"Initialized on GPIO{self.pin_number}.")
            # Початкове зчитування стану
            if self._device.is_active: # is_active == True, якщо пін LOW (геркон замкнений)
                 self._logger.info(f"Initial state: GATES OPEN (contact closed, pin is LOW)")
            else: # is_active == False, якщо пін HIGH (геркон розімкнений)
                 self._logger.info(f"Initial state: GATES CLOSED (contact open, pin is HIGH)")

        except Exception as e:
            self._logger.error(f"Failed to initialize on GPIO{self.pin_number}: {e}")
            self._device = None

    @property
    def are_gates_open(self):
        """
        Перевіряє, чи ворота відкриті.

        Returns:
            bool: True, якщо ворота відкриті (геркон замкнений), False в іншому випадку.
                  None, якщо датчик не ініціалізований.
        """
        if self._device is None:
            return None
        # Якщо pull_up=True, то:
        # Геркон замкнений (ворота відкриті) -> пін LOW -> self._device.value == 0 -> self._device.is_active == True
        # Геркон розімкнений (ворота закриті) -> пін HIGH -> self._device.value == 1 -> self._device.is_active == False
        return self._device.is_active

    @property
    def are_gates_closed(self):
        """
        Перевіряє, чи ворота закриті.

        Returns:
            bool: True, якщо ворота закриті (геркон розімкнений), False в іншому випадку.
                  None, якщо датчик не ініціалізований.
        """
        if self._device is None:
            return None
        return not self._device.is_active # Протилежно до are_gates_open

    def wait_for_open(self, timeout=None):
        """Чекає, поки ворота відкриються (геркон замкнеться)."""
        if self._device:
            self._logger.debug(f"Waiting for gates to open (timeout: {timeout}s)...")
            self._device.wait_for_active(timeout) # is_active (LOW) -> ворота відкриті
            self._logger.debug(f"Gates detected as open.")

    def wait_for_close(self, timeout=None):
        """Чекає, поки ворота закриються (геркон розімкнеться)."""
        if self._device:
            self._logger.debug(f"Waiting for gates to close (timeout: {timeout}s)...")
            self._device.wait_for_inactive(timeout) # is_inactive (HIGH) -> ворота закриті
            self._logger.debug(f"Gates detected as closed.")

# --- Клас для Ультразвукового датчика ---
class UltrasonicSensor:
    """
    Клас для роботи з ультразвуковим датчиком відстані AJ-SPO4M (або сумісним HC-SR04).
    """
    # Відстані в метрах
    DEFAULT_THRESHOLD_VEHICLE_APPROACH = 1.0  # 1 метр для фіксації під'їзду
    DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR = 2.0 # 2 метри, щоб вважати, що авто проїхало і зона вільна
    PASS_CONFIRMATION_TIME_S = 1.5 # Секунди, протягом яких зона має бути вільною, щоб підтвердити проїзд

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
        self._logger = logging.getLogger(f"{__name__}.{self.name}") # Специфічний логгер для екземпляра
        self.last_clear_time_monotonic = None
        try:
            self._sensor = DistanceSensor(
                echo=echo_pin,
                trigger=trigger_pin,
                max_distance=4,  # Макс. вимірювана відстань - 4 метри
                # threshold_distance не використовується активно класом, ми робимо свої перевірки
                pin_factory=factory,
                queue_len=3 # Усереднення по 3 вимірам для стабільності
            )
            self._logger.info(f"Initialized on Trigger:GPIO{trigger_pin}, Echo:GPIO{echo_pin}.")
            # Дати датчику трохи часу на стабілізацію
            time.sleep(0.5)
            self._logger.info(f"Initial distance: {self.get_distance():.2f} m")

        except Exception as e:
            self._logger.error(f"Failed to initialize: {e}")
            self._sensor = None

    def get_distance(self):
        """
        Повертає поточну виміряну відстань в метрах.

        Returns:
            float: Відстань в метрах, або float('inf') у разі помилки/відсутності об'єкта в межах max_distance.
                   None, якщо датчик не ініціалізований.
        """
        if self._sensor is None:
            return None
        try:
            distance = self._sensor.distance
            return float(distance) # Переконуємось, що це float
        except Exception as e:
            self._logger.warning(f"Could not read distance: {e}")
            return float('inf')

    def is_vehicle_approaching(self, threshold_m=None):
        """
        Перевіряє, чи під'їхав автомобіль (відстань менша за поріг).

        Args:
            threshold_m (float, optional): Порогова відстань в метрах.
                                         Якщо None, використовується DEFAULT_THRESHOLD_VEHICLE_APPROACH.
        Returns:
            bool: True, якщо автомобіль під'їхав, False в іншому випадку.
                  None, якщо датчик не ініціалізований.
        """
        if self._sensor is None:
            return None
        
        current_threshold = threshold_m if threshold_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_APPROACH
        dist = self.get_distance()
        if dist is None or dist == float('inf'): # Помилка датчика або об'єкт занадто далеко
            return False
        return dist < current_threshold

    def has_vehicle_passed(self,
                           threshold_clear_m=None,
                           confirmation_time_s=None):
        """
        Перевіряє, чи автомобіль проїхав зону воріт.
        Логіка: відстань стала більшою за поріг `threshold_clear_m`
        протягом `confirmation_time_s` секунд.

        Args:
            threshold_clear_m (float, optional): Поріг "чистої зони" в метрах.
                                               Якщо None, використовується DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR.
            confirmation_time_s (float, optional): Час в секундах, протягом якого зона має бути чистою.
                                                 Якщо None, використовується PASS_CONFIRMATION_TIME_S.
        Returns:
            bool: True, якщо автомобіль проїхав, False в іншому випадку.
                  None, якщо датчик не ініціалізований.
        """
        if self._sensor is None:
            return None

        current_threshold_clear = threshold_clear_m if threshold_clear_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR
        current_confirmation_time = confirmation_time_s if confirmation_time_s is not None else self.PASS_CONFIRMATION_TIME_S

        dist = self.get_distance()
        if dist is None or dist == float('inf'): # Помилка датчика або об'єкт занадто далеко
            self.last_clear_time_monotonic = None # Скидаємо таймер
            return False # Не можна підтвердити проїзд, якщо не бачимо чистої зони

        if dist > current_threshold_clear:
            # Зона чиста, починаємо або продовжуємо відлік
            if self.last_clear_time_monotonic is None:
                self.last_clear_time_monotonic = time.monotonic()
                self._logger.debug(f"Clear zone detected (dist: {dist:.2f}m > {current_threshold_clear:.2f}m). Starting confirmation timer.")
                return False # Ще не підтверджено
            elif time.monotonic() - self.last_clear_time_monotonic >= current_confirmation_time:
                # Зона була чистою достатньо довго
                self._logger.debug(f"Vehicle passed confirmation: clear zone maintained for {time.monotonic() - self.last_clear_time_monotonic:.2f}s.")
                self.last_clear_time_monotonic = None # Скидаємо для наступного разу
                return True
            else:
                # Зона чиста, але час підтвердження ще не вийшов
                return False
        else:
            # Об'єкт все ще близько або з'явився знову, скидаємо таймер
            if self.last_clear_time_monotonic is not None:
                 self._logger.debug(f"Object detected in zone (dist: {dist:.2f}m <= {current_threshold_clear:.2f}m). Resetting confirmation timer.")
            self.last_clear_time_monotonic = None
            return False

    def wait_for_approach(self, threshold_m=None, timeout=None):
        """Чекає, поки об'єкт не наблизиться на відстань threshold_m."""
        if self._sensor:
            current_threshold = threshold_m if threshold_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_APPROACH
            self._logger.debug(f"Waiting for vehicle approach (< {current_threshold:.2f}m, timeout: {timeout}s)...")
            # Використовуємо цикл замість вбудованого wait_for_in_range для більшого контролю
            start_time = time.monotonic()
            while True:
                if self.is_vehicle_approaching(current_threshold):
                    self._logger.debug(f"Vehicle approached.")
                    return True
                if timeout is not None and (time.monotonic() - start_time) > timeout:
                    self._logger.debug(f"Timeout waiting for vehicle approach.")
                    return False
                time.sleep(0.1) # Невелика затримка, щоб не навантажувати CPU

    def wait_for_clear_after_pass(self, threshold_m=None, confirmation_s=None, timeout=None):
        """Чекає, поки об'єкт не проїде (зона стане чистою з підтвердженням)."""
        if self._sensor:
            current_threshold_clear = threshold_m if threshold_m is not None else self.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR
            current_confirmation_time = confirmation_s if confirmation_s is not None else self.PASS_CONFIRMATION_TIME_S
            self._logger.debug(f"Waiting for vehicle to pass and clear zone (> {current_threshold_clear:.2f}m for {current_confirmation_time:.1f}s, timeout: {timeout}s)...")
            
            start_time_overall = time.monotonic()
            while True:
                if self.has_vehicle_passed(current_threshold_clear, current_confirmation_time):
                    self._logger.debug(f"Vehicle has passed and zone is clear.")
                    return True
                if timeout is not None and (time.monotonic() - start_time_overall) > timeout:
                    self._logger.debug(f"Timeout waiting for vehicle to pass and clear zone.")
                    return False
                time.sleep(0.1) # Невелика затримка

# --- Приклад використання (для тестування модуля окремо) ---
if __name__ == '__main__':
    logger.info("Тестування модуля sensors.py з оновленими пінами...")

    # --- Тест Геркона ---
    REED_PIN = 22  # Оновлений GPIO пін для геркона
    reed_sensor = ReedSwitch(pin_number=REED_PIN, name=f"GateReedSwitch_GPIO{REED_PIN}")

    if reed_sensor._device:
        print(f"\nТестування геркона на GPIO{REED_PIN}:")
        print("Піднесіть/віддаліть магніт від геркона протягом наступних 10 секунд.")
        for i in range(20):
            state_msg = "ВОРОТА ВІДКРИТІ (контакт замкнений)" if reed_sensor.are_gates_open else "ВОРОТА ЗАКРИТІ (контакт розімкнений)"
            pin_state = "LOW" if reed_sensor.are_gates_open else "HIGH"
            print(f"Стан геркона: {state_msg}, GPIO{REED_PIN} is {pin_state}")
            time.sleep(0.5)
        print("Тест геркона завершено.\n")
    else:
        print(f"Геркон на GPIO{REED_PIN} не ініціалізовано. Тест пропущено.")

    # --- Тест Ультразвукового датчика ---
    TRIGGER_PIN = 23  # Оновлений GPIO пін для Trigger
    ECHO_PIN = 24     # Оновлений GPIO пін для Echo
    ultrasonic_sensor = UltrasonicSensor(trigger_pin=TRIGGER_PIN, echo_pin=ECHO_PIN, name=f"ApproachUltrasonic_T{TRIGGER_PIN}_E{ECHO_PIN}")

    if ultrasonic_sensor._sensor:
        print(f"Тестування ультразвукового датчика (Trigger: GPIO{TRIGGER_PIN}, Echo: GPIO{ECHO_PIN}):")
        print(f"  ВАЖЛИВО: Переконайтеся, що на Echo піні (GPIO{ECHO_PIN}) є дільник напруги (5V -> 3.3V)!")
        print(f"  Поріг під'їзду: {ultrasonic_sensor.DEFAULT_THRESHOLD_VEHICLE_APPROACH:.2f} м")
        print(f"  Поріг проїзду (зона чиста): {ultrasonic_sensor.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR:.2f} м")
        print(f"  Час підтвердження проїзду: {ultrasonic_sensor.PASS_CONFIRMATION_TIME_S:.1f} с")
        
        print("\nВимірювання відстані та стану протягом 20 секунд:")
        start_time_ultrasonic_test = time.monotonic()
        
        last_approaching_state = False
        last_passed_state = False

        while time.monotonic() - start_time_ultrasonic_test < 20: # Тест протягом 20 секунд
            distance = ultrasonic_sensor.get_distance()
            
            current_approaching_state = ultrasonic_sensor.is_vehicle_approaching()
            current_passed_state = ultrasonic_sensor.has_vehicle_passed()

            log_msg = f"  Відстань: {distance:.2f} м"
            
            if current_approaching_state:
                log_msg += " | ПІД'ЇЗД!"
            if current_passed_state:
                log_msg += " | ПРОЇЗД!"
            
            print(log_msg)

            if current_approaching_state != last_approaching_state:
                if current_approaching_state:
                    logger.info(f"Подія: Автомобіль ПІД'ЇХАВ (відстань < {ultrasonic_sensor.DEFAULT_THRESHOLD_VEHICLE_APPROACH:.2f} м)")
                last_approaching_state = current_approaching_state
            
            if current_passed_state != last_passed_state:
                if current_passed_state:
                     logger.info(f"Подія: Автомобіль ПРОЇХАВ (зона чиста > {ultrasonic_sensor.DEFAULT_THRESHOLD_VEHICLE_PASSED_CLEAR:.2f} м протягом {ultrasonic_sensor.PASS_CONFIRMATION_TIME_S:.1f} с)")
                last_passed_state = current_passed_state
            
            time.sleep(0.5)
        print("Тест ультразвукового датчика завершено.\n")
    else:
        print(f"Ультразвуковий датчик (Trig:GPIO{TRIGGER_PIN}, Echo:GPIO{ECHO_PIN}) не ініціалізовано. Тест пропущено.")

    logger.info("Тестування модуля sensors.py завершено.")