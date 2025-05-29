# ultrasonic_test.py
import time
import logging
import os
import sys

# Додаємо шлях до кореневої директорії проекту для імпорту core.sensors_manager
project_root = os.path.abspath(os.path.dirname(__file__))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    # Припускаємо, що sensors_manager.py знаходиться в директорії core/
    from core.sensors_manager import UltrasonicSensor, PiGPIOFactory
    # Спробуємо налаштувати PiGPIOFactory для UltrasonicSensor, як це робиться в sensors_manager
    from gpiozero import Device

    try:
        Device.pin_factory = PiGPIOFactory()
        FACTORY_USED_MSG = "Використовується PiGPIOFactory."
    except Exception as e_factory:
        Device.pin_factory = None  # gpiozero обере фабрику за замовчуванням
        FACTORY_USED_MSG = f"PiGPIOFactory не вдалося ініціалізувати ({e_factory}), використовується фабрика за замовчуванням."
except ImportError as e:
    print(f"Помилка імпорту: {e}")
    print("Переконайтеся, що файл core/sensors_manager.py існує та доступний.")


    # Створюємо мок-клас, якщо імпорт не вдався, для тестування логіки на ПК
    class UltrasonicSensor:
        DEFAULT_PASSAGE_THRESHOLD_M = 0.3
        DEFAULT_APPROACH_THRESHOLD_M = 1.0

        def __init__(self, trigger_pin, echo_pin, name="MockUltrasonic"):
            self.name = name
            self.trigger_pin = trigger_pin
            self.echo_pin = echo_pin
            self._logger = logging.getLogger(f"MockUltrasonic.{name}")
            self.mock_distance = 2.0  # Початкова відстань
            self._logger.info(f"Мок УЗД '{name}' створено (Trig:{trigger_pin}, Echo:{echo_pin}).")
            self._logger.info(
                FACTORY_USED_MSG if 'FACTORY_USED_MSG' in globals() else "Фабрика gpiozero не ініціалізована.")

        def get_distance(self):
            self._logger.debug(f"'{self.name}' (Мок) повертає відстань: {self.mock_distance:.2f} м")
            return self.mock_distance

        def detect_object_in_passage(self, passage_threshold_m=None):
            threshold = passage_threshold_m if passage_threshold_m is not None else self.DEFAULT_PASSAGE_THRESHOLD_M
            return self.mock_distance < threshold

        def wait_for_object_to_enter_passage(self, passage_threshold_m=None, timeout_s=None):
            self._logger.info(f"'{self.name}' (Мок): імітація очікування входу об'єкта...")
            time.sleep(0.5)  # Імітація
            return self.detect_object_in_passage(passage_threshold_m)

        def wait_for_object_to_clear_passage(self, passage_threshold_m=None, timeout_s=None):
            self._logger.info(f"'{self.name}' (Мок): імітація очікування виходу об'єкта...")
            time.sleep(0.5)  # Імітація
            return not self.detect_object_in_passage(passage_threshold_m)

        def is_vehicle_approaching(self, threshold_m=None):  # Для сумісності з попередніми тестами
            threshold = threshold_m if threshold_m is not None else self.DEFAULT_APPROACH_THRESHOLD_M
            return self.mock_distance < threshold

        def cleanup(self):
            self._logger.info(f"'{self.name}' (Мок): cleanup викликано.")

# Налаштування логування
logging.basicConfig(
    level=logging.DEBUG,  # DEBUG для детального виводу
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("UltrasonicTest")

# --- Параметри Тесту ---
# Вкажіть GPIO піни (BCM нумерація), до яких підключений ваш УЗД
# Це мають бути ті самі піни, що й для ultrasonic_sensor_entry у вашому SensorManager
TRIGGER_PIN = 23
ECHO_PIN = 24

# Поріг для визначення "об'єкт у проїзді" (менше цього значення - об'єкт є)
PASSAGE_THRESHOLD = 0.3  # метри
# Поріг для "наближення" (для is_vehicle_approaching, якщо будете тестувати)
APPROACH_THRESHOLD = 1.0  # метри

# Таймаути для очікування
ENTER_TIMEOUT = 5  # секунд на очікування входу об'єкта
CLEAR_TIMEOUT = 5  # секунд на очікування виходу об'єкта


def run_test_loop(sensor: UltrasonicSensor):
    logger.info(f"\n--- Початок безперервного тестування датчика '{sensor.name}' ---")
    logger.info("Піднесіть та віддаліть об'єкт від датчика. Натисніть Ctrl+C для зупинки.")

    last_state_in_passage = False
    try:
        while True:
            distance = sensor.get_distance()
            if distance is not None:
                logger.info(f"Поточна відстань: {distance:.3f} м")

                is_in_passage = sensor.detect_object_in_passage(PASSAGE_THRESHOLD)
                if is_in_passage != last_state_in_passage:
                    if is_in_passage:
                        logger.info(f"!!! ОБ'ЄКТ У ПРОЇЗДІ (< {PASSAGE_THRESHOLD} м) !!!")
                    else:
                        logger.info(f"--- Проїзд вільний (> {PASSAGE_THRESHOLD} м) ---")
                    last_state_in_passage = is_in_passage
            else:
                logger.warning("Не вдалося отримати відстань з датчика.")

            time.sleep(0.5)  # Пауза між зчитуваннями

    except KeyboardInterrupt:
        logger.info("Тестування зупинено користувачем.")
    except Exception as e:
        logger.error(f"Помилка під час тестування: {e}", exc_info=True)


def main():
    logger.info(f"Ініціалізація УЗД на Trigger: GPIO{TRIGGER_PIN}, Echo: GPIO{ECHO_PIN}")

    # Створюємо екземпляр UltrasonicSensor
    # Якщо sensors_manager.py знаходиться в core/, а цей скрипт в корені:
    # from core.sensors_manager import UltrasonicSensor (вже зроблено на початку)

    ultrasonic_sensor = UltrasonicSensor(trigger_pin=TRIGGER_PIN, echo_pin=ECHO_PIN, name="TestUS")

    # Перевірка, чи датчик ініціалізувався (для реального gpiozero)
    if hasattr(ultrasonic_sensor,
               '_sensor') and ultrasonic_sensor._sensor is None and 'MockUltrasonic' not in ultrasonic_sensor.name:
        logger.error("Не вдалося ініціалізувати реальний УЗД. Перевірте підключення та налаштування pigpio.")
        return

    # --- Тест 1: detect_object_in_passage ---
    logger.info("\n--- Тест 1: detect_object_in_passage ---")
    logger.info(f"Будь ласка, розмістіть об'єкт ближче {PASSAGE_THRESHOLD}м, потім далі.")
    for _ in range(5):  # 5 зчитувань
        dist_current = ultrasonic_sensor.get_distance()  # Для мока може знадобитися ручне встановлення
        if dist_current is not None:
            logger.info(
                f"  Поточна відстань: {dist_current:.3f}м. Об'єкт у проїзді? -> {ultrasonic_sensor.detect_object_in_passage(PASSAGE_THRESHOLD)}")
        else:
            logger.warning("  Не вдалося отримати відстань.")
        time.sleep(1)

    # --- Тест 2: wait_for_object_to_enter_passage ---
    logger.info(f"\n--- Тест 2: wait_for_object_to_enter_passage (таймаут {ENTER_TIMEOUT}с) ---")
    logger.info(f"Приберіть об'єкт, потім піднесіть його ближче {PASSAGE_THRESHOLD}м протягом {ENTER_TIMEOUT}с.")
    # Для мока: ultrasonic_sensor.mock_distance = 2.0; потім змінити на 0.1
    if ultrasonic_sensor.wait_for_object_to_enter_passage(PASSAGE_THRESHOLD, ENTER_TIMEOUT):
        logger.info("  УСПІХ: Об'єкт увійшов у проїзд.")
    else:
        logger.warning("  ПОМИЛКА/ТАЙМАУТ: Об'єкт не увійшов у проїзд.")

    # --- Тест 3: wait_for_object_to_clear_passage ---
    logger.info(f"\n--- Тест 3: wait_for_object_to_clear_passage (таймаут {CLEAR_TIMEOUT}с) ---")
    logger.info(f"Залиште об'єкт близько (<{PASSAGE_THRESHOLD}м), потім приберіть його протягом {CLEAR_TIMEOUT}с.")
    # Для мока: ultrasonic_sensor.mock_distance = 0.1; потім змінити на 2.0
    if ultrasonic_sensor.wait_for_object_to_clear_passage(PASSAGE_THRESHOLD, CLEAR_TIMEOUT):
        logger.info("  УСПІХ: Об'єкт покинув проїзд.")
    else:
        logger.warning("  ПОМИЛКА/ТАЙМАУТ: Об'єкт не покинув проїзд (або був відсутній спочатку).")

    # --- Тест 4: Безперервне зчитування (зупинка по Ctrl+C) ---
    run_test_loop(ultrasonic_sensor)

    if hasattr(ultrasonic_sensor, 'cleanup'):
        ultrasonic_sensor.cleanup()
    logger.info("Тестування УЗД завершено.")


if __name__ == "__main__":
    main()