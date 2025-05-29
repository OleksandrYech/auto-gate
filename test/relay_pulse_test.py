# relay_pulse_test.py
import time
import logging

# Намагаємося імпортувати gpiozero та налаштувати PiGPIOFactory для кращої стабільності
try:
    from gpiozero import OutputDevice
    from gpiozero.pins.pigpio import PiGPIOFactory

    try:
        # Використовуємо PiGPIOFactory, якщо доступно
        OutputDevice.pin_factory = PiGPIOFactory()
        FACTORY_USED = "PiGPIOFactory"
    except Exception:  # Broad exception for issues like pigpio daemon not running
        # Якщо PiGPIOFactory недоступна, gpiozero використає фабрику за замовчуванням (RPi.GPIO або іншу)
        OutputDevice.pin_factory = None  # Явно скидаємо, щоб gpiozero обрала наступну доступну
        FACTORY_USED = "Default (e.g., RPi.GPIO)"
    GPIOZERO_AVAILABLE = True
except ImportError:
    GPIOZERO_AVAILABLE = False


    # Створюємо мок-клас, якщо gpiozero недоступний (для тестування логіки на ПК)
    class OutputDevice:
        def __init__(self, pin, active_high=True, initial_value=False):
            self.pin = pin
            self._active_high = active_high
            self._value = initial_value
            self._is_active = False  # is_active показує, чи пристрій "увімкнено"
            self.logger = logging.getLogger(f"MockOutputDevice.Pin{pin}")
            self.logger.info(f"Мок OutputDevice створено для піна {pin}")

        def on(self):
            self._is_active = True
            self.logger.info(f"Мок OutputDevice pin {self.pin} -> УВІМКНЕНО (стан LOW для active_high=False)")

        def off(self):
            self._is_active = False
            self.logger.info(f"Мок OutputDevice pin {self.pin} -> ВИМКНЕНО (стан HIGH для active_high=False)")

        @property
        def is_active(self):
            return self._is_active

        def close(self):
            self.logger.info(f"Мок OutputDevice pin {self.pin} закрито.")

# Налаштування логування
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s',
    handlers=[logging.StreamHandler()]
)
logger = logging.getLogger("RelayPulseTest")

# --- Параметри Тесту ---
OPEN_PIN = 17  # GPIO пін для реле OPEN
CLOSE_PIN = 27  # GPIO пін для реле CLOSE

PULSE_DURATION = 0.5  # Тривалість імпульсу в секундах
NUM_PULSES = 3  # Кількість імпульсів для кожного реле
PAUSE_BETWEEN_PULSES = 1.0  # Пауза між імпульсами в секундах
PAUSE_BETWEEN_RELAYS = 2.0  # Пауза між тестуванням OPEN та CLOSE реле


def send_pulse(relay_device: OutputDevice, relay_name: str, duration: float):
    """Надсилає один імпульс на вказане реле."""
    pin_number_str = str(getattr(relay_device, 'pin', 'N/A'))
    if GPIOZERO_AVAILABLE and hasattr(relay_device.pin, 'number'):  # Для реального gpiozero
        pin_number_str = str(relay_device.pin.number)

    logger.info(f"Подача імпульсу на реле '{relay_name}' (GPIO{pin_number_str}) на {duration}с...")
    try:
        relay_device.on()  # Активувати реле (пін LOW, оскільки active_high=False)
        time.sleep(duration)
    except Exception as e:
        logger.error(f"Помилка під час активації або очікування реле '{relay_name}': {e}", exc_info=True)
    finally:
        try:
            if relay_device.is_active:
                relay_device.off()  # Деактивувати реле (пін HIGH)
                logger.info(f"Реле '{relay_name}' (GPIO{pin_number_str}) деактивовано.")
            else:
                # Це може статися, якщо .on() не спрацював або був перерваний
                logger.warning(f"Реле '{relay_name}' не було активним перед спробою вимкнення в finally.")
        except Exception as e_off:
            logger.error(f"Помилка під час деактивації реле '{relay_name}': {e_off}", exc_info=True)


def main():
    if not GPIOZERO_AVAILABLE:
        logger.warning("Бібліотека gpiozero недоступна. Тест буде виконуватися з мок-об'єктами.")
    else:
        logger.info(f"Використовується фабрика пінів: {FACTORY_USED}")

    open_relay = None
    close_relay = None

    try:
        # Ініціалізація реле (low-level: active_high=False, initial_value=True -> початково вимкнене)
        logger.info(f"Ініціалізація реле OPEN на GPIO {OPEN_PIN}")
        open_relay = OutputDevice(OPEN_PIN, active_high=False, initial_value=True)

        logger.info(f"Ініціалізація реле CLOSE на GPIO {CLOSE_PIN}")
        close_relay = OutputDevice(CLOSE_PIN, active_high=False, initial_value=True)

        logger.info("Реле ініціалізовано. Початок тестування імпульсів.")

        # Тестування реле OPEN
        logger.info(f"\n--- Тестування реле OPEN (GPIO {OPEN_PIN}) ---")
        for i in range(NUM_PULSES):
            logger.info(f"Імпульс OPEN #{i + 1}/{NUM_PULSES}")
            send_pulse(open_relay, "OPEN", PULSE_DURATION)
            if i < NUM_PULSES - 1:  # Не робити паузу після останнього імпульсу цієї серії
                logger.debug(f"Пауза {PAUSE_BETWEEN_PULSES}с...")
                time.sleep(PAUSE_BETWEEN_PULSES)

        logger.info(f"\nПауза перед тестуванням реле CLOSE: {PAUSE_BETWEEN_RELAYS}с...")
        time.sleep(PAUSE_BETWEEN_RELAYS)

        # Тестування реле CLOSE
        logger.info(f"\n--- Тестування реле CLOSE (GPIO {CLOSE_PIN}) ---")
        for i in range(NUM_PULSES):
            logger.info(f"Імпульс CLOSE #{i + 1}/{NUM_PULSES}")
            send_pulse(close_relay, "CLOSE", PULSE_DURATION)
            if i < NUM_PULSES - 1:
                logger.debug(f"Пауза {PAUSE_BETWEEN_PULSES}с...")
                time.sleep(PAUSE_BETWEEN_PULSES)

        logger.info("\nТестування імпульсів завершено.")

    except Exception as e:
        logger.critical(f"Загальна помилка під час тестування реле: {e}", exc_info=True)
    finally:
        logger.info("Очищення ресурсів GPIO...")
        if open_relay:
            open_relay.off()  # Переконуємося, що вимкнене перед закриттям
            open_relay.close()
            logger.info(f"Реле OPEN (GPIO {OPEN_PIN}) закрито.")
        if close_relay:
            close_relay.off()  # Переконуємося, що вимкнене перед закриттям
            close_relay.close()
            logger.info(f"Реле CLOSE (GPIO {CLOSE_PIN}) закрито.")
        logger.info("Очищення завершено.")


if __name__ == "__main__":
    main()