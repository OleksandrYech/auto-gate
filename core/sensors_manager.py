# core/sensors_manager.py
import logging
from typing import Optional

try:
    from gpiozero import DigitalInputDevice
    from gpiozero.pins.pigpio import PiGPIOFactory

    GPIOZERO_AVAILABLE = True

    try:
        factory = PiGPIOFactory()
        logging.getLogger(__name__).info("Використовується PiGPIOFactory для керування пінами.")
    except Exception:
        factory = None
        logging.getLogger(__name__).warning(
            "Не вдалося ініціалізувати PiGPIOFactory. Використовується фабрика за замовчуванням.")

except ImportError:
    GPIOZERO_AVAILABLE = False
    factory = None

    class DigitalInputDevice:
        def __init__(self, pin, pull_up=None, pin_factory=None):
            self.pin = pin
            # Імітуємо, що ворота спочатку закриті (геркон розімкнений)
            self.is_active = False
            logging.getLogger(__name__).info(f"Мок DigitalInputDevice створено для піна {pin}")

        def close(self): pass
        def wait_for_active(self, timeout=None): self.is_active = True; return True
        def wait_for_inactive(self, timeout=None): self.is_active = False; return True


class ReedSwitch:
    """Клас для роботи з герконовим датчиком."""

    def __init__(self, pin_number: int, name: str = "ReedSwitch"):
        self.name = name
        self.pin_number = pin_number
        self._logger = logging.getLogger(f"{__name__}.{self.name}")
        self._device: Optional[DigitalInputDevice] = None

        if not GPIOZERO_AVAILABLE:
            self._logger.warning("gpiozero недоступна. Геркон працюватиме в мок-режимі.")
            self._device = DigitalInputDevice(pin_number, pull_up=True)
            return

        try:
            # pull_up=True: is_active стає True, коли геркон замикається на землю (ворота ВІДКРИТІ).
            self._device = DigitalInputDevice(pin_number, pull_up=True, pin_factory=factory)
            self._logger.info(f"Геркон '{self.name}' ініціалізовано на GPIO{self.pin_number}.")
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати геркон '{self.name}': {e}", exc_info=True)

    @property
    def are_gates_open(self) -> bool:
        """Повертає True, якщо ворота повністю відкриті (контакт геркона замкнений)."""
        if self._device is None: return False
        return self._device.is_active

    @property
    def is_gate_closed_or_moving(self) -> bool:
        """Повертає True, якщо ворота не у повністю відкритому положенні (геркон розімкнений)."""
        if self._device is None: return True  # Безпечне значення за замовчуванням
        return not self._device.is_active

    def wait_for_open(self, timeout: Optional[float] = None) -> bool:
        """Очікує, поки ворота повністю відкриються (геркон замкнеться)."""
        if not self._device: return False
        try:
            return self._device.wait_for_active(timeout)
        except Exception:
            return False

    def wait_for_closed(self, timeout: Optional[float] = None) -> bool:
        """
        Очікує, поки ворота почнуть закриватися (геркон розімкнеться).
        """
        if not self._device: return False
        try:
            return self._device.wait_for_inactive(timeout)
        except Exception:
            return False

    def cleanup(self):
        if self._device:
            self._device.close()
            self._logger.info(f"Ресурси геркона '{self.name}' звільнено.")


class SensorManager:
    """Керує всіма сенсорами системи."""

    def __init__(self, reed_pin: int, reed_name: str = "GateReedSensor"):
        self._logger = logging.getLogger(f"{__name__}.SensorManager")
        self._logger.info("Ініціалізація SensorManager...")
        self.reed_switch: Optional[ReedSwitch] = None
        try:
            self.reed_switch = ReedSwitch(pin_number=reed_pin, name=reed_name)
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати {reed_name}: {e}", exc_info=True)
        self._logger.info("SensorManager ініціалізацію завершено.")

    def cleanup(self):
        if self.reed_switch:
            self.reed_switch.cleanup()
