# core/gate_controller.py
import time
import logging
from typing import Optional

try:
    from gpiozero import OutputDevice
except ImportError:
    # Мок-клас для тестування на ПК
    class OutputDevice:
        def __init__(self, pin, active_high=True, initial_value=False):
            self.pin, self._is_active = pin, initial_value
            self.logger = logging.getLogger(f"MockOutputDevice.Pin{pin}")
        def on(self): self._is_active = True; self.logger.info(f"Мок pin {self.pin} -> УВІМКНЕНО")
        def off(self): self._is_active = False; self.logger.info(f"Мок pin {self.pin} -> ВИМКНЕНО")
        def close(self): self.logger.info(f"Мок pin {self.pin} закрито.")

class GateController:
    def __init__(self,
                 open_relay_pin: int,
                 close_relay_pin: int,
                 relay_pulse_duration_s: float = 0.5):

        self.relay_pulse_duration_s = relay_pulse_duration_s
        self._logger = logging.getLogger(f"{__name__}.GateController")
        self.open_relay: Optional[OutputDevice] = None
        self.close_relay: Optional[OutputDevice] = None
        self.relays_initialized = False

        try:
            # Ініціалізуємо реле з правильними параметрами для low-level trigger
            self.open_relay = OutputDevice(open_relay_pin, active_high=False, initial_value=True)
            self.close_relay = OutputDevice(close_relay_pin, active_high=False, initial_value=True)
            
            # --- ВИПРАВЛЕННЯ ТУТ ---
            # Явно встановлюємо неактивний стан (HIGH) одразу після ініціалізації
            self.open_relay.off()
            self.close_relay.off()
            # ---------------------

            self.relays_initialized = True
            self._logger.info(f"Реле ініціалізовано та встановлено у неактивний стан (HIGH): OPEN на GPIO{open_relay_pin}, CLOSE на GPIO{close_relay_pin}.")
        except Exception as e:
            self._logger.critical(f"Критична помилка ініціалізації реле: {e}", exc_info=True)

    def _activate_relay_pulse(self, relay_device: Optional[OutputDevice], action_name: str):
        if not self.relays_initialized or not relay_device:
            self._logger.error(f"Реле для '{action_name}' не ініціалізовано.")
            return

        self._logger.info(f"КОМАНДА: Подача імпульсу LOW на реле '{action_name}'.")
        try:
            # .on() для active_high=False реле подає сигнал LOW
            relay_device.on()
            time.sleep(self.relay_pulse_duration_s)
        finally:
            # .off() для active_high=False реле повертає сигнал HIGH
            relay_device.off()

    def open_gate(self):
        """Надсилає імпульс на відкриття воріт."""
        self._activate_relay_pulse(self.open_relay, "OPEN")

    def close_gate(self):
        """Надсилає імпульс на закриття воріт."""
        self._activate_relay_pulse(self.close_relay, "CLOSE")

    def cleanup(self):
        self._logger.info("Очищення ресурсів GateController...")
        if self.relays_initialized:
            # Перед закриттям ще раз гарантовано встановлюємо неактивний стан
            if self.open_relay:
                self.open_relay.off()
                self.open_relay.close()
            if self.close_relay:
                self.close_relay.off()
                self.close_relay.close()
