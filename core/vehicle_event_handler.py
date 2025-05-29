# core/vehicle_event_handler.py
import logging
import time
import threading
import os
import numpy as np
# import cv2
from utils.image_utils import save_image

logger = logging.getLogger(__name__)

# константи
DEFAULT_REED_OPEN_TIMEOUT_S = 15
DEFAULT_REED_OPEN_RETRIES = 1  # Кількість повторних спроб відкрити
DEFAULT_REED_CLOSE_TIMEOUT_S = 5
DEFAULT_REED_CLOSE_RETRIES = 1  # Кількість повторних спроб закрити
DEFAULT_GATE_FINISH_CLOSING_DELAY_S = 10  # 10 секунд на остаточне закриття
DEFAULT_AUTO_CLOSE_TIMER_DURATION_S = 4  # Таймер на закриття після проїзду


class VehicleEventHandler:
    def __init__(self,  # ... (параметри як раніше) ...
                 config: dict = None):
        # ... (ініціалізація атрибутів як раніше) ...
        self.reed_open_timeout_s = self.config.get("reed_open_timeout_s", DEFAULT_REED_OPEN_TIMEOUT_S)
        self.reed_open_retries = self.config.get("reed_open_retries", DEFAULT_REED_OPEN_RETRIES)
        self.reed_close_timeout_s = self.config.get("reed_close_timeout_s", DEFAULT_REED_CLOSE_TIMEOUT_S)
        self.reed_close_retries = self.config.get("reed_close_retries", DEFAULT_REED_CLOSE_RETRIES)
        self.gate_finish_closing_delay_s = self.config.get("gate_finish_closing_delay_s",
                                                           DEFAULT_GATE_FINISH_CLOSING_DELAY_S)
        self.auto_close_timer_duration_s = self.config.get("auto_close_timer_duration_s",
                                                           DEFAULT_AUTO_CLOSE_TIMER_DURATION_S)

        self.ultrasonic_passage_threshold = 0.3  # м, поріг для проїзду
        # ... (решта __init__)

    # ... (_is_duplicate_log залишається) ...

    def _attempt_open_gate_with_retry(self) -> bool:
        """Намагається відкрити ворота, перевіряє геркон, робить повторні спроби."""
        if not self.gate_controller or not self.sensor_manager or not self.sensor_manager.reed_switch:
            self._logger.error("GateController або ReedSwitch не ініціалізовано для відкриття воріт.")
            return False

        for attempt in range(self.reed_open_retries + 1):  # +1 для початкової спроби
            self._logger.info(f"Спроба відкриття воріт #{attempt + 1}...")
            if not self.gate_controller.open_gate():  # open_gate сама може чекати геркон, якщо reed_confirmation_timeout_s > 0
                self._logger.warning(f"Команда open_gate() повернула помилку на спробі #{attempt + 1}.")

            start_wait = time.monotonic()
            while time.monotonic() - start_wait < self.reed_open_timeout_s:
                if self.sensor_manager.reed_switch.are_gates_open:
                    self._logger.info("Ворота успішно відкрито (підтверджено герконом).")
                    return True
                time.sleep(0.2)  # Перевірка стану геркона

            self._logger.warning(
                f"Ворота не відкрилися (геркон) протягом {self.reed_open_timeout_s}с після спроби #{attempt + 1}.")
            if attempt < self.reed_open_retries:
                self._logger.info("Повторна спроба відкриття...")
                time.sleep(1)  # Невелика пауза перед повтором
            else:
                self._logger.error(f"Не вдалося відкрити ворота після {self.reed_open_retries + 1} спроб.")
                return False
        return False  # Мало б вийти раніше

    def _wait_for_vehicle_passage_after_open(self, gate_side_name: str) -> bool:
        """
        Очікує, поки автомобіль увійде та покине зону УЗД, АЛЕ тільки якщо ворота відкриті.
        """
        if not self.sensor_manager or \
                not (getattr(self.sensor_manager, 'ultrasonic_sensor_entry', None) or \
                     getattr(self.sensor_manager, 'ultrasonic_sensor_exit', None)):
            self._logger.error("УЗД для контролю проїзду недоступний.")
            return False

        # Визначаємо, який УЗД використовувати
        passage_sensor = self.sensor_manager.ultrasonic_sensor_entry
        if gate_side_name == "exit" and hasattr(self.sensor_manager,
                                                'ultrasonic_sensor_exit') and self.sensor_manager.ultrasonic_sensor_exit:
            passage_sensor = self.sensor_manager.ultrasonic_sensor_exit

        if not passage_sensor:
            self._logger.error(f"Не вдалося визначити УЗД для сторони '{gate_side_name}'.")
            return False

        self._logger.info(
            f"Очікування проїзду автомобіля через УЗД ({gate_side_name}). Поріг: {self.ultrasonic_passage_threshold}м.")

        # 1. Перевірка, чи ворота ВІДКРИТІ (геркон замкнений)
        if not self.sensor_manager.reed_switch.are_gates_open:
            self._logger.warning(
                f"Ворота не відкриті (геркон). Скасування очікування проїзду через УЗД ({gate_side_name}).")
            return False
        self._logger.info(f"Ворота відкриті (геркон). УЗД ({gate_side_name}) активний для детекції проїзду.")

        # 2. Очікування, поки автомобіль увійде в зону УЗД (стане < 0.3м)
        # Таймаут на в'їзд у зону УЗД після відкриття воріт, наприклад, 10-15 секунд
        vehicle_entered_passage_zone = passage_sensor.wait_for_object_to_enter_passage(
            passage_threshold_m=self.ultrasonic_passage_threshold,
            timeout_s=15
        )
        if not vehicle_entered_passage_zone:
            self._logger.warning(f"Автомобіль не увійшов у зону УЗД ({gate_side_name}) після відкриття воріт.")
            return False

        # 3. Очікування, поки автомобіль покине зону УЗД (стане > 0.3м)
        # Таймаут на сам проїзд, наприклад, 10 секунд
        vehicle_cleared_passage_zone = passage_sensor.wait_for_object_to_clear_passage(
            passage_threshold_m=self.ultrasonic_passage_threshold,
            timeout_s=10
        )

        if vehicle_cleared_passage_zone:
            self._logger.info(f"Автомобіль повністю проїхав зону УЗД ({gate_side_name}).")
            return True
        else:
            self._logger.warning(f"Автомобіль не покинув зону УЗД ({gate_side_name}) або таймаут.")
            # Якщо авто все ще в зоні, це може бути перешкода
            if passage_sensor.detect_object_in_passage(self.ultrasonic_passage_threshold):
                self._logger.warning(f"УВАГА: Автомобіль або перешкода все ще в зоні УЗД '{gate_side_name}'!")
            return False

    def _manage_auto_close_with_obstruction_check(self):
        """
        Запускає таймер на 4с, моніторить УЗД, перезапускає таймер при перешкоді.
        Повертає True, якщо 4с пройшли без перешкод, False - якщо скасовано або помилка.
        """
        if not self.gate_controller or not self.sensor_manager or \
                not (getattr(self.sensor_manager, 'ultrasonic_sensor_entry', None) or \
                     getattr(self.sensor_manager, 'ultrasonic_sensor_exit', None)):
            self._logger.error("Неможливо керувати автозакриттям: компоненти не ініціалізовані.")
            return False

        # Визначаємо, який УЗД використовувати для перевірки проїзду (зазвичай той, що контролює сам проріз)
        passage_check_sensor = self.sensor_manager.ultrasonic_sensor_entry
        # Якщо є окремий для виїзду, і це сценарій виїзду, можна його вибрати, але для прорізу зазвичай один.

        start_time = time.monotonic()
        self.gate_controller.start_auto_close_timer(timeout_s=self.auto_close_timer_duration_s)

        while time.monotonic() - start_time < self.auto_close_timer_duration_s:
            if self.shutdown_event and self.shutdown_event.is_set(): return False  # Перевірка на завершення

            if passage_check_sensor.detect_object_in_passage(self.ultrasonic_passage_threshold):
                self._logger.info(
                    f"Перешкода виявлена УЗД під час {self.auto_close_timer_duration_s}с таймера! Зупинка та очікування.")
                self.gate_controller.cancel_auto_close_timer()

                # Очікуємо, поки перешкода зникне
                obstacle_cleared = passage_check_sensor.wait_for_object_to_clear_passage(
                    passage_threshold_m=self.ultrasonic_passage_threshold,
                    timeout_s=60  # Довгий таймаут на зникнення перешкоди
                )
                if not obstacle_cleared:
                    self._logger.warning("Перешкода не зникла з зони УЗД після очікування. Автозакриття скасовано.")
                    return False  # Не вдалося очистити, не закриваємо

                self._logger.info("Перешкода зникла. Перезапуск таймера автозакриття.")
                start_time = time.monotonic()  # Скидаємо таймер цього циклу
                self.gate_controller.start_auto_close_timer(timeout_s=self.auto_close_timer_duration_s)
                # Продовжуємо цикл моніторингу з початку

            time.sleep(0.1)  # Інтервал перевірки УЗД під час таймера

        # Якщо цикл завершився, означає, що 4 секунди пройшли без постійної перешкоди
        # І таймер GateController має спрацювати (або вже спрацював і викликав close_gate)
        self._logger.info(
            f"{self.auto_close_timer_duration_s}с таймер завершився. GateController має ініціювати закриття.")
        return True

    def _attempt_close_gate_with_retry(self) -> bool:
        """Намагається закрити ворота, перевіряє геркон, робить повторні спроби."""
        if not self.gate_controller or not self.sensor_manager or not self.sensor_manager.reed_switch:
            self._logger.error("GateController або ReedSwitch не ініціалізовано для закриття воріт.")
            return False

        for attempt in range(self.reed_close_retries + 1):
            self._logger.info(f"Спроба закриття воріт #{attempt + 1}...")
            # GateController.close_gate() вже включає перевірку УЗД на перешкоду
            if not self.gate_controller.close_gate():  # Якщо close_gate повернув False (напр. через перешкоду)
                # self.gate_controller.gate_state вже має бути "OBSTRUCTED"
                self._logger.warning(f"Команда close_gate() не виконана (можливо, перешкода) на спробі #{attempt + 1}.")
                # У цьому випадку повторна спроба може бути недоцільною, якщо перешкода не зникла.
                # Але якщо проблема була тимчасовою, повтор може допомогти.
                # Залишаємо логіку повтору, але це місце для можливого покращення.
                if attempt < self.reed_close_retries:
                    time.sleep(1)  # Пауза перед повтором
                    continue  # Продовжуємо з наступною спробою
                else:  # Якщо це була остання спроба і вона невдала через перешкоду або іншу причину
                    return False

                    # Очікування підтвердження від геркона
            start_wait = time.monotonic()
            while time.monotonic() - start_wait < self.reed_close_timeout_s:
                if self.sensor_manager.reed_switch.are_gates_closed:
                    self._logger.info("Ворота успішно закрито (підтверджено герконом).")
                    return True
                time.sleep(0.2)

            self._logger.warning(
                f"Ворота не закрилися (геркон) протягом {self.reed_close_timeout_s}с після спроби #{attempt + 1}.")
            if attempt < self.reed_close_retries:
                self._logger.info("Повторна спроба закриття...")
                time.sleep(1)
            else:
                self._logger.error(f"Не вдалося закрити ворота після {self.reed_close_retries + 1} спроб.")
                return False
        return False

    # ... (entry_scenario_loop та exit_scenario_loop будуть оновлені для використання цих методів) ...

    def entry_scenario_loop(self):
        self._logger.info("Запуск циклу обробки В'ЇЗДУ...")
        while self.is_running and (self.shutdown_event is None or not self.shutdown_event.is_set()):
            # ... (логіка переривання закриття _handle_gate_closing_interruption) ...
            if hasattr(self.gate_controller, '_auto_close_timer') and \
                    self.gate_controller._auto_close_timer is not None and \
                    self.gate_controller._auto_close_timer.is_alive():
                current_poll_interval = self.poll_interval_gate_closing_s
                if self._handle_gate_closing_interruption("entry"):
                    self._logger.info("В'їзд: Закриття перервано новим авто. Повторний цикл детекції.")
                    time.sleep(0.1)
                    continue
            else:
                current_poll_interval = self.poll_interval_idle_s

            self._logger.debug("В'їзд: Очікування автомобіля (тільки CV)...")
            vehicle_detected_by_cv = False
            initial_detection_frame = None

            if self.camera_entry and self.camera_entry.is_initialized_successfully and self.cv_processor:
                frame_for_detection = self.camera_entry.capture_array()
                if frame_for_detection is not None:
                    vehicle_detections = self.cv_processor.detect_vehicle_in_frame(frame_for_detection,
                                                                                   camera_type="entry")
                    if vehicle_detections:
                        self._logger.info("В'їзд: CV зафіксував автомобіль.")
                        vehicle_detected_by_cv = True
                        initial_detection_frame = frame_for_detection

            if vehicle_detected_by_cv:
                self._logger.info("В'їзд: Автомобіль виявлено через CV. Початок обробки.")
                if initial_detection_frame is None:  # Малоймовірно
                    self._logger.error("В'їзд: Помилка - CV виявив авто, але кадр не збережено.")
                    time.sleep(current_poll_interval);
                    continue

                timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                plate_text = self.cv_processor.get_plate_number_from_image(
                    initial_detection_frame, camera_type="entry",
                    save_intermediate_steps=True,
                    save_path_prefix=os.path.join(CV_DEBUG_SAVE_PATH, "entry")
                )

                if plate_text:
                    if not self._is_duplicate_log(plate_text):
                        if self.sheet_handler.find_vehicle_and_update_entry_time(plate_text):
                            self._logger.info(f"В'їзд: Авто '{plate_text}' АВТОРИЗОВАНО.")
                            if self._attempt_open_gate_with_retry():  # Крок 3, 4
                                if self._wait_for_vehicle_passage_after_open(
                                        "в'їзду"):  # Крок 5, 6 (з умовою на відкриті ворота)
                                    if self._manage_auto_close_with_obstruction_check():  # Крок 7, 8
                                        if self._attempt_close_gate_with_retry():  # Крок 9, 10
                                            self._logger.info(
                                                f"В'їзд: Цикл для '{plate_text}' завершено, ворота закрито. Очікування {self.gate_finish_closing_delay_s}с.")
                                            time.sleep(self.gate_finish_closing_delay_s)  # Крок 11
                                        else:
                                            self._logger.error("В'їзд: Не вдалося підтвердити закриття воріт.")
                                    # else: Закриття було скасовано через перешкоду під час таймера, або _manage_auto_close... повернув False
                                else:
                                    self._logger.warning(
                                        "В'їзд: Авто не підтвердило проїзд. Ворота залишаються відкритими (потрібна логіка примусового закриття?).")
                            else:
                                self._logger.error("В'їзд: Не вдалося відкрити ворота для авторизованого авто.")
                        else:
                            self._logger.info(f"В'їзд: Авто '{plate_text}' НЕ АВТОРИЗОВАНО.")
                            self.sheet_handler.add_unauthorized_attempt(plate_text)
                else:
                    self._logger.warning("В'їзд: Номерний знак не розпізнано.")
                    save_image(initial_detection_frame, ENTRY_IMAGES_PATH, f"entry_ocr_failed_{timestamp_str}.jpg")

            time.sleep(current_poll_interval)
        self._logger.info("Цикл обробки В'ЇЗДУ завершено.")

    def exit_scenario_loop(self):
        self._logger.info("Запуск циклу обробки ВИЇЗДУ...")
        while self.is_running and (self.shutdown_event is None or not self.shutdown_event.is_set()):
            # ... (логіка переривання закриття _handle_gate_closing_interruption) ...
            if hasattr(self.gate_controller, '_auto_close_timer') and \
                    self.gate_controller._auto_close_timer is not None and \
                    self.gate_controller._auto_close_timer.is_alive():
                current_poll_interval = self.poll_interval_gate_closing_s
                if self._handle_gate_closing_interruption("exit"):
                    self._logger.info("Виїзд: Закриття перервано новим авто. Повторний цикл детекції.")
                    time.sleep(0.1)
                    continue
            else:
                current_poll_interval = self.poll_interval_idle_s

            self._logger.debug("Виїзд: Очікування автомобіля (тільки CV)...")
            vehicle_detected_by_cv_exit = False
            initial_detection_frame_exit = None

            if self.camera_exit and self.camera_exit.is_initialized_successfully and self.cv_processor:
                frame_for_detection = self.camera_exit.capture_array()
                if frame_for_detection is not None:
                    vehicle_detections = self.cv_processor.detect_vehicle_in_frame(frame_for_detection,
                                                                                   camera_type="exit")
                    if vehicle_detections:
                        self._logger.info("Виїзд: CV зафіксував автомобіль.")
                        vehicle_detected_by_cv_exit = True
                        initial_detection_frame_exit = frame_for_detection

            if vehicle_detected_by_cv_exit:
                self._logger.info("Виїзд: Автомобіль виявлено через CV. Негайне відкриття воріт.")
                if self._attempt_open_gate_with_retry():  # Крок 3, 4
                    # Отримання кадру для логування НЗ ПІСЛЯ відкриття
                    if initial_detection_frame_exit is None and self.camera_exit and self.camera_exit.is_initialized_successfully:
                        initial_detection_frame_exit = self.camera_exit.capture_array()  # Можливо, авто вже трохи змістилося

                    if initial_detection_frame_exit is not None:
                        timestamp_str = time.strftime('%Y%m%d_%H%M%S')
                        plate_text = self.cv_processor.get_plate_number_from_image(
                            initial_detection_frame_exit, camera_type="exit",
                            save_intermediate_steps=True,
                            save_path_prefix=os.path.join(CV_DEBUG_SAVE_PATH, "exit")
                        )
                        if plate_text:
                            if not self._is_duplicate_log(plate_text):
                                self.sheet_handler.log_vehicle_exit(plate_text)
                        else:
                            self._logger.warning("Виїзд: Номерний знак не розпізнано, але виїзд дозволено.")
                            save_image(initial_detection_frame_exit, EXIT_IMAGES_PATH,
                                       f"exit_ocr_failed_{timestamp_str}.jpg")
                    else:
                        self._logger.error("Виїзд: Не вдалося отримати кадр для логування НЗ.")

                    if self._wait_for_vehicle_passage_after_open("виїзду"):  # Крок 5, 6
                        if self._manage_auto_close_with_obstruction_check():  # Крок 7, 8
                            if self._attempt_close_gate_with_retry():  # Крок 9, 10
                                self._logger.info(
                                    f"Виїзд: Цикл для виїзду завершено, ворота закрито. Очікування {self.gate_finish_closing_delay_s}с.")
                                time.sleep(self.gate_finish_closing_delay_s)  # Крок 11
                            else:
                                self._logger.error("Виїзд: Не вдалося підтвердити закриття воріт.")
                        # else: Закриття скасовано
                    else:
                        self._logger.warning(
                            "Виїзд: Авто не підтвердило проїзд. Ворота залишаються відкритими (потрібна логіка?).")
                else:
                    self._logger.error("Виїзд: Не вдалося відкрити ворота для авто на виїзд.")

            time.sleep(current_poll_interval)
        self._logger.info("Цикл обробки ВИЇЗДУ завершено.")

    # ... (start, stop як раніше) ...
    def start(self, shutdown_event_main):
        if self.is_running:
            self._logger.warning("VehicleEventHandler вже запущено.")
            return
        self._logger.info("Запуск VehicleEventHandler...")
        self.is_running = True
        self.shutdown_event = shutdown_event_main
        self.entry_thread = threading.Thread(target=self.entry_scenario_loop, name="EntryScenarioThread")
        self.exit_thread = threading.Thread(target=self.exit_scenario_loop, name="ExitScenarioThread")
        self.entry_thread.daemon = True
        self.exit_thread.daemon = True
        self.entry_thread.start()
        self.exit_thread.start()
        self._logger.info("Потоки обробки в'їзду та виїзду запущено.")

    def stop(self):
        if not self.is_running:
            self._logger.info("VehicleEventHandler вже зупинено або не було запущено.")
            return
        self._logger.info("Зупинка VehicleEventHandler...")
        self.is_running = False
        self._logger.info("VehicleEventHandler отримав сигнал на зупинку.")


# --- Блок для тестування ---
if __name__ == '__main__':
    # ... (Мок-класи та логіка тестування оновлюються для відповідності новим методам) ...
    pass  # Залиште або адаптуйте тестовий блок з попередньої версії