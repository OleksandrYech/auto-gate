# core/camera_manager.py
import time
import logging
from picamera2 import Picamera2, Preview
from libcamera import Transform
import os
import numpy as np
from typing import Optional, Dict, Any

# Налаштування логування для модуля camera_manager
logger = logging.getLogger(__name__)

# Очікувані частини назв моделей сенсорів для ідентифікації камер
DEFAULT_ENTRY_CAM_MODEL_SUBSTRING = 'imx708'  # Для Camera Module 3
DEFAULT_EXIT_CAM_MODEL_SUBSTRING = 'imx219'  # Для Camera Module 2 (або інший, що використовується як другий)

# Шлях за замовчуванням для збереження зображень, якщо CameraManager його використовує
DEFAULT_IMAGE_SAVE_PATH = "captured_images"


def get_camera_ids(entry_cam_model_sub: str = DEFAULT_ENTRY_CAM_MODEL_SUBSTRING,
                   exit_cam_model_sub: str = DEFAULT_EXIT_CAM_MODEL_SUBSTRING) -> dict:
    """
    Знаходить індекси (camera_num) для камери в'їзду та виїзду на основі
    підрядків у назвах їх моделей сенсорів.

    Args:
        entry_cam_model_sub (str): Підрядок для ідентифікації моделі камери в'їзду.
        exit_cam_model_sub (str): Підрядок для ідентифікації моделі камери виїзду.

    Returns:
        dict: Словник з ключами "entry" та "exit" та відповідними індексами камер (camera_num),
              або None для відповідного ключа, якщо камеру не знайдено.
              Приклад: {"entry": 0, "exit": 1}
    """
    camera_ids = {"entry": None, "exit": None}
    try:
        cameras_info = Picamera2.global_camera_info()
        if not cameras_info:
            logger.warning("Picamera2.global_camera_info() не знайшло жодної камери.")
            return camera_ids

        logger.info(f"Виявлені камери: {cameras_info}")

        # Спочатку шукаємо камеру в'їзду
        for info in cameras_info:
            model = info.get("Model", "").lower()
            cam_num = info.get("Num", cameras_info.index(info))  # Використовуємо 'Num' або індекс

            if entry_cam_model_sub in model:
                camera_ids["entry"] = cam_num
                logger.info(
                    f"Камера В'ЇЗДУ (підрядок '{entry_cam_model_sub}') знайдена: "
                    f"{info.get('Location', '')} {model} (ID: {cam_num})"
                )
                break  # Знайшли камеру в'їзду

        # Потім шукаємо камеру виїзду серед решти, уникаючи вже знайденої камери в'їзду
        for info in cameras_info:
            model = info.get("Model", "").lower()
            cam_num = info.get("Num", cameras_info.index(info))

            if exit_cam_model_sub in model and cam_num != camera_ids["entry"]:
                camera_ids["exit"] = cam_num
                logger.info(
                    f"Камера ВИЇЗДУ (підрядок '{exit_cam_model_sub}') знайдена: "
                    f"{info.get('Location', '')} {model} (ID: {cam_num})"
                )
                break  # Знайшли камеру виїзду

        # Якщо камеру виїзду не знайдено з її унікальним ID, а камер лише дві,
        # і одна вже визначена як в'їзна, то інша може бути виїзною.
        if camera_ids["exit"] is None and len(cameras_info) == 2 and camera_ids["entry"] is not None:
            logger.info(f"Камеру виїзду з підрядком '{exit_cam_model_sub}' не знайдено окремо.")
            for info in cameras_info:
                cam_num = info.get("Num", cameras_info.index(info))
                if cam_num != camera_ids["entry"]:
                    camera_ids["exit"] = cam_num
                    model = info.get("Model", "").lower()
                    logger.info(
                        f"Призначено другу доступну камеру як ВИЇЗНУ: "
                        f"{info.get('Location', '')} {model} (ID: {cam_num})"
                    )
                    break

        if camera_ids["entry"] is None:
            logger.warning(f"Камера В'ЇЗДУ (підрядок моделі '{entry_cam_model_sub}') НЕ знайдена.")
        if camera_ids["exit"] is None:
            logger.warning(f"Камера ВИЇЗДУ (підрядок моделі '{exit_cam_model_sub}') НЕ знайдена.")

    except Exception as e:
        logger.error(f"Помилка під час отримання ID камер: {e}", exc_info=True)
    return camera_ids


class CameraController:
    """
    Клас для керування камерою Raspberry Pi за допомогою Picamera2.
    """

    def __init__(self, camera_id: int, camera_name: str = "UnnamedCamera",
                 capture_resolution: tuple = (1920, 1080),
                 hflip: bool = False, vflip: bool = False,
                 image_save_path_base: str = DEFAULT_IMAGE_SAVE_PATH,  # Базовий шлях
                 camera_type_for_path: str = "unknown_cam"  # "entry" або "exit"
                 ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.capture_resolution = capture_resolution
        self.picam2: Optional[Picamera2] = None
        self._logger = logging.getLogger(f"{__name__}.{self.camera_name}_ID{self.camera_id}")

        # Формуємо шлях для збереження зображень для цієї камери
        self.image_save_path_for_this_camera = os.path.join(image_save_path_base, camera_type_for_path)
        os.makedirs(self.image_save_path_for_this_camera, exist_ok=True)

        self.is_initialized_successfully = False

        try:
            self.picam2 = Picamera2(camera_num=self.camera_id)
            cam_model_info = self.picam2.camera_properties.get('Model', 'Невідома модель')
            self._logger.info(
                f"Камера '{self.camera_name}' (ID: {self.camera_id}, Модель: {cam_model_info}) ініціалізується...")

            transform = Transform(hflip=hflip, vflip=vflip)
            config = self.picam2.create_still_configuration(
                main={"size": self.capture_resolution},
                lores={"size": (640, 480)},  # Для швидкого прев'ю або маленьких стрімів
                display="lores",  # Використовуємо lores для display stream
                transform=transform
            )
            self.picam2.configure(config)
            self._logger.info(
                f"Налаштовано для фото з роздільною здатністю {self.capture_resolution}, трансформація: hflip={hflip}, vflip={vflip}.")

            self.picam2.start()
            self._logger.info("Камеру запущено.")

            # Коротка пауза для стабілізації камери (автоекспозиція, баланс білого)
            self._logger.info("Очікування 2-3 секунди для автоналаштування камери...")
            time.sleep(2.5)
            self._logger.info("Камера готова.")
            self.is_initialized_successfully = True

        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати або запустити камеру '{self.camera_name}': {e}",
                               exc_info=True)
            if self.picam2:
                try:
                    self.picam2.close()
                except Exception as close_e:
                    self._logger.error(
                        f"Помилка закриття камери '{self.camera_name}' під час невдалої ініціалізації: {close_e}")
            self.picam2 = None

    def capture_image(self, filename_only: str = "capture.jpg") -> Optional[str]:
        """Захоплює зображення та зберігає його у файл у директорії камери."""
        if not self.is_initialized_successfully or not self.picam2 or not self.picam2.started:
            self._logger.error(f"Камера '{self.camera_name}' не готова. Неможливо захопити зображення.")
            return None

        full_output_path = os.path.join(self.image_save_path_for_this_camera, filename_only)

        try:
            self._logger.info(f"'{self.camera_name}': Захоплення зображення у файл {full_output_path}...")
            # capture_file сам створює директорії, якщо їх немає, але ми вже створили базову
            self.picam2.capture_file(full_output_path)
            self._logger.info(f"'{self.camera_name}': Зображення захоплено у {full_output_path}.")
            return full_output_path
        except Exception as e:
            self._logger.error(f"'{self.camera_name}': Не вдалося захопити зображення у {full_output_path}: {e}",
                               exc_info=True)
            return None

    def capture_array(self, stream_name: str = "main") -> Optional[np.ndarray]:
        """Захоплює зображення як масив numpy зі вказаного стріму."""
        if not self.is_initialized_successfully or not self.picam2 or not self.picam2.started:
            self._logger.error(f"Камера '{self.camera_name}' не готова. Неможливо захопити масив.")
            return None
        try:
            self._logger.debug(f"'{self.camera_name}': Захоплення масиву зі стріму '{stream_name}'...")
            # Для фотографій високої якості краще використовувати capture_array("main")
            # capture_request = self.picam2.capture_request() # Більш гнучкий спосіб
            # image_array = capture_request.make_array("main")
            # capture_request.release()
            image_array = self.picam2.capture_array(stream_name)
            self._logger.debug(
                f"'{self.camera_name}': Масив захоплено (форма: {image_array.shape}, тип: {image_array.dtype}).")
            return image_array
        except Exception as e:
            self._logger.error(f"'{self.camera_name}': Не вдалося захопити масив зі стріму '{stream_name}': {e}",
                               exc_info=True)
            return None

    def start_preview(self, x: int = 100, y: int = 100, width: int = 640, height: int = 480):
        if not self.is_initialized_successfully or not self.picam2:
            self._logger.error(f"Камера '{self.camera_name}' не ініціалізована. Неможливо запустити прев'ю.")
            return
        try:
            self.picam2.start_preview(Preview.DRM, x=x, y=y, width=width, height=height)
            self._logger.info(f"'{self.camera_name}': Прев'ю запущено. Позиція: ({x},{y}), Розмір: ({width}x{height})")
        except Exception as e:
            self._logger.error(f"'{self.camera_name}': Не вдалося запустити прев'ю: {e}. "
                               "Переконайтеся, що графічне середовище доступне.")

    def stop_preview(self):
        if not self.is_initialized_successfully or not self.picam2: return
        try:
            self.picam2.stop_preview()
            self._logger.info(f"'{self.camera_name}': Прев'ю зупинено.")
        except Exception as e:  # Може виникнути помилка, якщо прев'ю не було запущено
            self._logger.debug(f"'{self.camera_name}': Помилка зупинки прев'ю (можливо, не було запущено): {e}")

    def close(self):
        if self.picam2:
            try:
                self._logger.debug(f"Закриття камери '{self.camera_name}'...")
                if self.picam2.started:
                    self.stop_preview()
                    self.picam2.stop()
                    self._logger.info(f"Камеру '{self.camera_name}' зупинено.")
                self.picam2.close()
                self._logger.info(f"Ресурси камери '{self.camera_name}' звільнено.")
            except Exception as e:
                self._logger.error(f"Помилка закриття камери '{self.camera_name}': {e}", exc_info=True)
            finally:
                self.picam2 = None  # Гарантуємо, що об'єкт скинуто
                self.is_initialized_successfully = False

    def __del__(self):
        self.close()


class CameraManager:
    """
    Клас для ініціалізації та керування камерами в'їзду та виїзду.
    """

    def __init__(self,
                 entry_cam_model_sub: str = DEFAULT_ENTRY_CAM_MODEL_SUBSTRING,
                 exit_cam_model_sub: str = DEFAULT_EXIT_CAM_MODEL_SUBSTRING,
                 entry_cam_config: Optional[dict] = None,
                 exit_cam_config: Optional[dict] = None,
                 image_base_path: str = DEFAULT_IMAGE_SAVE_PATH):
        self._logger = logging.getLogger(f"{__name__}.CameraManager")
        self._logger.info("Ініціалізація CameraManager...")

        self.cam_entry: Optional[CameraController] = None
        self.cam_exit: Optional[CameraController] = None

        # Налаштування за замовчуванням для кожної камери
        default_entry_cfg = {"name": "EntryCam", "resolution": (1920, 1080), "hflip": False, "vflip": False,
                             "type_for_path": "entry"}
        default_exit_cfg = {"name": "ExitCam", "resolution": (1280, 720), "hflip": False, "vflip": False,
                            "type_for_path": "exit"}

        cfg_entry = {**default_entry_cfg, **(entry_cam_config or {})}
        cfg_exit = {**default_exit_cfg, **(exit_cam_config or {})}

        try:
            camera_ids = get_camera_ids(entry_cam_model_sub, exit_cam_model_sub)

            if camera_ids.get("entry") is not None:
                self.cam_entry = CameraController(
                    camera_id=camera_ids["entry"],
                    camera_name=cfg_entry["name"],
                    capture_resolution=cfg_entry["resolution"],
                    hflip=cfg_entry["hflip"],
                    vflip=cfg_entry["vflip"],
                    image_base_path=image_base_path,
                    camera_type_for_path=cfg_entry["type_for_path"]
                )
                if not self.cam_entry.is_initialized_successfully:
                    self._logger.error(f"Камера в'їзду ({cfg_entry['name']}) не ініціалізувалася успішно.")
                    self.cam_entry = None
            else:
                self._logger.warning(f"Камера В'ЇЗДУ (модель '{entry_cam_model_sub}') не знайдена.")

            if camera_ids.get("exit") is not None:
                self.cam_exit = CameraController(
                    camera_id=camera_ids["exit"],
                    camera_name=cfg_exit["name"],
                    capture_resolution=cfg_exit["resolution"],
                    hflip=cfg_exit["hflip"],
                    vflip=cfg_exit["vflip"],
                    image_base_path=image_base_path,
                    camera_type_for_path=cfg_exit["type_for_path"]
                )
                if not self.cam_exit.is_initialized_successfully:
                    self._logger.error(f"Камера виїзду ({cfg_exit['name']}) не ініціалізувалася успішно.")
                    self.cam_exit = None
            else:
                self._logger.warning(f"Камера ВИЇЗДУ (модель '{exit_cam_model_sub}') не знайдена.")

        except Exception as e:  # Обробка помилок від Picamera2.global_camera_info(), якщо вони критичні
            self._logger.error(f"Критична помилка під час ініціалізації камер у CameraManager: {e}", exc_info=True)

        self._logger.info("CameraManager ініціалізацію завершено.")

    def get_entry_camera(self) -> Optional[CameraController]:
        return self.cam_entry

    def get_exit_camera(self) -> Optional[CameraController]:
        return self.cam_exit

    def close_all_cameras(self):
        self._logger.info("Закриття всіх камер через CameraManager...")
        if self.cam_entry: self.cam_entry.close()
        if self.cam_exit: self.cam_exit.close()
        self._logger.info("Всі камери закрито (або спроба закриття виконана).")

    def __del__(self):
        self.close_all_cameras()


if __name__ == '__main__':
    # Налаштовуємо логування для тесту модуля
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s')

    logger.info("Тестування модуля camera_manager.py...")

    # Створюємо тестові директорії (якщо їх ще немає)
    os.makedirs(os.path.join(DEFAULT_IMAGE_SAVE_PATH, "entry_test"), exist_ok=True)
    os.makedirs(os.path.join(DEFAULT_IMAGE_SAVE_PATH, "exit_test"), exist_ok=True)

    entry_cam_test_config = {
        "name": "TestEntryCam", "resolution": (640, 480),  # Нижча роздільна здатність для тесту
        "hflip": False, "vflip": False, "type_for_path": "entry_test"
    }
    exit_cam_test_config = {
        "name": "TestExitCam", "resolution": (640, 480),
        "hflip": True, "vflip": True, "type_for_path": "exit_test"
    }

    cam_manager = CameraManager(
        entry_cam_config=entry_cam_test_config,
        exit_cam_config=exit_cam_test_config,
        image_base_path=DEFAULT_IMAGE_SAVE_PATH  # Базова папка "captured_images"
    )

    entry_cam = cam_manager.get_entry_camera()
    exit_cam = cam_manager.get_exit_camera()

    timestamp_str = time.strftime('%Y%m%d_%H%M%S')

    if entry_cam:
        logger.info(f"--- Тестування камери В'ЇЗДУ: {entry_cam.camera_name} ---")
        # entry_cam.start_preview() # Розкоментуйте, якщо потрібне прев'ю
        # time.sleep(5)
        entry_file = entry_cam.capture_image(f"test_entry_{timestamp_str}.jpg")
        if entry_file: logger.info(f"Зображення в'їзду збережено: {entry_file}")
        # entry_cam.stop_preview()
    else:
        logger.warning("Камера в'їзду не доступна для тестування.")

    if exit_cam:
        logger.info(f"--- Тестування камери ВИЇЗДУ: {exit_cam.camera_name} ---")
        # exit_cam.start_preview(x=700) # Прев'ю в іншому місці
        # time.sleep(5)
        exit_file = exit_cam.capture_image(f"test_exit_{timestamp_str}.jpg")
        if exit_file: logger.info(f"Зображення виїзду збережено: {exit_file}")
        # exit_cam.stop_preview()
    else:
        logger.warning("Камера виїзду не доступна для тестування.")

    cam_manager.close_all_cameras()
    logger.info("Тестування модуля camera_manager.py завершено.")