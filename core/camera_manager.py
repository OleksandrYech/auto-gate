# core/camera_manager.py
import time
import logging
from typing import Optional, Dict, Any  # Додано для типізації

# Намагаємося імпортувати бібліотеки Raspberry Pi
try:
    from picamera2 import Picamera2, Preview  # type: ignore
    from libcamera import Transform  # type: ignore

    PICAMERA2_AVAILABLE = True
except ImportError:
    PICAMERA2_AVAILABLE = False


    # Створюємо заглушки, якщо бібліотеки недоступні (для тестування на ПК)
    class Picamera2:
        def __init__(self, camera_num=0): self.logger = logging.getLogger("MockPicamera2"); self.logger.info(
            f"Мок Picamera2 створено для камери {camera_num}.")

        def camera_properties(self): return {}

        def create_still_configuration(self, main=None, lores=None, display=None, transform=None): return {}

        def configure(self, config): pass

        def start(self): pass

        def capture_file(self, filepath): self.logger.info(f"Мок: capture_file({filepath})"); return {}

        def capture_array(self, stream_name="main"): self.logger.info(
            f"Мок: capture_array({stream_name})"); return np.zeros((480, 640, 3), dtype=np.uint8)

        def start_preview(self, preview_type, x=0, y=0, width=0, height=0): self.logger.info("Мок: start_preview()")

        def stop_preview(self): self.logger.info("Мок: stop_preview()")

        def stop(self): self.logger.info("Мок: stop()")

        def close(self): self.logger.info("Мок: close()")

        @property
        def started(self): return True  # Імітуємо, що камера завжди "запущена" для мока

        @staticmethod
        def global_camera_info(): return []  # Повертаємо порожній список, якщо реальна бібліотека недоступна


    class Transform:  # type: ignore
        def __init__(self, hflip=False, vflip=False): pass


    class Preview:  # type: ignore
        DRM = None

import os
import numpy as np

logger = logging.getLogger(__name__)

DEFAULT_ENTRY_CAM_MODEL_SUBSTRING = 'imx219'
DEFAULT_EXIT_CAM_MODEL_SUBSTRING = 'imx219'
DEFAULT_IMAGE_SAVE_PATH = "captured_images"


def get_camera_ids(entry_cam_model_sub: str = DEFAULT_ENTRY_CAM_MODEL_SUBSTRING,
                   exit_cam_model_sub: str = DEFAULT_EXIT_CAM_MODEL_SUBSTRING) -> Dict[str, Optional[int]]:
    camera_ids: Dict[str, Optional[int]] = {"entry": None, "exit": None}
    if not PICAMERA2_AVAILABLE:
        logger.warning("Бібліотека Picamera2 недоступна. Повернення порожніх ID камер.")
        return camera_ids
    try:
        cameras_info = Picamera2.global_camera_info()
        if not cameras_info:
            logger.warning("Picamera2.global_camera_info() не знайшло жодної камери.")
            return camera_ids

        logger.info(f"Виявлені камери: {cameras_info}")

        # Шукаємо камери за підрядками моделей
        found_entry_cam_num = None
        found_exit_cam_num = None

        for info in cameras_info:
            model = info.get("Model", "").lower()
            cam_num = info.get("Num", cameras_info.index(info))

            if entry_cam_model_sub in model and found_entry_cam_num is None:
                found_entry_cam_num = cam_num
            elif exit_cam_model_sub in model and found_exit_cam_num is None and cam_num != found_entry_cam_num:
                # Переконуємося, що це не та сама камера, якщо підрядки можуть перетинатися
                found_exit_cam_num = cam_num

        camera_ids["entry"] = found_entry_cam_num
        camera_ids["exit"] = found_exit_cam_num

        # Додаткова логіка, якщо знайдено лише одну камеру, або якщо підрядки однакові
        if len(cameras_info) == 1 and found_entry_cam_num is not None:
            logger.info(
                f"Знайдено лише одну камеру (ID: {found_entry_cam_num}), яка відповідає критеріям в'їзної камери ('{entry_cam_model_sub}'). Призначається як в'їзна.")
        elif len(cameras_info) >= 2 and found_entry_cam_num is not None and found_exit_cam_num is None:
            # Якщо в'їзна знайдена, а виїзна за її унікальним підрядком - ні,
            # спробуємо призначити іншу доступну камеру як виїзну.
            for info in cameras_info:
                cam_num = info.get("Num", cameras_info.index(info))
                if cam_num != found_entry_cam_num:
                    camera_ids["exit"] = cam_num
                    logger.info(
                        f"Камера виїзду з підрядком '{exit_cam_model_sub}' не знайдена. Призначено другу камеру (ID: {cam_num}, Модель: {info.get('Model', '')}) як виїзну.")
                    break

        if camera_ids["entry"] is not None:
            logger.info(f"Камера В'ЇЗДУ (ID: {camera_ids['entry']}) успішно ідентифікована.")
        else:
            logger.warning(f"Камера В'ЇЗДУ (підрядок моделі '{entry_cam_model_sub}') НЕ знайдена.")

        if camera_ids["exit"] is not None:
            logger.info(f"Камера ВИЇЗДУ (ID: {camera_ids['exit']}) успішно ідентифікована.")
        else:
            logger.warning(f"Камера ВИЇЗДУ (підрядок моделі '{exit_cam_model_sub}') НЕ знайдена.")

    except Exception as e:
        logger.error(f"Помилка під час отримання ID камер: {e}", exc_info=True)
    return camera_ids


class CameraController:
    def __init__(self, camera_id: int, camera_name: str = "UnnamedCamera",
                 capture_resolution: tuple = (1920, 1080),
                 hflip: bool = False, vflip: bool = False,
                 image_base_path: str = DEFAULT_IMAGE_SAVE_PATH,
                 camera_type_for_path: str = "unknown_cam"
                 ):
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.capture_resolution = capture_resolution
        self.picam2: Optional[Picamera2] = None  # Явна ініціалізація None
        self._logger = logging.getLogger(f"{__name__}.{self.camera_name}_ID{self.camera_id}")

        self.image_save_path_for_this_camera = os.path.join(image_base_path, camera_type_for_path)
        try:  # Створюємо директорію одразу
            os.makedirs(self.image_save_path_for_this_camera, exist_ok=True)
        except OSError as e:
            self._logger.error(
                f"Не вдалося створити директорію для збереження зображень камери '{self.camera_name}': {self.image_save_path_for_this_camera}. Помилка: {e}")
            # Можна або кинути виняток, або продовжити, але збереження не працюватиме
            self.is_initialized_successfully = False
            return

        self.is_initialized_successfully = False

        if not PICAMERA2_AVAILABLE:
            self._logger.error(
                f"Бібліотека Picamera2 недоступна. Неможливо ініціалізувати камеру '{self.camera_name}'.")
            return

        try:
            self.picam2 = Picamera2(camera_num=self.camera_id)
            cam_model_info = self.picam2.camera_properties.get('Model', 'Невідома модель')  # type: ignore
            self._logger.info(
                f"Камера '{self.camera_name}' (ID: {self.camera_id}, Модель: {cam_model_info}) ініціалізується...")

            transform = Transform(hflip=hflip, vflip=vflip)
            config = self.picam2.create_still_configuration(  # type: ignore
                main={"size": self.capture_resolution},
                lores={"size": (640, 480)},
                display="lores",
                transform=transform
            )
            self.picam2.configure(config)  # type: ignore
            self._logger.info(
                f"Налаштовано для фото з роздільною здатністю {self.capture_resolution}, трансформація: hflip={hflip}, vflip={vflip}.")

            self.picam2.start()  # type: ignore
            self._logger.info("Камеру запущено.")

            self._logger.info("Очікування 2-3 секунди для автоналаштування камери...")
            time.sleep(2.5)
            self._logger.info("Камера готова.")
            self.is_initialized_successfully = True

        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати або запустити камеру '{self.camera_name}': {e}",
                               exc_info=True)
            if self.picam2:
                try:
                    self.picam2.close()  # type: ignore
                except Exception as close_e:
                    self._logger.error(
                        f"Помилка закриття камери '{self.camera_name}' під час невдалої ініціалізації: {close_e}")
            self.picam2 = None
            self.is_initialized_successfully = False

    def capture_image(self, filename_only: str = "capture.jpg") -> Optional[str]:
        if not self.is_initialized_successfully or not self.picam2 or not self.picam2.started:  # type: ignore
            self._logger.error(f"Камера '{self.camera_name}' не готова. Неможливо захопити зображення.")
            return None

        full_output_path = os.path.join(self.image_save_path_for_this_camera, filename_only)

        try:
            self._logger.info(f"'{self.camera_name}': Захоплення зображення у файл {full_output_path}...")
            self.picam2.capture_file(full_output_path)  # type: ignore
            self._logger.info(f"'{self.camera_name}': Зображення захоплено у {full_output_path}.")
            return full_output_path
        except Exception as e:
            self._logger.error(f"'{self.camera_name}': Не вдалося захопити зображення у {full_output_path}: {e}",
                               exc_info=True)
            return None

    def capture_array(self, stream_name: str = "main") -> Optional[np.ndarray]:
        if not self.is_initialized_successfully or not self.picam2 or not self.picam2.started:  # type: ignore
            self._logger.error(f"Камера '{self.camera_name}' не готова. Неможливо захопити масив.")
            return None
        try:
            self._logger.debug(f"'{self.camera_name}': Захоплення масиву зі стріму '{stream_name}'...")
            image_array = self.picam2.capture_array(stream_name)  # type: ignore
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
        if not PICAMERA2_AVAILABLE or Preview.DRM is None:  # type: ignore
            self._logger.warning(
                f"'{self.camera_name}': Preview.DRM недоступний (можливо, не на Raspberry Pi з дисплеєм). Прев'ю не запущено.")
            return
        try:
            self.picam2.start_preview(Preview.DRM, x=x, y=y, width=width, height=height)  # type: ignore
            self._logger.info(f"'{self.camera_name}': Прев'ю запущено. Позиція: ({x},{y}), Розмір: ({width}x{height})")
        except Exception as e:
            self._logger.error(f"'{self.camera_name}': Не вдалося запустити прев'ю: {e}. "
                               "Переконайтеся, що графічне середовище доступне.")

    def stop_preview(self):
        if not self.is_initialized_successfully or not self.picam2: return
        try:
            self.picam2.stop_preview()  # type: ignore
            self._logger.info(f"'{self.camera_name}': Прев'ю зупинено.")
        except Exception as e:
            self._logger.debug(f"'{self.camera_name}': Помилка зупинки прев'ю (можливо, не було запущено): {e}")

    def close(self):
        if hasattr(self, 'picam2') and self.picam2:
            try:
                self._logger.debug(f"Закриття камери '{self.camera_name}'...")
                if self.picam2.started:  # type: ignore
                    self.stop_preview()
                    self.picam2.stop()  # type: ignore
                    self._logger.info(f"Камеру '{self.camera_name}' зупинено.")
                self.picam2.close()  # type: ignore
                self._logger.info(f"Ресурси камери '{self.camera_name}' звільнено.")
            except Exception as e:
                self._logger.error(f"Помилка закриття камери '{self.camera_name}': {e}", exc_info=True)
            finally:
                self.picam2 = None
                self.is_initialized_successfully = False

    def __del__(self):
        self.close()


class CameraManager:
    def __init__(self,
                 entry_cam_model_sub: str = DEFAULT_ENTRY_CAM_MODEL_SUBSTRING,
                 exit_cam_model_sub: str = DEFAULT_EXIT_CAM_MODEL_SUBSTRING,
                 entry_cam_config: Optional[Dict[str, Any]] = None,
                 exit_cam_config: Optional[Dict[str, Any]] = None,
                 image_base_path: str = DEFAULT_IMAGE_SAVE_PATH):
        self._logger = logging.getLogger(f"{__name__}.CameraManager")
        self._logger.info("Ініціалізація CameraManager...")

        self.cam_entry: Optional[CameraController] = None
        self.cam_exit: Optional[CameraController] = None

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

        except Exception as e:
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
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s')

    logger.info("Тестування модуля camera_manager.py...")
    if not PICAMERA2_AVAILABLE:
        logger.warning("Бібліотека Picamera2 недоступна. Тестування буде обмеженим (з мок-об'єктами).")

    test_image_base_path = "test_camera_captures"
    os.makedirs(os.path.join(test_image_base_path, "entry_test"), exist_ok=True)
    os.makedirs(os.path.join(test_image_base_path, "exit_test"), exist_ok=True)

    entry_cam_test_config = {
        "name": "TestEntryCam", "resolution": (640, 480),
        "hflip": False, "vflip": False, "type_for_path": "entry_test"  # Змінено vflip для тесту
    }
    exit_cam_test_config = {
        "name": "TestExitCam", "resolution": (640, 480),
        "hflip": False, "vflip": False, "type_for_path": "exit_test"
    }

    cam_manager = CameraManager(
        entry_cam_config=entry_cam_test_config,
        exit_cam_config=exit_cam_test_config,
        image_base_path=test_image_base_path
    )

    entry_cam = cam_manager.get_entry_camera()
    exit_cam = cam_manager.get_exit_camera()
    timestamp_str = time.strftime('%Y%m%d_%H%M%S')

    if entry_cam:
        logger.info(f"--- Тестування камери В'ЇЗДУ: {entry_cam.camera_name} ---")
        entry_file = entry_cam.capture_image(f"test_entry_{timestamp_str}.jpg")
        if entry_file: logger.info(f"Зображення в'їзду збережено: {entry_file}")
        array_entry = entry_cam.capture_array()
        if array_entry is not None: logger.info(f"Масив з камери в'їзду отримано, форма: {array_entry.shape}")

    else:
        logger.warning("Камера в'їзду не доступна для тестування.")

    if exit_cam:
        logger.info(f"--- Тестування камери ВИЇЗДУ: {exit_cam.camera_name} ---")
        exit_file = exit_cam.capture_image(f"test_exit_{timestamp_str}.jpg")
        if exit_file: logger.info(f"Зображення виїзду збережено: {exit_file}")
        array_exit = exit_cam.capture_array()
        if array_exit is not None: logger.info(f"Масив з камери виїзду отримано, форма: {array_exit.shape}")
    else:
        logger.warning("Камера виїзду не доступна для тестування.")

    cam_manager.close_all_cameras()
    logger.info("Тестування модуля camera_manager.py завершено.")