# core/camera_manager.py
import time
import logging
from picamera2 import Picamera2, Preview
from libcamera import Transform
import os

# Налаштування логування для модуля camera_manager
logger = logging.getLogger(__name__)

DEFAULT_ENTRY_CAM_MODEL_SUBSTRING = 'imx708'  # Для Camera Module 3
DEFAULT_EXIT_CAM_MODEL_SUBSTRING = 'imx219'  # Для Camera Module 2

# Шлях для збереження зображень (може бути налаштований)
DEFAULT_IMAGE_SAVE_PATH = "captured_images"


# --- Функція для отримання ID камер ---
def get_camera_ids(entry_cam_model_sub=DEFAULT_ENTRY_CAM_MODEL_SUBSTRING,
                   exit_cam_model_sub=DEFAULT_EXIT_CAM_MODEL_SUBSTRING):
    """
    Знаходить індекси (camera_num) для камери в'їзду та виїзду на основі
    підрядків у назвах їх моделей сенсорів.

    Args:
        entry_cam_model_sub (str): Підрядок для ідентифікації моделі камери в'їзду.
        exit_cam_model_sub (str): Підрядок для ідентифікації моделі камери виїзду.

    Returns:
        dict: Словник з ключами "entry" та "exit" та відповідними індексами камер,
              або None для відповідного ключа, якщо камеру не знайдено.
    """
    camera_ids = {"entry": None, "exit": None}
    try:
        cameras_info = Picamera2.global_camera_info()
        if not cameras_info:
            logger.warning("Picamera2.global_camera_info() не знайшло жодної камери.")
            return camera_ids

        logger.info(f"Виявлені камери: {cameras_info}")

        for i, info in enumerate(cameras_info):
            model = info.get("Model", "").lower()
            cam_num = info.get("Num", i)  # Використовуємо 'Num' якщо є, інакше індекс 'i'

            if entry_cam_model_sub in model and camera_ids["entry"] is None:
                camera_ids["entry"] = cam_num
                logger.info(
                    f"Камера В'ЇЗДУ (підрядок моделі '{entry_cam_model_sub}') знайдена: {info.get('Location', '')} {model} (ID: {cam_num})")
            elif exit_cam_model_sub in model and camera_ids["exit"] is None:
                camera_ids["exit"] = cam_num
                logger.info(
                    f"Камера ВИЇЗДУ (підрядок моделі '{exit_cam_model_sub}') знайдена: {info.get('Location', '')} {model} (ID: {cam_num})")

        if camera_ids["entry"] is None:
            logger.warning(f"Камера В'ЇЗДУ (підрядок моделі '{entry_cam_model_sub}') НЕ знайдена.")
        if camera_ids["exit"] is None:
            logger.warning(f"Камера ВИЇЗДУ (підрядок моделі '{exit_cam_model_sub}') НЕ знайдена.")

    except Exception as e:
        logger.error(f"Помилка під час отримання ID камер: {e}", exc_info=True)
    return camera_ids


# --- Клас для Керування Камерою ---
class CameraController:
    """
    Клас для керування камерою Raspberry Pi за допомогою Picamera2.
    """

    def __init__(self, camera_id, camera_name="UnnamedCamera", capture_resolution=(1920, 1080),
                 hflip=False, vflip=False, image_save_path_prefix=None):
        """
        Ініціалізує конкретну камеру.

        Args:
            camera_id (int): Індекс камери (camera_num).
            camera_name (str): Описова назва камери.
            capture_resolution (tuple): Роздільна здатність для захоплення.
            hflip (bool): Віддзеркалити по горизонталі.
            vflip (bool): Віддзеркалити по вертикалі.
            image_save_path_prefix (str, optional): Префікс шляху для збереження зображень (напр., "entry_cam/").
        """
        self.camera_id = camera_id
        self.camera_name = camera_name
        self.capture_resolution = capture_resolution
        self.picam2 = None
        self._logger = logging.getLogger(f"{__name__}.{self.camera_name}_ID{self.camera_id}")
        self.image_save_path_prefix = image_save_path_prefix or f"{camera_name.lower()}_"
        self.is_initialized_successfully = False

        try:
            self.picam2 = Picamera2(camera_num=self.camera_id)
            cam_model_info = self.picam2.camera_properties.get('Model', 'Невідома модель')
            self._logger.info(
                f"Камера '{self.camera_name}' (ID: {self.camera_id}, Модель: {cam_model_info}) ініціалізовано.")

            transform = Transform(hflip=hflip, vflip=vflip)
            # Використовуємо main stream для фото, lores для прев'ю (якщо використовується)
            config = self.picam2.create_still_configuration(
                main={"size": self.capture_resolution},
                lores={"size": (640, 480)},  # Роздільна здатність для lores stream, якщо потрібне прев'ю
                display="lores",
                transform=transform
            )
            self.picam2.configure(config)
            self._logger.info(
                f"Налаштовано для фото з роздільною здатністю {self.capture_resolution}, трансформація: hflip={hflip}, vflip={vflip}.")

            self.picam2.start()
            self._logger.info("Камеру запущено.")

            self._logger.info("Очікування 2-3 секунди для налаштування камери...")
            time.sleep(3)
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

    def capture_image(self, output_filename="captured_image.jpg"):  # Змінено для передачі повного імені файлу
        """Захоплює зображення та зберігає його у вказаний файл."""
        if not self.picam2 or not self.picam2.started:
            self._logger.error(
                f"Камера '{self.camera_name}' не ініціалізована або не запущена. Неможливо захопити зображення.")
            return None

        # Переконуємося, що директорія для збереження існує
        output_dir = os.path.dirname(output_filename)
        if output_dir:  # Якщо в імені файлу є шлях
            os.makedirs(output_dir, exist_ok=True)

        try:
            self._logger.info(f"'{self.camera_name}': Спроба захопити зображення у файл {output_filename}...")
            metadata = self.picam2.capture_file(output_filename)
            self._logger.info(f"'{self.camera_name}': Зображення захоплено у {output_filename}. Метадані: {metadata}")
            return output_filename
        except Exception as e:
            self._logger.error(f"'{self.camera_name}': Не вдалося захопити зображення у {output_filename}: {e}",
                               exc_info=True)
            return None

    def capture_array(self, array_name="main"):
        """Захоплює зображення як масив numpy."""
        if not self.picam2 or not self.picam2.started:
            self._logger.error(
                f"Камера '{self.camera_name}' не ініціалізована або не запущена. Неможливо захопити масив.")
            return None
        try:
            self._logger.info(f"'{self.camera_name}': Спроба захопити масив зі стріму '{array_name}'...")
            image_array = self.picam2.capture_array(array_name)
            self._logger.info(
                f"'{self.camera_name}': Масив захоплено зі стріму '{array_name}' з формою {image_array.shape}.")
            return image_array
        except Exception as e:
            self._logger.error(f"'{self.camera_name}': Не вдалося захопити масив зі стріму '{array_name}': {e}",
                               exc_info=True)
            return None

    def start_preview(self, x=100, y=100, width=640, height=480):
        if not self.picam2 or not self.is_initialized_successfully:
            self._logger.error(f"Камера '{self.camera_name}' не ініціалізована. Неможливо запустити прев'ю.")
            return
        try:
            # Використовуємо Preview.DRM для Raspberry Pi OS Desktop, або Preview.QTGL/Preview.QT для X11 Forwarding
            self.picam2.start_preview(Preview.DRM, x=x, y=y, width=width, height=height)  #
            self._logger.info(f"'{self.camera_name}': Прев'ю запущено. Позиція: ({x},{y}), Розмір: ({width}x{height})")
        except Exception as e:
            # Може виникнути помилка, якщо немає дисплея (напр. headless система)
            self._logger.error(f"'{self.camera_name}': Не вдалося запустити прев'ю: {e}. "
                               "Якщо ви працюєте в headless режимі, це очікувано.")

    def stop_preview(self):
        if not self.picam2 or not self.is_initialized_successfully: return
        try:
            self.picam2.stop_preview()
            self._logger.info(f"'{self.camera_name}': Прев'ю зупинено.")
        except Exception as e:
            self._logger.error(f"'{self.camera_name}': Помилка зупинки прев'ю: {e}")

    def close(self):  #
        if self.picam2:
            try:
                if self.picam2.started:
                    self.stop_preview()  # Зупиняємо прев'ю перед зупинкою камери
                    self.picam2.stop()
                    self._logger.info(f"Камеру '{self.camera_name}' зупинено.")
                self.picam2.close()
                self._logger.info(f"Ресурси камери '{self.camera_name}' звільнено.")
                self.picam2 = None
                self.is_initialized_successfully = False
            except Exception as e:
                self._logger.error(f"Помилка закриття камери '{self.camera_name}': {e}", exc_info=True)

    def __del__(self):
        self.close()


# --- Клас Менеджера Камер ---
class CameraManager:
    """
    Клас для ініціалізації та керування камерами в'їзду та виїзду.
    """

    def __init__(self,
                 entry_cam_model_sub: str = DEFAULT_ENTRY_CAM_MODEL_SUBSTRING,
                 exit_cam_model_sub: str = DEFAULT_EXIT_CAM_MODEL_SUBSTRING,
                 entry_cam_config: dict = None,  # Словник з параметрами для камери в'їзду
                 exit_cam_config: dict = None,  # Словник з параметрами для камери виїзду
                 image_base_path: str = DEFAULT_IMAGE_SAVE_PATH):
        """
        Ініціалізує CameraManager.

        Args:
            entry_cam_model_sub (str): Підрядок для ідентифікації моделі камери в'їзду.
            exit_cam_model_sub (str): Підрядок для ідентифікації моделі камери виїзду.
            entry_cam_config (dict): Конфігурація для камери в'їзду (name, resolution, hflip, vflip).
            exit_cam_config (dict): Конфігурація для камери виїзду.
            image_base_path (str): Базовий шлях для збереження зображень.
        """
        self._logger = logging.getLogger(f"{__name__}.CameraManager")
        self._logger.info("Ініціалізація CameraManager...")

        self.cam_entry: CameraController = None
        self.cam_exit: CameraController = None
        self.image_base_path = image_base_path
        os.makedirs(self.image_base_path, exist_ok=True)  # Створюємо директорію, якщо її немає

        # Налаштування за замовчуванням, якщо не передано
        default_entry_cfg = {"name": "EntryCam", "resolution": (1920, 1080), "hflip": True, "vflip": False}
        default_exit_cfg = {"name": "ExitCam", "resolution": (1280, 720), "hflip": True, "vflip": False}

        cfg_entry = {**default_entry_cfg, **(entry_cam_config or {})}
        cfg_exit = {**default_exit_cfg, **(exit_cam_config or {})}

        try:
            camera_ids = get_camera_ids(entry_cam_model_sub, exit_cam_model_sub)

            if camera_ids["entry"] is not None:
                self.cam_entry = CameraController(
                    camera_id=camera_ids["entry"],
                    camera_name=cfg_entry["name"],
                    capture_resolution=cfg_entry["resolution"],
                    hflip=cfg_entry["hflip"],
                    vflip=cfg_entry["vflip"],
                    image_save_path_prefix=os.path.join(self.image_base_path, "entry") + os.sep
                    # Для збереження у entry/
                )
                if not self.cam_entry.is_initialized_successfully:
                    self.cam_entry = None  # Якщо ініціалізація не вдалася
            else:
                self._logger.warning(
                    f"Камера В'ЇЗДУ (модель '{entry_cam_model_sub}') не знайдена або не вдалося отримати її ID.")

            if camera_ids["exit"] is not None:
                self.cam_exit = CameraController(
                    camera_id=camera_ids["exit"],
                    camera_name=cfg_exit["name"],
                    capture_resolution=cfg_exit["resolution"],
                    hflip=cfg_exit["hflip"],
                    vflip=cfg_exit["vflip"],
                    image_save_path_prefix=os.path.join(self.image_base_path, "exit") + os.sep  # Для збереження у exit/
                )
                if not self.cam_exit.is_initialized_successfully:
                    self.cam_exit = None  # Якщо ініціалізація не вдалася
            else:
                self._logger.warning(
                    f"Камера ВИЇЗДУ (модель '{exit_cam_model_sub}') не знайдена або не вдалося отримати її ID.")

        except Exception as e:
            self._logger.error(f"Помилка під час ініціалізації камер у CameraManager: {e}", exc_info=True)

        self._logger.info("CameraManager ініціалізацію завершено.")

    def get_entry_camera(self) -> CameraController | None:
        """Повертає екземпляр камери в'їзду."""
        return self.cam_entry

    def get_exit_camera(self) -> CameraController | None:
        """Повертає екземпляр камери виїзду."""
        return self.cam_exit

    def close_all_cameras(self):
        """Закриває всі керовані камери."""
        self._logger.info("Закриття всіх камер через CameraManager...")
        if self.cam_entry:
            self.cam_entry.close()
        if self.cam_exit:
            self.cam_exit.close()
        self._logger.info("Всі камери закрито.")

    def __del__(self):
        self.close_all_cameras()


# --- Приклад використання (для тестування модуля окремо) ---
if __name__ == '__main__':
    # Налаштування базового логування для тестування
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s')

    logger.info("Тестування модуля camera_manager.py...")

    # Створюємо директорії для тестових зображень, якщо їх немає
    os.makedirs(os.path.join(DEFAULT_IMAGE_SAVE_PATH, "entry"), exist_ok=True)
    os.makedirs(os.path.join(DEFAULT_IMAGE_SAVE_PATH, "exit"), exist_ok=True)

    # Приклад конфігурацій для камер
    entry_camera_settings = {
        "name": "TestEntryCam",
        "resolution": (640, 480),  # Нижча роздільна здатність для швидкого тесту
        "hflip": True,
        "vflip": False
    }
    exit_camera_settings = {
        "name": "TestExitCam",
        "resolution": (640, 480),
        "hflip": True,
        "vflip": True
    }

    try:
        # Ініціалізуємо менеджер камер
        # За замовчуванням він шукатиме моделі 'imx708' та 'imx219'
        cam_manager = CameraManager(
            entry_cam_config=entry_camera_settings,
            exit_cam_config=exit_camera_settings,
            image_base_path="test_captures"  # Окрема папка для тестових знімків
        )

        entry_cam = cam_manager.get_entry_camera()
        exit_cam = cam_manager.get_exit_camera()

        if entry_cam:
            logger.info(f"\n--- Тестування камери В'ЇЗДУ: {entry_cam.camera_name} ---")
            # entry_cam.start_preview(x=50, y=50, width=320, height=240)
            # time.sleep(2)
            # Формуємо ім'я файлу з використанням префіксу шляху з CameraController
            entry_image_filename = f"{entry_cam.image_save_path_prefix}test_capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"

            captured_entry = entry_cam.capture_image(output_filename=entry_image_filename)  #
            if captured_entry:
                logger.info(f"Камера В'ЇЗДУ: Зображення збережено як {captured_entry}")
            else:
                logger.error("Камера В'ЇЗДУ: Не вдалося захопити зображення.")
            # entry_cam.stop_preview()
        else:
            logger.warning("Камера В'ЇЗДУ не доступна для тестування.")

        if exit_cam:
            logger.info(f"\n--- Тестування камери ВИЇЗДУ: {exit_cam.camera_name} ---")
            # exit_cam.start_preview(x=400, y=50, width=320, height=240)
            # time.sleep(2)
            exit_image_filename = f"{exit_cam.image_save_path_prefix}test_capture_{time.strftime('%Y%m%d_%H%M%S')}.jpg"

            captured_exit = exit_cam.capture_image(output_filename=exit_image_filename)  #
            if captured_exit:
                logger.info(f"Камера ВИЇЗДУ: Зображення збережено як {captured_exit}")
            else:
                logger.error("Камера ВИЇЗДУ: Не вдалося захопити зображення.")
            # exit_cam.stop_preview()
        else:
            logger.warning("Камера ВИЇЗДУ не доступна для тестування.")

        # Закриваємо камери через менеджер
        cam_manager.close_all_cameras()

    except Exception as e:
        logger.error(f"Помилка під час тестування CameraManager: {e}", exc_info=True)
        # Спробувати очистити ресурси, якщо менеджер був частково створений
        if 'cam_manager' in locals() and cam_manager:
            cam_manager.close_all_cameras()

    logger.info("\nТестування модуля camera_manager.py завершено.")
