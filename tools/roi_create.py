# tools/roi_create.py
import cv2
import numpy as np
import json
import os
import sys
import time

# Додаємо шлях до кореневої директорії проекту, щоб імпортувати core модулі
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from core.camera_manager import CameraManager, \
        CameraController  # Припускаємо, що CameraController теж потрібен для типу
except ImportError as e:
    print(f"Помилка імпорту camera_manager: {e}")
    print(
        f"Переконайтеся, що core.camera_manager доступний і PYTHONPATH налаштовано, або запустіть з кореня проекту: python tools/{os.path.basename(__file__)}")
    sys.exit(1)

# --- Глобальні налаштування ---
ROI_CONFIG_FILE = os.path.join(project_root, "config", "roi_config.json")
WINDOW_NAME = "ROI Creator - Натисніть 'h' для допомоги"

# Змінні для малювання ROI
drawing = False
roi_points = []  # Буде зберігати [(x1, y1), (x2, y2)]
current_frame_display = None  # Кадр для відображення з намальованим ROI

# Словник для зберігання поточних налаштувань ROI
# Структура: {"entry_camera_roi": {"x1":0, "y1":0, "x2":0, "y2":0, "enabled": False}, ...}
active_rois = {}


def load_roi_config():
    """Завантажує існуючу конфігурацію ROI, якщо файл існує."""
    global active_rois
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, 'r') as f:
                active_rois = json.load(f)
            print(f"Завантажено існуючу конфігурацію ROI з: {ROI_CONFIG_FILE}")
        except json.JSONDecodeError:
            print(f"Помилка декодування JSON у файлі: {ROI_CONFIG_FILE}. Буде створено новий файл.")
            active_rois = {}
        except Exception as e:
            print(f"Не вдалося завантажити {ROI_CONFIG_FILE}: {e}. Буде створено новий файл.")
            active_rois = {}
    else:
        print(f"Файл {ROI_CONFIG_FILE} не знайдено. Буде створено новий при збереженні.")
        active_rois = {}

    # Переконуємося, що ключі для обох камер існують з значеннями за замовчуванням
    if "entry_camera_roi" not in active_rois:
        active_rois["entry_camera_roi"] = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "enabled": False}
    if "exit_camera_roi" not in active_rois:
        active_rois["exit_camera_roi"] = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "enabled": False}


def save_roi_config():
    """Зберігає поточну конфігурацію ROI у файл."""
    global active_rois
    try:
        os.makedirs(os.path.dirname(ROI_CONFIG_FILE), exist_ok=True)
        with open(ROI_CONFIG_FILE, 'w') as f:
            json.dump(active_rois, f, indent=2)
        print(f"Конфігурацію ROI збережено у: {ROI_CONFIG_FILE}")
    except Exception as e:
        print(f"Помилка збереження конфігурації ROI: {e}")


def mouse_callback(event, x, y, flags, param):
    """Обробник подій миші для малювання прямокутника."""
    global roi_points, drawing, current_frame_display

    active_camera_type = param  # Передаємо тип активної камери

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_points = [(x, y)]  # Початкова точка
        print(f"Початок малювання ROI: {roi_points[0]}")

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Малюємо тимчасовий прямокутник для візуалізації
            temp_frame = current_frame_display.copy()
            cv2.rectangle(temp_frame, roi_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Завершальна точка та фіксація ROI
        # Переконуємося, що x1 < x2 та y1 < y2
        x1, y1 = roi_points[0]
        x2, y2 = x, y

        final_x1 = min(x1, x2)
        final_y1 = min(y1, y2)
        final_x2 = max(x1, x2)
        final_y2 = max(y1, y2)

        # Оновлюємо roi_points фінальними координатами
        roi_points = [(final_x1, final_y1), (final_x2, final_y2)]

        # Застосовуємо та зберігаємо цей ROI для активної камери
        if active_camera_type:
            roi_key = f"{active_camera_type}_camera_roi"
            active_rois[roi_key]["x1"] = roi_points[0][0]
            active_rois[roi_key]["y1"] = roi_points[0][1]
            active_rois[roi_key]["x2"] = roi_points[1][0]
            active_rois[roi_key]["y2"] = roi_points[1][1]
            # За замовчуванням увімкнено при новому малюванні
            # active_rois[roi_key]["enabled"] = True
            print(f"ROI для '{active_camera_type}' встановлено: {roi_points}. Натисніть 's' для збереження у файл.")

            # Оновлюємо відображення з фінальним прямокутником
            current_frame_display_with_final_roi = current_frame_display.copy()
            cv2.rectangle(current_frame_display_with_final_roi, roi_points[0], roi_points[1], (0, 0, 255),
                          2)  # Червоний для фінального
            cv2.imshow(WINDOW_NAME, current_frame_display_with_final_roi)


def display_help(frame_shape):
    """Відображає інструкції на екрані."""
    help_text_lines = [
        "'e': Камера В'їзду (Entry)",
        "'x': Камера Виїзду (Exit)",
        "'r': Оновити кадр з камери",
        "Миша: Намалювати ROI (після вибору камери)",
        "'s': Зберегти поточний ROI для активної камери (у пам'ять)",
        "'c': Очистити/скинути поточний ROI для активної камери",
        "'t': Увімк./Вимк. ROI для активної камери",
        "'h': Показати/сховати цю допомогу",
        "'q': Вийти та ЗБЕРЕГТИ ВСІ ROI у файл"
    ]

    overlay = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
    y0, dy = 30, 25
    for i, line in enumerate(help_text_lines):
        y = y0 + i * dy
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    return overlay


def main():
    global roi_points, current_frame_display, active_rois

    print("Запуск інструменту створення ROI...")
    load_roi_config()

    # Ініціалізація менеджера камер
    # Передаємо None для конфігурацій камер, щоб використовувалися значення за замовчуванням
    # або визначені всередині CameraManager/CameraController.
    # Для цього інструменту нам потрібні лише базові кадри.
    cam_manager = CameraManager(
        entry_cam_config={"name": "ROICreateEntry"},
        exit_cam_config={"name": "ROICreateExit"}
    )

    entry_cam_controller = cam_manager.get_entry_camera()
    exit_cam_controller = cam_manager.get_exit_camera()

    if not entry_cam_controller and not exit_cam_controller:
        print("Жодна камера не доступна. Завершення роботи.")
        return

    active_camera_obj: CameraController = None
    active_camera_type_str = None  # "entry" or "exit"
    current_frame_orig = None  # Оригінальний кадр з камери

    show_help = True  # Показувати допомогу спочатку

    cv2.namedWindow(WINDOW_NAME)
    # Передаємо active_camera_type_str як параметр, щоб mouse_callback знав, для якої камери ROI
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=active_camera_type_str)

    print("\n--- Керування ---")
    for line in display_help((480, 640))[1]:  # Отримуємо текст для виводу в консоль
        print(line)
    print("\nСпочатку виберіть камеру ('e' або 'x').")

    while True:
        if active_camera_obj and active_camera_obj.is_initialized_successfully:
            if current_frame_orig is None:  # Захоплюємо кадр, якщо його немає
                current_frame_orig = active_camera_obj.capture_array()
                if current_frame_orig is None:
                    print(f"Не вдалося отримати кадр з камери {active_camera_obj.camera_name}. Спробуйте 'r'.")
                    # Створюємо чорний кадр, щоб програма не впала
                    current_frame_orig = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(current_frame_orig, "NO SIGNAL", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            current_frame_display = current_frame_orig.copy()

            # Відображення поточного активного ROI для вибраної камери
            if active_camera_type_str:
                roi_key = f"{active_camera_type_str}_camera_roi"
                current_saved_roi = active_rois.get(roi_key)
                if current_saved_roi and current_saved_roi["x2"] > current_saved_roi[
                    "x1"]:  # Перевірка, чи ROI валідний
                    pt1 = (current_saved_roi["x1"], current_saved_roi["y1"])
                    pt2 = (current_saved_roi["x2"], current_saved_roi["y2"])
                    color = (0, 255, 255) if current_saved_roi.get("enabled", False) else (100, 100, 100)
                    thickness = 2 if current_saved_roi.get("enabled", False) else 1
                    cv2.rectangle(current_frame_display, pt1, pt2, color, thickness)
                    status_text = "ENABLED" if current_saved_roi.get("enabled", False) else "DISABLED"
                    cv2.putText(current_frame_display, f"ROI: {status_text}", (pt1[0], pt1[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            # Відображення поточного процесу малювання
            if drawing and len(roi_points) == 1:  # Якщо почали малювати, але ще не відпустили кнопку
                # Це малювання буде перекрито mouse_move, але залишмо для ясності
                pass  # mouse_move малює тимчасовий прямокутник
            elif not drawing and len(roi_points) == 2:  # Якщо завершили малювати (але ще не зберегли через 's')
                # Це для візуалізації останнього намальованого ROI, який може бути збережений
                cv2.rectangle(current_frame_display, roi_points[0], roi_points[1], (0, 0, 255),
                              2)  # Червоний для поточного вибору

            if show_help:
                help_overlay = display_help(current_frame_display.shape)
                # Змішуємо зображення з допомогою для напівпрозорості
                alpha = 0.6
                cv2.addWeighted(help_overlay, alpha, current_frame_display, 1 - alpha, 0, current_frame_display)

            cv2.imshow(WINDOW_NAME, current_frame_display)

        else:  # Якщо камера не вибрана або не ініціалізована
            # Показуємо порожнє вікно з інструкцією вибрати камеру
            placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder_frame, "Виберiть камеру: 'e' (Entry) або 'x' (Exit)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            if show_help:
                help_overlay = display_help(placeholder_frame.shape)
                alpha = 0.6
                cv2.addWeighted(help_overlay, alpha, placeholder_frame, 1 - alpha, 0, placeholder_frame)
            cv2.imshow(WINDOW_NAME, placeholder_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("Збереження конфігурації та вихід...")
            save_roi_config()
            break
        elif key == ord('e'):
            print("Обрано камеру В'їзду (Entry).")
            active_camera_obj = entry_cam_controller
            active_camera_type_str = "entry"
            current_frame_orig = None  # Скинути поточний кадр, щоб захопити новий
            roi_points = []  # Скинути поточне малювання
            cv2.setMouseCallback(WINDOW_NAME, mouse_callback,
                                 param=active_camera_type_str)  # Оновлюємо параметр для callback
            if not active_camera_obj or not active_camera_obj.is_initialized_successfully:
                print("ПОМИЛКА: Камера В'їзду недоступна!")
        elif key == ord('x'):
            print("Обрано камеру Виїзду (Exit).")
            active_camera_obj = exit_cam_controller
            active_camera_type_str = "exit"
            current_frame_orig = None
            roi_points = []
            cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=active_camera_type_str)
            if not active_camera_obj or not active_camera_obj.is_initialized_successfully:
                print("ПОМИЛКА: Камера Виїзду недоступна!")
        elif key == ord('r'):
            if active_camera_obj:
                print("Оновлення кадру...")
                current_frame_orig = None  # Сигналізує про необхідність захоплення нового кадру
                roi_points = []
            else:
                print("Спочатку виберіть камеру ('e' або 'x').")
        elif key == ord('s'):
            if active_camera_type_str and len(roi_points) == 2:
                roi_key = f"{active_camera_type_str}_camera_roi"
                active_rois[roi_key]["x1"] = roi_points[0][0]
                active_rois[roi_key]["y1"] = roi_points[0][1]
                active_rois[roi_key]["x2"] = roi_points[1][0]
                active_rois[roi_key]["y2"] = roi_points[1][1]
                # При збереженні ROI, якщо він ще не був увімкнений, увімкнемо його.
                # Якщо користувач хоче його вимкнути, він може натиснути 't'.
                if not active_rois[roi_key].get("enabled", False):  # Якщо був False або відсутній
                    active_rois[roi_key]["enabled"] = True
                print(
                    f"Поточний намальований ROI для '{active_camera_type_str}' збережено в пам'яті: {active_rois[roi_key]}")
                roi_points = []  # Готові до нового малювання
            elif not active_camera_type_str:
                print("Спочатку виберіть камеру ('e' або 'x').")
            else:
                print("ROI не намальовано повністю. Використовуйте мишу, щоб намалювати прямокутник.")
        elif key == ord('c'):
            if active_camera_type_str:
                roi_key = f"{active_camera_type_str}_camera_roi"
                active_rois[roi_key] = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "enabled": False}
                roi_points = []
                print(f"ROI для '{active_camera_type_str}' очищено та вимкнено.")
            else:
                print("Спочатку виберіть камеру ('e' або 'x').")
        elif key == ord('t'):  # Toggle enabled
            if active_camera_type_str:
                roi_key = f"{active_camera_type_str}_camera_roi"
                if roi_key in active_rois:
                    active_rois[roi_key]["enabled"] = not active_rois[roi_key].get("enabled", False)
                    status = "УВІМКНЕНО" if active_rois[roi_key]["enabled"] else "ВИМКНЕНО"
                    print(f"ROI для '{active_camera_type_str}' тепер {status}.")
                else:
                    print(
                        f"Для камери '{active_camera_type_str}' ще не визначено ROI. Намалюйте та збережіть ('s') спочатку.")
            else:
                print("Спочатку виберіть камеру ('e' або 'x').")
        elif key == ord('h'):
            show_help = not show_help
            print(f"Показ допомоги: {'Увімкнено' if show_help else 'Вимкнено'}")

    # Звільнення ресурсів
    if entry_cam_controller:
        entry_cam_controller.close()
    if exit_cam_controller:
        exit_cam_controller.close()
    cv2.destroyAllWindows()
    print("Роботу завершено.")


if __name__ == "__main__":
    main()