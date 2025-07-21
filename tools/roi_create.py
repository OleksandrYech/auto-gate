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
    from core.camera_manager import CameraManager, CameraController
except ImportError as e:
    print(f"Помилка імпорту camera_manager: {e}")
    sys.exit(1)

# --- Глобальні налаштування ---
ROI_CONFIG_FILE = os.path.join(project_root, "config", "roi_config.json")
WINDOW_NAME = "ROI Creator - Натисніть 'h' для допомоги"

# Глобальні змінні
drawing = False
roi_points = []
current_frame_display = None
active_rois = {}


def load_roi_config():
    """Завантажує існуючу конфігурацію ROI."""
    global active_rois
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, 'r') as f:
                active_rois = json.load(f)
            print(f"Завантажено існуючу конфігурацію ROI з: {ROI_CONFIG_FILE}")
        except Exception as e:
            print(f"Не вдалося завантажити {ROI_CONFIG_FILE}: {e}. Буде створено новий файл.")
            active_rois = {}
    else:
        print(f"Файл {ROI_CONFIG_FILE} не знайдено. Буде створено новий при збереженні.")
    
    # Гарантуємо наявність ключів
    active_rois.setdefault("entry_camera_roi", {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "enabled": False})
    active_rois.setdefault("exit_camera_roi", {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "enabled": False})


def save_roi_config():
    """Зберігає поточну конфігурацію ROI."""
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
    global roi_points, drawing, current_frame_display, active_rois

    active_camera_type = param
    if not active_camera_type: return

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_points = [(x, y)]

    elif event == cv2.EVENT_MOUSEMOVE and drawing:
        temp_frame = current_frame_display.copy()
        cv2.rectangle(temp_frame, roi_points[0], (x, y), (0, 255, 0), 2)
        cv2.imshow(WINDOW_NAME, temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        x1, y1 = roi_points[0]
        x2, y2 = x, y
        
        # Нормалізуємо координати
        final_x1, final_y1 = min(x1, x2), min(y1, y2)
        final_x2, final_y2 = max(x1, x2), max(y1, y2)
        
        roi_key = f"{active_camera_type}_camera_roi"
        active_rois[roi_key] = {
            "x1": final_x1, "y1": final_y1,
            "x2": final_x2, "y2": final_y2,
            "enabled": True  # Автоматично вмикаємо новий ROI
        }
        print(f"ROI для '{active_camera_type}' встановлено. Натисніть 'q' для збереження та виходу.")
        roi_points = [] # Скидаємо точки для наступного малювання


def display_help(frame_shape):
    # ... (код без змін)
    return overlay


def main():
    global roi_points, current_frame_display, active_rois

    print("Запуск інструменту створення ROI...")
    load_roi_config()

    cam_manager = CameraManager(
        entry_cam_config={"name": "ROICreateEntry"},
        exit_cam_config={"name": "ROICreateExit"}
    )
    entry_cam = cam_manager.get_entry_camera()
    exit_cam = cam_manager.get_exit_camera()

    if not entry_cam and not exit_cam:
        print("Жодна камера не доступна. Завершення роботи.")
        return

    # --- ВИПРАВЛЕННЯ ПОЧИНАЄТЬСЯ ТУТ ---
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)
    mouse_callback_is_set = False # Прапор, щоб налаштувати мишу лише один раз
    # ------------------------------------

    active_camera_obj = None
    active_camera_type = None
    current_frame_orig = None
    show_help = True

    while True:
        if active_camera_obj:
            if current_frame_orig is None:
                current_frame_orig = active_camera_obj.capture_array()
                if current_frame_orig is None:
                    current_frame_orig = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(current_frame_orig, "NO SIGNAL", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            
            current_frame_display = current_frame_orig.copy()
            
            # Малюємо збережений ROI
            roi_key = f"{active_camera_type}_camera_roi"
            if roi_key in active_rois and active_rois[roi_key]['x2'] > 0:
                roi = active_rois[roi_key]
                color = (0, 255, 255) if roi.get("enabled", False) else (100, 100, 100)
                cv2.rectangle(current_frame_display, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), color, 2)
        else:
            current_frame_display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(current_frame_display, "Виберiть камеру: 'e' (Entry) або 'x' (Exit)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        if show_help:
            # ... (логіка відображення допомоги без змін)
            pass

        cv2.imshow(WINDOW_NAME, current_frame_display)

        # --- ВИПРАВЛЕННЯ: налаштовуємо мишу ПІСЛЯ першого показу вікна ---
        if not mouse_callback_is_set:
            cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=active_camera_type)
            mouse_callback_is_set = True
        # -----------------------------------------------------------------

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("Збереження конфігурації та вихід...")
            save_roi_config()
            break
        elif key == ord('e'):
            print("Обрано камеру В'їзду (Entry).")
            active_camera_obj, active_camera_type = entry_cam, "entry"
            current_frame_orig = None
            cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=active_camera_type) # Оновлюємо параметр
        elif key == ord('x'):
            print("Обрано камеру Виїзду (Exit).")
            active_camera_obj, active_camera_type = exit_cam, "exit"
            current_frame_orig = None
            cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=active_camera_type) # Оновлюємо параметр
        
        # ... (решта обробників клавіш без змін)

    cam_manager.close_all_cameras()
    cv2.destroyAllWindows()
    print("Роботу завершено.")

if __name__ == "__main__":
    main()
