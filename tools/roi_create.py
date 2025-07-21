# tools/roi_create.py
import cv2
import numpy as np
import json
import os
import sys
import time

# Додаємо шлях до кореневої директорії проекту
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
        except Exception:
            active_rois = {}
    
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
        
        final_x1, final_y1 = min(x1, x2), min(y1, y2)
        final_x2, final_y2 = max(x1, x2), max(y1, y2)
        
        roi_key = f"{active_camera_type}_camera_roi"
        active_rois[roi_key] = {"x1": final_x1, "y1": final_y1, "x2": final_x2, "y2": final_y2, "enabled": True}
        print(f"ROI для '{active_camera_type}' встановлено. Натисніть 'q' для збереження та виходу.")
        roi_points = []

def main():
    global current_frame_display, active_rois

    print("Запуск інструменту створення ROI...")
    load_roi_config()

    print("Ініціалізація камер...")
    cam_manager = CameraManager(
        entry_cam_config={"name": "ROICreateEntry"},
        exit_cam_config={"name": "ROICreateExit"}
    )
    entry_cam = cam_manager.get_entry_camera()
    exit_cam = cam_manager.get_exit_camera()

    if not entry_cam and not exit_cam:
        print("Жодна камера не доступна. Завершення роботи.")
        return

    # Створюємо вікно одразу
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_AUTOSIZE)

    active_camera_obj = None
    active_camera_type = None
    
    print("\n--- Керування ---")
    print("'e': Камера В'їзду (Entry)")
    print("'x': Камера Виїзду (Exit)")
    print("'c': Очистити ROI для активної камери")
    print("'t': Увімкнути/Вимкнути ROI для активної камери")
    print("'q': Вийти та ЗБЕРЕГТИ")
    print("--------------------")

    while True:
        # 1. Готуємо кадр для відображення
        if active_camera_obj and active_camera_obj.is_initialized_successfully:
            frame = active_camera_obj.capture_array()
            if frame is None:
                frame = np.zeros((480, 640, 3), dtype=np.uint8)
                cv2.putText(frame, "NO SIGNAL", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
            current_frame_display = frame.copy()

            # Малюємо збережений ROI
            roi_key = f"{active_camera_type}_camera_roi"
            if roi_key in active_rois and active_rois[roi_key]['x2'] > 0:
                roi = active_rois[roi_key]
                color = (0, 255, 255) if roi.get("enabled", False) else (100, 100, 100)
                status = "ON" if roi.get("enabled", False) else "OFF"
                cv2.rectangle(current_frame_display, (roi['x1'], roi['y1']), (roi['x2'], roi['y2']), color, 2)
                draw_text_with_background(current_frame_display, f"ROI: {status}", (roi['x1'], roi['y1']-5), bg_color=color)

        else:
            current_frame_display = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(current_frame_display, "Виберiть камеру: 'e' (Entry) або 'x' (Exit)", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)

        # 2. Показуємо кадр
        cv2.imshow(WINDOW_NAME, current_frame_display)

        # 3. Чекаємо на дію користувача
        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            save_roi_config()
            break
        elif key == ord('e'):
            print("Обрано камеру В'їзду (Entry).")
            active_camera_obj, active_camera_type = entry_cam, "entry"
            cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=active_camera_type)
        elif key == ord('x'):
            print("Обрано камеру Виїзду (Exit).")
            active_camera_obj, active_camera_type = exit_cam, "exit"
            cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=active_camera_type)
        elif key == ord('c'):
            if active_camera_type:
                roi_key = f"{active_camera_type}_camera_roi"
                active_rois[roi_key] = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "enabled": False}
                print(f"ROI для '{active_camera_type}' очищено.")
        elif key == ord('t'):
            if active_camera_type:
                roi_key = f"{active_camera_type}_camera_roi"
                current_status = active_rois[roi_key].get("enabled", False)
                active_rois[roi_key]["enabled"] = not current_status
                status_text = "УВІМКНЕНО" if not current_status else "ВИМКНЕНО"
                print(f"ROI для '{active_camera_type}' тепер {status_text}.")
    
    cam_manager.close_all_cameras()
    cv2.destroyAllWindows()
    print("Роботу завершено.")

# Допоміжна функція для малювання тексту з фоном
def draw_text_with_background(image, text, origin, font_face=cv2.FONT_HERSHEY_SIMPLEX, font_scale=0.5, text_color=(0,0,0), bg_color=(255,255,255), thickness=1):
    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)
    bg_rect_pt1 = (origin[0], origin[1] - text_height - baseline)
    bg_rect_pt2 = (origin[0] + text_width, origin[1] + baseline)
    cv2.rectangle(image, bg_rect_pt1, bg_rect_pt2, bg_color, cv2.FILLED)
    cv2.putText(image, text, origin, font_face, font_scale, text_color, thickness, cv2.LINE_AA)

if __name__ == "__main__":
    main()
