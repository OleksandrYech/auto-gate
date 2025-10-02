# tools/roi_create.py
import cv2
import numpy as np
import json
import os
import sys

# Add the project root directory to the path to import core modules
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))
if project_root not in sys.path:
    sys.path.append(project_root)

try:
    from core.camera_manager import CameraManager, CameraController
except ImportError as e:
    print(f"Error importing camera_manager: {e}")
    print(
        f"Make sure core.camera_manager is available and PYTHONPATH is configured, or run from the project root: python tools/{os.path.basename(__file__)}")
    sys.exit(1)

# --- Global Settings ---
ROI_CONFIG_FILE = os.path.join(project_root, "config", "roi_config.json")
WINDOW_NAME = "ROI Creator - Press 'h' for help"

# Variables for drawing ROI
drawing = False
roi_points = []  # Will store [(x1, y1), (x2, y2)]
current_frame_display = None  # Frame for display with the drawn ROI

# Dictionary to store current ROI settings
active_rois = {}


def load_roi_config():
    """Loads an existing ROI configuration if the file exists."""
    global active_rois
    if os.path.exists(ROI_CONFIG_FILE):
        try:
            with open(ROI_CONFIG_FILE, 'r') as f:
                active_rois = json.load(f)
            print(f"Loaded existing ROI configuration from: {ROI_CONFIG_FILE}")
        except json.JSONDecodeError:
            print(f"Error decoding JSON in file: {ROI_CONFIG_FILE}. A new file will be created.")
            active_rois = {}
        except Exception as e:
            print(f"Failed to load {ROI_CONFIG_FILE}: {e}. A new file will be created.")
            active_rois = {}
    else:
        print(f"File {ROI_CONFIG_FILE} not found. A new one will be created upon saving.")
        active_rois = {}

    # Ensure keys for both cameras exist with default values
    if "entry_camera_roi" not in active_rois:
        active_rois["entry_camera_roi"] = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "enabled": False}
    if "exit_camera_roi" not in active_rois:
        active_rois["exit_camera_roi"] = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "enabled": False}


def save_roi_config():
    """Saves the current ROI configuration to a file."""
    global active_rois
    try:
        os.makedirs(os.path.dirname(ROI_CONFIG_FILE), exist_ok=True)
        with open(ROI_CONFIG_FILE, 'w') as f:
            json.dump(active_rois, f, indent=2)
        print(f"ROI configuration saved to: {ROI_CONFIG_FILE}")
    except Exception as e:
        print(f"Error saving ROI configuration: {e}")


def mouse_callback(event, x, y, flags, param):
    """Mouse event handler for drawing a rectangle."""
    global roi_points, drawing, current_frame_display

    active_camera_type = param  # Pass the active camera type

    if event == cv2.EVENT_LBUTTONDOWN:
        drawing = True
        roi_points = [(x, y)]  # Starting point
        print(f"Start drawing ROI at: {roi_points[0]}")

    elif event == cv2.EVENT_MOUSEMOVE:
        if drawing:
            # Draw a temporary rectangle for visualization
            temp_frame = current_frame_display.copy()
            cv2.rectangle(temp_frame, roi_points[0], (x, y), (0, 255, 0), 2)
            cv2.imshow(WINDOW_NAME, temp_frame)

    elif event == cv2.EVENT_LBUTTONUP:
        drawing = False
        # Final point and fixing the ROI
        # Ensure that x1 < x2 and y1 < y2
        x1, y1 = roi_points[0]
        x2, y2 = x, y

        final_x1 = min(x1, x2)
        final_y1 = min(y1, y2)
        final_x2 = max(x1, x2)
        final_y2 = max(y1, y2)

        # Update roi_points with final coordinates
        roi_points = [(final_x1, final_y1), (final_x2, final_y2)]

        # Apply and save this ROI for the active camera
        if active_camera_type:
            roi_key = f"{active_camera_type}_camera_roi"
            active_rois[roi_key]["x1"] = roi_points[0][0]
            active_rois[roi_key]["y1"] = roi_points[0][1]
            active_rois[roi_key]["x2"] = roi_points[1][0]
            active_rois[roi_key]["y2"] = roi_points[1][1]
            # By default, enabled when newly drawn
            if not active_rois[roi_key].get("enabled"):
                 active_rois[roi_key]["enabled"] = True
            print(f"ROI for '{active_camera_type}' set: {roi_points}. Press 's' to save to memory.")

            # Update the display with the final rectangle
            current_frame_display_with_final_roi = current_frame_display.copy()
            cv2.rectangle(current_frame_display_with_final_roi, roi_points[0], roi_points[1], (0, 0, 255),
                          2)  # Red for the final one
            cv2.imshow(WINDOW_NAME, current_frame_display_with_final_roi)


def display_help(frame_shape):
    """Displays instructions on the screen."""
    help_text_lines = [
        "'e': Entry Camera",
        "'x': Exit Camera",
        "'r': Refresh frame from camera",
        "Mouse: Draw ROI (after selecting a camera)",
        "'s': Save current ROI for active camera (to memory)",
        "'c': Clear/reset current ROI for active camera",
        "'t': Toggle ROI for active camera",
        "'h': Show/hide this help",
        "'q': Exit and SAVE ALL ROIs to file"
    ]

    overlay = np.zeros((frame_shape[0], frame_shape[1], 3), dtype=np.uint8)
    y0, dy = 30, 25
    for i, line in enumerate(help_text_lines):
        y = y0 + i * dy
        cv2.putText(overlay, line, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 1, cv2.LINE_AA)

    return overlay


def main():
    global roi_points, current_frame_display, active_rois

    print("Starting ROI creation tool...")
    load_roi_config()

    # Initialize the camera manager
    cam_manager = CameraManager(
        entry_cam_config={"name": "ROICreateEntry"},
        exit_cam_config={"name": "ROICreateExit"}
    )

    entry_cam_controller = cam_manager.get_entry_camera()
    exit_cam_controller = cam_manager.get_exit_camera()

    if not entry_cam_controller and not exit_cam_controller:
        print("No camera available. Exiting.")
        return

    active_camera_obj: Optional[CameraController] = None
    active_camera_type_str = None  # "entry" or "exit"
    current_frame_orig = None  # Original frame from the camera

    show_help = True  # Show help initially

    cv2.namedWindow(WINDOW_NAME)
    cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=active_camera_type_str)

    print("\n--- Controls ---")
    print("First, select a camera ('e' or 'x').")

    while True:
        if active_camera_obj and active_camera_obj.is_initialized_successfully:
            if current_frame_orig is None:  # Capture a frame if there isn't one
                current_frame_orig = active_camera_obj.capture_array()
                if current_frame_orig is None:
                    print(f"Failed to get frame from camera {active_camera_obj.camera_name}. Try 'r'.")
                    current_frame_orig = np.zeros((480, 640, 3), dtype=np.uint8)
                    cv2.putText(current_frame_orig, "NO SIGNAL", (50, 240), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

            current_frame_display = current_frame_orig.copy()

            # Display the current active ROI for the selected camera
            if active_camera_type_str:
                roi_key = f"{active_camera_type_str}_camera_roi"
                current_saved_roi = active_rois.get(roi_key)
                if current_saved_roi and current_saved_roi["x2"] > current_saved_roi["x1"]:
                    pt1 = (current_saved_roi["x1"], current_saved_roi["y1"])
                    pt2 = (current_saved_roi["x2"], current_saved_roi["y2"])
                    color = (0, 255, 255) if current_saved_roi.get("enabled", False) else (100, 100, 100)
                    thickness = 2 if current_saved_roi.get("enabled", False) else 1
                    cv2.rectangle(current_frame_display, pt1, pt2, color, thickness)
                    status_text = "ENABLED" if current_saved_roi.get("enabled", False) else "DISABLED"
                    cv2.putText(current_frame_display, f"ROI: {status_text}", (pt1[0], pt1[1] - 10),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            if show_help:
                help_overlay = display_help(current_frame_display.shape)
                alpha = 0.6
                cv2.addWeighted(help_overlay, alpha, current_frame_display, 1 - alpha, 0, current_frame_display)

            cv2.imshow(WINDOW_NAME, current_frame_display)

        else:  # If no camera is selected or initialized
            placeholder_frame = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(placeholder_frame, "Select camera: 'e' (Entry) or 'x' (Exit)", (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 1)
            if show_help:
                help_overlay = display_help(placeholder_frame.shape)
                alpha = 0.6
                cv2.addWeighted(help_overlay, alpha, placeholder_frame, 1 - alpha, 0, placeholder_frame)
            cv2.imshow(WINDOW_NAME, placeholder_frame)

        key = cv2.waitKey(30) & 0xFF

        if key == ord('q'):
            print("Saving configuration and exiting...")
            save_roi_config()
            break
        elif key == ord('e'):
            print("Selected Entry camera.")
            active_camera_obj = entry_cam_controller
            active_camera_type_str = "entry"
            current_frame_orig = None
            roi_points = []
            cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=active_camera_type_str)
            if not active_camera_obj or not active_camera_obj.is_initialized_successfully:
                print("ERROR: Entry camera is not available!")
        elif key == ord('x'):
            print("Selected Exit camera.")
            active_camera_obj = exit_cam_controller
            active_camera_type_str = "exit"
            current_frame_orig = None
            roi_points = []
            cv2.setMouseCallback(WINDOW_NAME, mouse_callback, param=active_camera_type_str)
            if not active_camera_obj or not active_camera_obj.is_initialized_successfully:
                print("ERROR: Exit camera is not available!")
        elif key == ord('r'):
            if active_camera_obj:
                print("Refreshing frame...")
                current_frame_orig = None
                roi_points = []
            else:
                print("First, select a camera ('e' or 'x').")
        elif key == ord('s'):
            if active_camera_type_str and len(roi_points) == 2:
                roi_key = f"{active_camera_type_str}_camera_roi"
                active_rois[roi_key]["x1"] = roi_points[0][0]
                active_rois[roi_key]["y1"] = roi_points[0][1]
                active_rois[roi_key]["x2"] = roi_points[1][0]
                active_rois[roi_key]["y2"] = roi_points[1][1]
                if not active_rois[roi_key].get("enabled"):
                    active_rois[roi_key]["enabled"] = True
                print(f"Current drawn ROI for '{active_camera_type_str}' saved to memory: {active_rois[roi_key]}")
                roi_points = []
            elif not active_camera_type_str:
                print("First, select a camera ('e' or 'x').")
            else:
                print("ROI not fully drawn. Use the mouse to draw a rectangle.")
        elif key == ord('c'):
            if active_camera_type_str:
                roi_key = f"{active_camera_type_str}_camera_roi"
                active_rois[roi_key] = {"x1": 0, "y1": 0, "x2": 0, "y2": 0, "enabled": False}
                roi_points = []
                print(f"ROI for '{active_camera_type_str}' cleared and disabled.")
            else:
                print("First, select a camera ('e' or 'x').")
        elif key == ord('t'):  # Toggle enabled
            if active_camera_type_str:
                roi_key = f"{active_camera_type_str}_camera_roi"
                if roi_key in active_rois:
                    current_status = active_rois[roi_key].get("enabled", False)
                    active_rois[roi_key]["enabled"] = not current_status
                    status = "ENABLED" if not current_status else "DISABLED"
                    print(f"ROI for '{active_camera_type_str}' is now {status}.")
                else:
                    print(f"ROI for camera '{active_camera_type_str}' is not yet defined. Draw and save ('s') first.")
            else:
                print("First, select a camera ('e' or 'x').")
        elif key == ord('h'):
            show_help = not show_help
            print(f"Show help: {'Enabled' if show_help else 'Disabled'}")

    # Release resources
    if entry_cam_controller:
        entry_cam_controller.close()
    if exit_cam_controller:
        exit_cam_controller.close()
    cv2.destroyAllWindows()
    print("Finished.")


if __name__ == "__main__":
    main()
