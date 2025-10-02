# utils/image_utils.py
import cv2
import numpy as np
import os
import logging
from typing import Tuple, Optional, Union  # Додано для більш точної типізації

logger = logging.getLogger(__name__)


def save_image(image_array: np.ndarray,
               directory: str,
               filename: str) -> bool:
    """
    Зберігає зображення NumPy у вказаний файл у вказаній директорії.
    Директорія створюється, якщо вона не існує.

    Args:
        image_array (np.ndarray): Зображення для збереження (NumPy масив).
        directory (str): Директорія для збереження файлу.
        filename (str): Ім'я файлу (наприклад, "image.jpg", "frame_001.png").

    Returns:
        bool: True, якщо збереження успішне, False в іншому випадку.
    """
    if image_array is None:
        logger.error("Немає даних зображення для збереження (image_array is None).")
        return False
    if not filename:
        logger.error("Не вказано ім'я файлу для збереження зображення.")
        return False

    file_path = filename  # За замовчуванням, якщо directory порожня
    try:
        if directory:
            # Переконуємося, що директорія існує, якщо ні - створюємо
            if not os.path.exists(directory):
                os.makedirs(directory, exist_ok=True)
                logger.debug(f"Створено директорію: {directory}")
            file_path = os.path.join(directory, filename)

        success = cv2.imwrite(file_path, image_array)
        if success:
            logger.debug(f"Зображення успішно збережено як: {file_path}")
            return True
        else:
            logger.error(f"Не вдалося зберегти зображення у {file_path} (cv2.imwrite повернув False).")
            return False
    except Exception as e:
        logger.error(f"Помилка під час збереження зображення у {file_path}: {e}", exc_info=True)
        return False


def crop_image(image: np.ndarray,
               bbox: Tuple[int, int, int, int]) -> Optional[np.ndarray]:
    """
    Обрізає зображення за вказаними координатами рамки (x1, y1, x2, y2).
    Координати мають бути цілими числами.

    Args:
        image (np.ndarray): Вхідне зображення (NumPy масив).
        bbox (tuple): Кортеж з чотирьох цілих чисел (x1, y1, x2, y2).

    Returns:
        np.ndarray | None: Обрізане зображення або None, якщо виникла помилка
                           (наприклад, некоректні координати або порожнє вхідне зображення).
    """
    if image is None:
        logger.warning("Вхідне зображення для crop_image є None.")
        return None
    if not (isinstance(bbox, tuple) and len(bbox) == 4):
        logger.error(f"Некоректний формат bbox для crop_image: {bbox}. Очікується кортеж з 4 чисел.")
        return None

    try:
        x1, y1, x2, y2 = map(int, bbox)  # Переконуємося, що координати цілі
    except (ValueError, TypeError):
        logger.error(f"Некоректні значення координат у bbox для crop_image: {bbox}.")
        return None

    img_h, img_w = image.shape[:2]

    # Корекція координат, щоб вони не виходили за межі зображення
    # та щоб x1 < x2, y1 < y2
    # Спочатку нормалізуємо порядок x1,x2 та y1,y2
    _x1, _x2 = min(x1, x2), max(x1, x2)
    _y1, _y2 = min(y1, y2), max(y1, y2)

    # Потім обрізаємо за межами зображення
    final_x1 = max(0, _x1)
    final_y1 = max(0, _y1)
    final_x2 = min(img_w, _x2)
    final_y2 = min(img_h, _y2)

    if final_x1 >= final_x2 or final_y1 >= final_y2:
        logger.warning(
            f"Нульовий або від'ємний розмір рамки для обрізки: ({final_x1},{final_y1},{final_x2},{final_y2}) "
            f"для зображення ({img_w},{img_h}). Початковий bbox: {bbox}")
        return None

    return image[final_y1:final_y2, final_x1:final_x2]


def draw_text_with_background(
        image: np.ndarray,
        text: str,
        origin: Tuple[int, int],  # (x, y) - лівий нижній кут тексту (як для cv2.putText)
        font_face: int = cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.6,
        text_color: Tuple[int, int, int] = (255, 255, 255),  # Білий текст
        bg_color: Tuple[int, int, int] = (0, 0, 0),  # Чорний фон
        text_thickness: int = 1,  # Товщина для тексту
        padding: int = 3  # Відступ навколо тексту для фону
) -> None:
    """Малює текст із фоновим прямокутником для кращої видимості."""
    if image is None or not text:
        return

    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, text_thickness)

    bg_x1 = origin[0] - padding
    bg_y1 = origin[1] - text_height - baseline - padding
    bg_x2 = origin[0] + text_width + padding
    bg_y2 = origin[1] + baseline + padding  # baseline враховує "хвостики" букв типу 'g', 'y'

    # Малюємо фон
    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, cv2.FILLED)
    # Малюємо текст поверх фону
    cv2.putText(image, text, (origin[0], origin[1]), font_face, font_scale, text_color, text_thickness, cv2.LINE_AA)


def draw_bounding_box(
        image_on_which_to_draw: np.ndarray,
        bbox: Tuple[int, int, int, int],
        label: str = "",
        score: Optional[float] = None,
        color: Tuple[int, int, int] = (0, 255, 0),  # Зелений за замовчуванням
        thickness: int = 2,
        text_color: Tuple[int, int, int] = (0, 0, 0),  # Чорний текст
        bg_text_color: Optional[Tuple[int, int, int]] = None
        # Колір фону для тексту; якщо None, використовується color рамки
) -> None:
    """
    Малює прямокутну рамку (bbox) та опціональний підпис на зображенні.
    Зображення image_on_which_to_draw модифікується.

    Args:
        image_on_which_to_draw (np.ndarray): Зображення для малювання.
        bbox (tuple): Координати рамки (x1, y1, x2, y2).
        label (str, optional): Текст підпису.
        score (float, optional): Оцінка впевненості для відображення поруч із підписом.
        color (tuple, optional): Колір рамки (BGR).
        thickness (int, optional): Товщина лінії рамки.
        text_color (tuple, optional): Колір тексту.
        bg_text_color (tuple, optional): Колір фону для тексту. Якщо None, використовується `color`.
    """
    if image_on_which_to_draw is None: return

    try:
        x1, y1, x2, y2 = map(int, bbox[:4])
    except (ValueError, TypeError):
        logger.error(f"Некоректний формат bbox для draw_bounding_box: {bbox}")
        return

    cv2.rectangle(image_on_which_to_draw, (x1, y1), (x2, y2), color, thickness)

    display_text = label
    if score is not None:
        display_text = f"{label} {score:.2f}" if label else f"{score:.2f}"

    if display_text:
        font_scale = 0.5

        # Визначаємо розмір тексту для правильного позиціонування фону
        (text_width, text_height), baseline = cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, 1)

        # Позиція тексту трохи вище рамки
        text_y_origin = y1 - baseline - 3  # 3 - невеликий відступ від рамки
        # Якщо текст виходить за верхню межу, малюємо його всередині рамки зверху
        if text_y_origin < text_height:
            text_y_origin = y1 + text_height + baseline + 3  # Зміщено всередину зверху

        final_bg_color = bg_text_color if bg_text_color is not None else color

        draw_text_with_background(
            image_on_which_to_draw,
            display_text,
            (x1, text_y_origin),
            font_scale=font_scale,
            text_color=text_color,
            bg_color=final_bg_color,
            text_thickness=1
        )


# --- Приклад використання (для тестування модуля окремо) ---
if __name__ == '__main__':
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s')

    logger.info("Тестування модуля image_utils.py...")

    test_dir = "image_utils_test_output"  # Буде створено в поточній директорії запуску

    # Створюємо фіктивне зображення
    dummy_image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(dummy_image, "Test Image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    # --- Тест save_image ---
    saved = save_image(dummy_image, test_dir, "dummy_test_image.png")
    logger.info(f"Збереження dummy_image: {'Успішно' if saved else 'Невдало'}")

    # --- Тест crop_image ---
    bbox_to_crop = (50, 70, 300, 180)  # x1, y1, x2, y2
    cropped = crop_image(dummy_image, bbox_to_crop)
    if cropped is not None:
        logger.info(f"Розмір обрізаного зображення: {cropped.shape}")
        saved_cropped = save_image(cropped, test_dir, "dummy_cropped_image.png")
        logger.info(f"Збереження cropped_image: {'Успішно' if saved_cropped else 'Невдало'}")
    else:
        logger.warning("Обрізка не вдалася або повернула None.")

    # Тест crop_image з некоректними межами
    bbox_invalid = (350, 250, 500, 350)  # Частково за межами
    cropped_invalid = crop_image(dummy_image, bbox_invalid)
    if cropped_invalid is not None:
        logger.info(f"Обрізка з некоректними межами (розмір): {cropped_invalid.shape}")
        save_image(cropped_invalid, test_dir, "dummy_cropped_invalid_bbox.png")
    else:
        logger.warning("Обрізка з некоректними межами повернула None (очікувано, якщо рамка повністю поза).")

    # --- Тест малювання ---
    image_for_drawing = dummy_image.copy()

    bbox1 = (30, 30, 150, 100)
    draw_bounding_box(image_for_drawing, bbox1, "Car", score=0.95, color=(0, 255, 0))

    bbox2 = (180, 120, 350, 250)
    draw_bounding_box(image_for_drawing, bbox2, "Truck", score=0.80, color=(255, 0, 0), text_color=(255, 255, 255))

    bbox3 = (10, 200, 100, 280)  # Текст може вийти за межі, якщо малювати надто близько до краю
    draw_bounding_box(image_for_drawing, bbox3, "Cycle", color=(0, 0, 255))

    draw_text_with_background(image_for_drawing, "Самостійний Текст", (150, 280),
                              font_scale=0.7, text_color=(0, 255, 255), bg_color=(50, 50, 50))

    saved_drawing = save_image(image_for_drawing, test_dir, "dummy_with_drawings.png")
    logger.info(f"Збереження image_for_drawing: {'Успішно' if saved_drawing else 'Невдало'}")

    logger.info(f"Тестування image_utils.py завершено. Перевірте директорію: {os.path.abspath(test_dir)}")