# utils/image_utils.py
import cv2
import numpy as np
import os
import logging

logger = logging.getLogger(__name__)


def save_image(image_array: np.ndarray, directory: str, filename: str) -> bool:
    """
    Зберігає зображення NumPy у вказаний файл у вказаній директорії.
    Директорія створюється, якщо вона не існує.

    Args:
        image_array (np.ndarray): Зображення для збереження (NumPy масив).
        directory (str): Директорія для збереження файлу.
        filename (str): Ім'я файлу.

    Returns:
        bool: True, якщо збереження успішне, False в іншому випадку.
    """
    if image_array is None:
        logger.error("Немає даних зображення для збереження.")
        return False
    if not filename:
        logger.error("Не вказано ім'я файлу для збереження зображення.")
        return False

    try:
        if directory:  # Якщо директорія вказана, створюємо її
            os.makedirs(directory, exist_ok=True)
            file_path = os.path.join(directory, filename)
        else:  # Якщо директорія не вказана, зберігаємо в поточну
            file_path = filename

        cv2.imwrite(file_path, image_array)
        logger.debug(f"Зображення успішно збережено як: {file_path}")
        return True
    except Exception as e:
        logger.error(f"Не вдалося зберегти зображення у {file_path}: {e}", exc_info=True)
        return False


def crop_image(image: np.ndarray, bbox: tuple) -> np.ndarray | None:
    """
    Обрізає зображення за вказаними координатами рамки (x1, y1, x2, y2).

    Args:
        image (np.ndarray): Вхідне зображення (NumPy масив).
        bbox (tuple): Кортеж з чотирьох цілих чисел (x1, y1, x2, y2).

    Returns:
        np.ndarray | None: Обрізане зображення або None, якщо виникла помилка.
    """
    if image is None:
        logger.warning("Вхідне зображення для crop_image є None.")
        return None
    try:
        x1, y1, x2, y2 = map(int, bbox[:4])
    except (ValueError, TypeError):
        logger.error(f"Некоректний формат bbox для crop_image: {bbox}. Очікується кортеж з 4 чисел.")
        return None

    h, w = image.shape[:2]

    # Перевірка та корекція координат, щоб вони не виходили за межі зображення
    x1_c = max(0, x1)
    y1_c = max(0, y1)
    x2_c = min(w, x2)  # x2 може бути рівним ширині (аналогічно для висоти)
    y2_c = min(h, y2)

    if x1_c >= x2_c or y1_c >= y2_c:
        logger.warning(f"Некоректні або нульові розміри рамки для обрізки: ({x1_c},{y1_c},{x2_c},{y2_c}) "
                       f"для зображення розміром ({w},{h}). Початковий bbox: {bbox}")
        return None

    return image[y1_c:y2_c, x1_c:x2_c]


def draw_text_with_background(
        image: np.ndarray,
        text: str,
        origin: tuple,  # (x, y) - лівий нижній кут тексту
        font_face=cv2.FONT_HERSHEY_SIMPLEX,
        font_scale: float = 0.6,
        text_color: tuple = (255, 255, 255),  # Білий текст
        bg_color: tuple = (0, 0, 0),  # Чорний фон
        thickness: int = 1,
        padding: int = 3  # Відступ навколо тексту для фону
):
    """Малює текст із фоновим прямокутником для кращої видимості."""
    if image is None: return

    (text_width, text_height), baseline = cv2.getTextSize(text, font_face, font_scale, thickness)

    # Координати фонового прямокутника
    bg_x1 = origin[0] - padding
    bg_y1 = origin[1] + baseline - text_height - padding
    bg_x2 = origin[0] + text_width + padding
    bg_y2 = origin[1] + baseline + padding

    # Перевірка, щоб фон не виходив за межі зображення (опціонально, але корисно)
    # bg_x1 = max(0, bg_x1); bg_y1 = max(0, bg_y1)
    # bg_x2 = min(image.shape[1], bg_x2); bg_y2 = min(image.shape[0], bg_y2)

    cv2.rectangle(image, (bg_x1, bg_y1), (bg_x2, bg_y2), bg_color, cv2.FILLED)
    cv2.putText(image, text, origin, font_face, font_scale, text_color, thickness, cv2.LINE_AA)


def draw_bounding_box(
        image_on_which_to_draw: np.ndarray,  # Зображення, на якому малювати (буде змінено)
        bbox: tuple,
        label: str = "",
        score: float = None,
        color: tuple = (0, 255, 0),
        thickness: int = 2,
        text_color: tuple = (0, 0, 0),  # Чорний текст на світлому фоні рамки
        bg_text_color: tuple = None  # Колір фону для тексту (такий же, як color, якщо None)
):
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
        bg_text_color (tuple, optional): Колір фону для тексту. Якщо None, використовується color.
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
        # Визначаємо розмір тексту, щоб фон був відповідного розміру
        font_scale = 0.5
        font_thickness = 1

        # Позиція тексту трохи вище рамки
        text_origin_y = y1 - 10
        # Якщо текст виходить за верхню межу, малюємо його всередині рамки знизу
        if text_origin_y < 10: text_origin_y = y1 + 10 + (
        cv2.getTextSize(display_text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, font_thickness)[0][1])

        draw_text_with_background(
            image_on_which_to_draw,
            display_text,
            (x1, text_origin_y),  # Лівий верхній кут тексту
            font_scale=font_scale,
            text_color=text_color,
            bg_color=bg_text_color if bg_text_color is not None else color,  # Фон кольору рамки
            thickness=font_thickness,
            padding=2
        )


if __name__ == '__main__':
    # Налаштування логування для тестування цього модуля
    if not logging.getLogger().handlers:  # Перевіряємо, чи є вже обробники
        logging.basicConfig(level=logging.DEBUG,
                            format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')

    logger.info("Тестування модуля image_utils.py...")

    # --- Тест save_image та crop_image ---
    test_dir = "image_utils_test_output"
    os.makedirs(test_dir, exist_ok=True)

    # Створюємо фіктивне зображення
    dummy_image = np.zeros((300, 400, 3), dtype=np.uint8)
    cv2.putText(dummy_image, "Test Image", (50, 150), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

    saved = save_image(dummy_image, test_dir, "dummy_test_image.png")
    logger.info(f"Збереження dummy_image: {'Успішно' if saved else 'Невдало'}")

    # Тест обрізки
    bbox_to_crop = (50, 50, 250, 200)  # x1, y1, x2, y2
    cropped = crop_image(dummy_image, bbox_to_crop)
    if cropped is not None:
        logger.info(f"Розмір обрізаного зображення: {cropped.shape}")
        saved_cropped = save_image(cropped, test_dir, "dummy_cropped_image.png")
        logger.info(f"Збереження cropped_image: {'Успішно' if saved_cropped else 'Невдало'}")
    else:
        logger.warning("Обрізка не вдалася або повернула None.")

    # --- Тест малювання рамок та тексту ---
    image_for_drawing = dummy_image.copy()  # Малюємо на копії

    # Тест draw_bounding_box
    bbox1 = (30, 30, 150, 100)
    draw_bounding_box(image_for_drawing, bbox1, "Об'єкт 1", score=0.95, color=(0, 255, 0))

    bbox2 = (180, 120, 350, 250)
    draw_bounding_box(image_for_drawing, bbox2, "Інший", score=0.80, color=(255, 0, 0), text_color=(255, 255, 255))

    # Тест draw_text_with_background (незалежно)
    draw_text_with_background(image_for_drawing, "Тестовий Текст З Фоном", (10, 280),
                              font_scale=0.7, text_color=(0, 255, 255), bg_color=(100, 0, 0))

    saved_drawing = save_image(image_for_drawing, test_dir, "dummy_with_drawings.png")
    logger.info(f"Збереження image_for_drawing: {'Успішно' if saved_drawing else 'Невдало'}")

    logger.info(f"Тестування image_utils.py завершено. Перевірте директорію: {os.path.abspath(test_dir)}")