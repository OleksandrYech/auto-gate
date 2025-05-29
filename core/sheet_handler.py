# core/sheet_handler.py
import gspread
# from google.oauth2.service_account import Credentials
# from gspread import authorize
from oauth2client.service_account import ServiceAccountCredentials
import logging
from datetime import datetime
import os  # Для перевірки шляху до credentials у тесті
from typing import Optional
import time

# --- Global Configuration ---
logger = logging.getLogger(__name__)  # Створюємо логгер для цього модуля

# --- Constants ---
SCOPES = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']
CREDENTIALS_FILE = 'config/credentials.json'

# --- Google Sheet Details ---
YOUR_SPREADSHEET_URL = 'https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0'

# Назва аркуша
VEHICLES_SHEET_NAME = 'Vehicles'

# В'їзд авторизованих: Стовпець A (Номер), Стовпець B (Час останнього в'їзду)
ENTRY_PLATE_COL_A_NUM = 1  # Номер стовпця A (для gspread)
ENTRY_TIMESTAMP_COL_B_NUM = 2  # Номер стовпця B
ENTRY_DATA_START_ROW = 3  # Дані починаються з 3-го рядка

# Неавторизовані спроби: Стовпець D (Номер), Стовпець E (Час спроби)
UNAUTHORIZED_PLATE_COL_D_NUM = 4  # Номер стовпця D
UNAUTHORIZED_TIMESTAMP_COL_E_NUM = 5  # Номер стовпця E
UNAUTHORIZED_DATA_START_ROW = 3  # Дані починаються з 3-го рядка

# Лог виїзду: Стовпець G ("Номер"), Стовпець H ("Останній Виїзд")
EXIT_PLATE_COL_G_NUM = 7  # Номер стовпця G
EXIT_TIMESTAMP_COL_H_NUM = 8  # Номер стовпця H
EXIT_DATA_START_ROW = 3  # Дані починаються з 3-го рядка

# --- Глобальна змінна для клієнта gspread ---
_sheet_client_instance: Optional[gspread.Client] = None


def _get_sheet_client() -> Optional[gspread.Client]:
    """ Ініціалізує та повертає клієнт gspread. """
    global _sheet_client_instance
    if _sheet_client_instance is None:
        logger.info(f"Спроба підключення до Google Sheets API через {CREDENTIALS_FILE}...")
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(CREDENTIALS_FILE, SCOPES)
            _sheet_client_instance = gspread.authorize(creds)
            logger.info("Успішно підключено та авторизовано з Google Sheets API.")
        except FileNotFoundError:
            logger.error(f"Файл облікових даних '{CREDENTIALS_FILE}' не знайдено.")
            _sheet_client_instance = None
        except Exception as e:
            logger.error(f"Не вдалося авторизуватися або підключитися до Google Sheets API: {e}", exc_info=True)
            _sheet_client_instance = None
    return _sheet_client_instance


def _get_worksheet(sheet_name: str) -> Optional[gspread.Worksheet]:
    """ Допоміжна функція для отримання конкретного аркуша. """
    client = _get_sheet_client()
    if not client: return None
    try:
        if not YOUR_SPREADSHEET_URL or YOUR_SPREADSHEET_URL == 'YOUR_SPREADSHEET_URL_HERE':
            logger.error("URL таблиці не налаштовано. Встановіть значення YOUR_SPREADSHEET_URL.")
            return None
        spreadsheet = client.open_by_url(YOUR_SPREADSHEET_URL)
        worksheet = spreadsheet.worksheet(sheet_name)
        return worksheet
    except gspread.exceptions.SpreadsheetNotFound:
        logger.error(f"Таблицю не знайдено за URL: {YOUR_SPREADSHEET_URL}.")
        return None
    except gspread.exceptions.WorksheetNotFound:
        logger.error(f"Аркуш '{sheet_name}' не знайдено в таблиці.")
        return None
    except Exception as e:
        logger.error(f"Помилка відкриття аркуша '{sheet_name}': {e}", exc_info=True)
        return None


def find_vehicle_and_update_entry_time(plate_number: str) -> bool:
    """
    Знаходить номерний знак в аркуші 'Vehicles' (Стовпець A).
    Якщо знайдено (починаючи з ENTRY_DATA_START_ROW), оновлює час останнього в'їзду в Стовпці B.
    """
    worksheet = _get_worksheet(VEHICLES_SHEET_NAME)
    if not worksheet: return False

    try:
        logger.debug(f"Пошук номера '{plate_number}' для в'їзду в '{VEHICLES_SHEET_NAME}'.")
        cell: Optional[gspread.Cell] = None
        try:
            # Шукаємо в усьому стовпці A
            cells_found = worksheet.findall(plate_number, in_column=ENTRY_PLATE_COL_A_NUM)
            # Фільтруємо ті, що знаходяться на або нижче ENTRY_DATA_START_ROW
            valid_cells = [c for c in cells_found if c.row >= ENTRY_DATA_START_ROW]
            if valid_cells:
                cell = valid_cells[0]  # Беремо перше співпадіння, що задовольняє умову
        except gspread.exceptions.CellNotFound:  # findall поверне порожній список, а не виняток
            pass

        if cell:  # cell тепер містить gspread.Cell, якщо знайдено і валідно
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.update_cell(cell.row, ENTRY_TIMESTAMP_COL_B_NUM, current_datetime)
            logger.info(f"Авто '{plate_number}' знайдено в рядку {cell.row}. Час в'їзду оновлено: {current_datetime}.")
            return True
        else:
            logger.info(
                f"Номер '{plate_number}' не знайдено (або не в діапазоні даних) в стовпці A аркуша '{VEHICLES_SHEET_NAME}'.")
            return False
    except Exception as e:
        logger.error(f"Помилка пошуку/оновлення авто '{plate_number}' для в'їзду: {e}", exc_info=True)
        return False


def add_unauthorized_attempt(plate_number: str) -> bool:
    """
    Додає запис про неавторизовану спробу в'їзду в аркуш 'Vehicles' (Стовпці D та E).
    """
    worksheet = _get_worksheet(VEHICLES_SHEET_NAME)
    if not worksheet: return False

    try:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        # Знаходимо наступний порожній рядок у стовпці D, починаючи з UNAUTHORIZED_DATA_START_ROW
        col_d_values = worksheet.col_values(UNAUTHORIZED_PLATE_COL_D_NUM)
        next_row_to_write = UNAUTHORIZED_DATA_START_ROW

        # Починаємо пошук з індексу, що відповідає UNAUTHORIZED_DATA_START_ROW
        start_index_for_search = UNAUTHORIZED_DATA_START_ROW - 1
        if start_index_for_search < len(col_d_values):
            for i in range(start_index_for_search, len(col_d_values)):
                if not col_d_values[i]:  # Якщо комірка порожня
                    next_row_to_write = i + 1  # gspread рядки 1-індексовані
                    break
            else:  # Якщо всі комірки до кінця col_d_values заповнені
                next_row_to_write = len(col_d_values) + 1
        # Якщо col_d_values коротший за start_index_for_search, next_row_to_write залишається UNAUTHORIZED_DATA_START_ROW

        # Переконуємося, що не пишемо вище за стартовий рядок
        if next_row_to_write < UNAUTHORIZED_DATA_START_ROW:
            next_row_to_write = UNAUTHORIZED_DATA_START_ROW

        worksheet.update_cell(next_row_to_write, UNAUTHORIZED_PLATE_COL_D_NUM, plate_number)
        worksheet.update_cell(next_row_to_write, UNAUTHORIZED_TIMESTAMP_COL_E_NUM, current_datetime)

        logger.info(f"Неавторизовану спробу '{plate_number}' залоговано о {current_datetime} "
                    f"в аркуші '{VEHICLES_SHEET_NAME}', рядок {next_row_to_write} (стовпці D,E).")
        return True
    except Exception as e:
        logger.error(f"Помилка додавання неавторизованої спроби для '{plate_number}': {e}", exc_info=True)
        return False


def log_vehicle_exit(plate_number: str) -> bool:
    """
    Записує номерний знак та час виїзду автомобіля в аркуш 'Vehicles' (Стовпці G та H).
    """
    worksheet = _get_worksheet(VEHICLES_SHEET_NAME)
    if not worksheet: return False

    try:
        current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

        # Знаходимо наступний порожній рядок у стовпці G, починаючи з EXIT_DATA_START_ROW
        col_g_values = worksheet.col_values(EXIT_PLATE_COL_G_NUM)
        next_row_to_write = EXIT_DATA_START_ROW

        start_index_for_search = EXIT_DATA_START_ROW - 1
        if start_index_for_search < len(col_g_values):
            for i in range(start_index_for_search, len(col_g_values)):
                if not col_g_values[i]:
                    next_row_to_write = i + 1
                    break
            else:
                next_row_to_write = len(col_g_values) + 1

        if next_row_to_write < EXIT_DATA_START_ROW:
            next_row_to_write = EXIT_DATA_START_ROW

        worksheet.update_cell(next_row_to_write, EXIT_PLATE_COL_G_NUM, plate_number)
        worksheet.update_cell(next_row_to_write, EXIT_TIMESTAMP_COL_H_NUM, current_datetime)

        logger.info(f"Виїзд автомобіля залоговано: Номер '{plate_number}', Час {current_datetime} "
                    f"в аркуші '{VEHICLES_SHEET_NAME}', рядок {next_row_to_write} (стовпці G,H).")
        return True
    except Exception as e:
        logger.error(f"Помилка логування виїзду авто для номера '{plate_number}': {e}", exc_info=True)
        return False


# --- Головний блок для тестування модуля ---
if __name__ == '__main__':
    # Налаштування базового логування для виводу в консоль під час тестування
    if not logger.hasHandlers():
        console_handler = logging.StreamHandler()
        formatter = logging.Formatter('%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s')
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)
        logger.setLevel(logging.DEBUG)
        # Щоб бачити логи gspread, можна додати:
        # logging.getLogger('gspread').setLevel(logging.DEBUG)

    logger.info("--- Тестування модуля sheets.py ---")

    if YOUR_SPREADSHEET_URL == 'YOUR_SPREADSHEET_URL_HERE':
        logger.critical("КРИТИЧНО: YOUR_SPREADSHEET_URL не встановлено у sheets.py.")
    elif not os.path.exists(CREDENTIALS_FILE):
        logger.critical(
            f"КРИТИЧНО: Файл облікових даних '{CREDENTIALS_FILE}' не знайдено: {os.path.abspath(CREDENTIALS_FILE)}.")
    else:
        logger.info(f"Використовується URL таблиці: {YOUR_SPREADSHEET_URL}")
        logger.info(f"Очікується файл облікових даних: {os.path.abspath(CREDENTIALS_FILE)}")

        client_test = _get_sheet_client()
        if not client_test:
            logger.error("Не вдалося отримати клієнт для Google Sheets. Подальші тести неможливі.")
        else:
            logger.info("Клієнт для Google Sheets успішно отримано.")

            # --- Тест: Пошук та оновлення в'їзду ---
            # "EXISTING_PLATE" номер, що точно є у стовпці A
            test_plate_entry_allowed = "AA1111AA"
            logger.info(f"\nТест 1: Пошук та оновлення в'їзду для '{test_plate_entry_allowed}'")
            if find_vehicle_and_update_entry_time(test_plate_entry_allowed):
                logger.info(f"  УСПІХ: Час в'їзду для '{test_plate_entry_allowed}' оновлено.")
            else:
                logger.warning(
                    f"  ПОМИЛКА/НЕ ЗНАЙДЕНО: '{test_plate_entry_allowed}'. Перевірте номер та права доступу.")

            # --- Тест: Спроба знайти неіснуючий номер для в'їзду ---
            test_plate_entry_non_existent = f"NONEXISTENT_{int(time.time()) % 1000}"
            logger.info(f"\nТест 2: Пошук неіснуючого номера '{test_plate_entry_non_existent}' для в'їзду")
            if not find_vehicle_and_update_entry_time(test_plate_entry_non_existent):
                logger.info(f"  УСПІХ (очікувано): Номер '{test_plate_entry_non_existent}' не знайдено.")
            else:
                logger.error(f"  ПОМИЛКА (неочікувано): Номер '{test_plate_entry_non_existent}' знайдено.")

            # --- Тест: Додавання неавторизованої спроби ---
            test_plate_unauthorized = f"UNAUTH_{int(time.time()) % 1000}"
            logger.info(f"\nТест 3: Логування неавторизованої спроби для '{test_plate_unauthorized}'")
            if add_unauthorized_attempt(test_plate_unauthorized):
                logger.info(f"  УСПІХ: Спроба '{test_plate_unauthorized}' залогована (Стовпці D,E).")
            else:
                logger.warning(f"  ПОМИЛКА: Не вдалося залогувати спробу '{test_plate_unauthorized}'.")

            time.sleep(1)  # Щоб уникнути однакових міток часу
            test_plate_unauthorized_2 = f"UNAUTH_NEXT_{int(time.time()) % 1000}"
            logger.info(f"\nТест 4: Логування ще однієї неавторизованої спроби '{test_plate_unauthorized_2}'")
            if add_unauthorized_attempt(test_plate_unauthorized_2):
                logger.info(f"  УСПІХ: Спроба '{test_plate_unauthorized_2}' залогована.")
            else:
                logger.warning(f"  ПОМИЛКА: Не вдалося залогувати спробу '{test_plate_unauthorized_2}'.")

            # --- Тест: Логування виїзду автомобіля ---
            test_plate_exit = f"EXITCAR_{int(time.time()) % 1000}"
            logger.info(f"\nТест 5: Логування виїзду для '{test_plate_exit}'")
            if log_vehicle_exit(test_plate_exit):
                logger.info(f"  УСПІХ: Виїзд '{test_plate_exit}' залоговано (Стовпці G,H).")
            else:
                logger.warning(f"  ПОМИЛКА: Не вдалося залогувати виїзд '{test_plate_exit}'.")

            time.sleep(1)
            test_plate_exit_2 = f"EXITNEXT_{int(time.time()) % 1000}"
            logger.info(f"\nТест 6: Логування ще одного виїзду '{test_plate_exit_2}'")
            if log_vehicle_exit(test_plate_exit_2):
                logger.info(f"  УСПІХ: Виїзд '{test_plate_exit_2}' залоговано.")
            else:
                logger.warning(f"  ПОМИЛКА: Не вдалося залогувати виїзд '{test_plate_exit_2}'.")

    logger.info("\n--- Завершено тестування модуля sheets.py ---")
    logger.info("Будь ласка, перевірте вашу Google Таблицю для верифікації результатів.")