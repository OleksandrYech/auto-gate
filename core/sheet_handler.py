# core/sheet_handler.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials  # Використовуємо згідно з вашим файлом
import logging
from datetime import datetime
import os
from typing import Optional, List

# --- Константи ---
# Області доступу для Google API
DEFAULT_SCOPES: List[str] = ['https://www.googleapis.com/auth/spreadsheets',
                             'https://www.googleapis.com/auth/drive.file']
# Стандартне ім'я файлу облікових даних
DEFAULT_CREDENTIALS_FILE: str = 'credentials.json'

# !!! ВАЖЛИВО: Замініть це на URL вашої Google Таблиці, АБО передавайте через конструктор з main.py !!!
DEFAULT_SPREADSHEET_URL: str = 'https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0'  # Наприклад, 'https://docs.google.com/spreadsheets/d/your_sheet_id/edit#gid=0'

# Назва аркуша
DEFAULT_VEHICLES_SHEET_NAME: str = 'Vehicles'  # Припускаємо, що ваш головний аркуш так називається

# Структура аркуша "Vehicles" (номери стовпців 1-індексовані для gspread)
# В'їзд авторизованих:
ENTRY_PLATE_COL_A_NUM: int = 1  # Стовпець A: Номер
ENTRY_TIMESTAMP_COL_B_NUM: int = 2  # Стовпець B: Час останнього в'їзду
ENTRY_DATA_START_ROW: int = 3  # Дані починаються з 3-го рядка

# Неавторизовані спроби:
UNAUTHORIZED_PLATE_COL_D_NUM: int = 4  # Стовпець D: Номер (неавторизований)
UNAUTHORIZED_TIMESTAMP_COL_E_NUM: int = 5  # Стовпець E: Дата спроби
UNAUTHORIZED_DATA_START_ROW: int = 3  # Дані починаються з 3-го рядка

# Лог виїзду:
# Стовпець F: Візуальний роздільник (не використовується кодом)
EXIT_PLATE_COL_G_NUM: int = 7  # Стовпець G: Номер (Виїзд)
EXIT_TIMESTAMP_COL_H_NUM: int = 8  # Стовпець H: Останній Виїзд (Виїзд)
EXIT_DATA_START_ROW: int = 3  # Дані починаються з 3-го рядка


class SheetHandler:
    """
    Клас для взаємодії з Google Sheets для авторизації транспортних засобів
    та логування подій в'їзду/виїзду.
    """

    def __init__(self,
                 spreadsheet_url: str = DEFAULT_SPREADSHEET_URL,
                 credentials_file_path: str = DEFAULT_CREDENTIALS_FILE,  # Дозволяємо передавати повний шлях
                 scopes: Optional[List[str]] = None,
                 vehicles_sheet_name: str = DEFAULT_VEHICLES_SHEET_NAME):
        """
        Ініціалізація SheetHandler.

        Args:
            spreadsheet_url (str): URL Google Таблиці.
            credentials_file_path (str): Шлях до файлу облікових даних JSON.
                                       Якщо відносний, шукається спочатку поруч, потім у 'config/'.
            scopes (List[str], optional): Список областей доступу для Google API.
            vehicles_sheet_name (str): Назва основного аркуша в таблиці.
        """
        self._logger = logging.getLogger(f"{__name__}.{self.__class__.__name__}")

        self.spreadsheet_url = spreadsheet_url
        self.credentials_file = self._resolve_credentials_path(credentials_file_path)
        self.scopes = scopes if scopes is not None else DEFAULT_SCOPES
        self.vehicles_sheet_name = vehicles_sheet_name

        self._client: Optional[gspread.Client] = None
        self._active_worksheet: Optional[gspread.Worksheet] = None

        if self.spreadsheet_url == 'YOUR_SPREADSHEET_URL_HERE' or not self.spreadsheet_url:
            self._logger.critical("URL Google Таблиці не встановлено! Будь ласка, встановіть spreadsheet_url.")
            # Можна кинути виняток або залишити, методи не працюватимуть.
            # Для main.py краще кинути виняток або перевіряти _client після ініціалізації.

        self._connect_and_authorize()

    def _resolve_credentials_path(self, creds_file_path: str) -> str:
        """Визначає абсолютний шлях до файлу облікових даних."""
        if os.path.isabs(creds_file_path):
            return creds_file_path

        # Спроба знайти відносно поточного файлу (core/)
        path_near_module = os.path.join(os.path.dirname(os.path.abspath(__file__)), creds_file_path)
        if os.path.exists(path_near_module):
            return path_near_module

        # Спроба знайти відносно кореня проекту (припускаючи, що core/ є піддиректорією)
        project_root_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", creds_file_path)
        if os.path.exists(project_root_path):
            return project_root_path

        # Спроба знайти у 'config/' відносно кореня проекту
        config_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config",
                                   os.path.basename(creds_file_path))
        if os.path.exists(config_path):
            return config_path

        # Якщо не знайдено, повертаємо оригінальний шлях (може спричинити FileNotFoundError пізніше)
        self._logger.warning(
            f"Файл облікових даних '{creds_file_path}' не знайдено у стандартних місцях. Буде використано як є.")
        return creds_file_path

    def _connect_and_authorize(self) -> bool:
        if self._client:
            return True

        self._logger.info(f"Спроба підключення до Google Sheets API через {self.credentials_file}...")
        try:
            if not os.path.exists(self.credentials_file):
                self._logger.error(
                    f"Файл облікових даних '{self.credentials_file}' не знайдено за шляхом: {os.path.abspath(self.credentials_file)}")
                return False

            creds = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_file, self.scopes)
            self._client = gspread.authorize(creds)
            self._logger.info("Успішно підключено та авторизовано з Google Sheets API.")
            return True
        except Exception as e:
            self._logger.error(f"Не вдалося авторизуватися або підключитися до Google Sheets API: {e}", exc_info=True)
            self._client = None  # Переконуємося, що клієнт None у разі помилки
            return False

    def _get_worksheet(self, sheet_name: Optional[str] = None) -> Optional[gspread.Worksheet]:
        if not self._client:
            self._logger.warning("Клієнт Google Sheets не ініціалізовано. Спроба повторного підключення.")
            if not self._connect_and_authorize():
                return None

        target_sheet_name = sheet_name if sheet_name is not None else self.vehicles_sheet_name

        if self._active_worksheet and self._active_worksheet.title == target_sheet_name:
            return self._active_worksheet

        try:
            if not self.spreadsheet_url or self.spreadsheet_url == 'YOUR_SPREADSHEET_URL_HERE':
                self._logger.error("URL таблиці не налаштовано.")
                return None

            spreadsheet = self._client.open_by_url(self.spreadsheet_url)
            worksheet = spreadsheet.worksheet(target_sheet_name)
            self._active_worksheet = worksheet
            return worksheet
        except gspread.exceptions.SpreadsheetNotFound:
            self._logger.error(f"Таблицю не знайдено за URL: {self.spreadsheet_url}.")
        except gspread.exceptions.WorksheetNotFound:
            self._logger.error(f"Аркуш '{target_sheet_name}' не знайдено в таблиці.")
        except Exception as e:
            self._logger.error(f"Помилка відкриття аркуша '{target_sheet_name}': {e}", exc_info=True)

        self._active_worksheet = None
        return None

    def find_vehicle_and_update_entry_time(self, plate_number: str) -> bool:
        worksheet = self._get_worksheet()
        if not worksheet:
            self._logger.error(
                f"Не вдалося отримати аркуш '{self.vehicles_sheet_name}' для пошуку авто '{plate_number}'.")
            return False

        try:
            self._logger.debug(f"Пошук номера '{plate_number}' для в'їзду в '{self.vehicles_sheet_name}'.")
            cell: Optional[gspread.Cell] = None
            try:
                cells_found = worksheet.findall(plate_number, in_column=ENTRY_PLATE_COL_A_NUM)
                valid_cells = [c for c in cells_found if c.row >= ENTRY_DATA_START_ROW]
                if valid_cells:
                    cell = valid_cells[0]
            except gspread.exceptions.CellNotFound:
                pass

            if cell:
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                worksheet.update_cell(cell.row, ENTRY_TIMESTAMP_COL_B_NUM, current_datetime)
                self._logger.info(
                    f"Авто '{plate_number}' знайдено в рядку {cell.row}. Час в'їзду оновлено: {current_datetime}.")
                return True
            else:
                self._logger.info(
                    f"Номер '{plate_number}' не знайдено (або не в діапазоні даних) в стовпці A аркуша '{self.vehicles_sheet_name}'.")
                return False
        except Exception as e:
            self._logger.error(f"Помилка пошуку/оновлення авто '{plate_number}' для в'їзду: {e}", exc_info=True)
            return False

    def _find_next_empty_row(self, worksheet: gspread.Worksheet, column_num: int, start_row: int) -> int:
        """Знаходить номер наступного порожнього рядка в заданому стовпці, починаючи зі start_row."""
        col_values = worksheet.col_values(column_num)
        next_row = start_row

        # Індексація для col_values починається з 0, gspread рядки - з 1
        idx_start = start_row - 1

        if idx_start < len(col_values):
            for i in range(idx_start, len(col_values)):
                if not col_values[i]:  # Якщо комірка порожня
                    next_row = i + 1
                    return next_row
            # Якщо всі комірки до кінця заповнені
            next_row = len(col_values) + 1
        # Якщо col_values коротший (наприклад, стовпець порожній після заголовків),
        # next_row залишиться start_row (або буде збільшений, якщо start_row теж порожній)

        # Переконуємося, що не пишемо вище за стартовий рядок
        return max(next_row, start_row)

    def add_unauthorized_attempt(self, plate_number: str) -> bool:
        worksheet = self._get_worksheet()
        if not worksheet:
            self._logger.error(f"Не вдалося отримати аркуш для логування неавторизованої спроби '{plate_number}'.")
            return False

        try:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            next_row_to_write = self._find_next_empty_row(worksheet, UNAUTHORIZED_PLATE_COL_D_NUM,
                                                          UNAUTHORIZED_DATA_START_ROW)

            worksheet.update_cell(next_row_to_write, UNAUTHORIZED_PLATE_COL_D_NUM, plate_number)
            worksheet.update_cell(next_row_to_write, UNAUTHORIZED_TIMESTAMP_COL_E_NUM, current_datetime)

            self._logger.info(f"Неавторизовану спробу '{plate_number}' залоговано о {current_datetime} "
                              f"в '{self.vehicles_sheet_name}', рядок {next_row_to_write}.")
            return True
        except Exception as e:
            self._logger.error(f"Помилка додавання неавторизованої спроби для '{plate_number}': {e}", exc_info=True)
            return False

    def log_vehicle_exit(self, plate_number: str) -> bool:
        worksheet = self._get_worksheet()
        if not worksheet:
            self._logger.error(f"Не вдалося отримати аркуш для логування виїзду '{plate_number}'.")
            return False

        try:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            next_row_to_write = self._find_next_empty_row(worksheet, EXIT_PLATE_COL_G_NUM, EXIT_DATA_START_ROW)

            worksheet.update_cell(next_row_to_write, EXIT_PLATE_COL_G_NUM, plate_number)
            worksheet.update_cell(next_row_to_write, EXIT_TIMESTAMP_COL_H_NUM, current_datetime)

            self._logger.info(f"Виїзд '{plate_number}' залоговано о {current_datetime} "
                              f"в '{self.vehicles_sheet_name}', рядок {next_row_to_write}.")
            return True
        except Exception as e:
            self._logger.error(f"Помилка логування виїзду для '{plate_number}': {e}", exc_info=True)
            return False


# --- Блок для тестування модуля ---
if __name__ == '__main__':
    # Налаштування логування для тестування
    if not logging.getLogger().hasHandlers():
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s',
            handlers=[logging.StreamHandler()]
        )

    test_logger_main = logging.getLogger(__name__)
    test_logger_main.info("--- Тестування модуля sheet_handler.py ---")

    # ВАЖЛИВО: Для тестування встановіть реальний URL та переконайтесь, що файл credentials існує
    # SPREADSHEET_URL_FOR_TEST = 'https://docs.google.com/spreadsheets/d/your_actual_sheet_id/edit#gid=0'
    SPREADSHEET_URL_FOR_TEST = DEFAULT_SPREADSHEET_URL  # Використовуємо значення за замовчуванням з констант

    # Визначаємо шлях до credentials.json, припускаючи, що скрипт в core/, а config/ на рівень вище
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)  # Батьківська директорія для core/
    CREDENTIALS_FILE_FOR_TEST = os.path.join(project_root, "config", os.path.basename(DEFAULT_CREDENTIALS_FILE))

    if SPREADSHEET_URL_FOR_TEST == 'YOUR_SPREADSHEET_URL_HERE':
        test_logger_main.critical("КРИТИЧНО: URL таблиці не встановлено для тестування.")
    elif not os.path.exists(CREDENTIALS_FILE_FOR_TEST):
        test_logger_main.critical(f"КРИТИЧНО: Файл облікових даних '{CREDENTIALS_FILE_FOR_TEST}' не знайдено.")
    else:
        test_logger_main.info(f"Використовується URL таблиці: {SPREADSHEET_URL_FOR_TEST}")
        test_logger_main.info(f"Очікується файл облікових даних: {CREDENTIALS_FILE_FOR_TEST}")

        handler = SheetHandler(
            spreadsheet_url=SPREADSHEET_URL_FOR_TEST,
            credentials_file_path=CREDENTIALS_FILE_FOR_TEST
        )

        if not handler._client:
            test_logger_main.error("Не вдалося отримати клієнт Google Sheets. Подальші тести неможливі.")
        else:
            test_logger_main.info("Клієнт для Google Sheets успішно отримано.")

            # --- Тести ---
            # ЗАМІНІТЬ ЦЕ НА РЕАЛЬНИЙ НОМЕР З ВАШОЇ ТАБЛИЦІ (Стовпець A, з 3-го рядка)
            existing_authorized_plate = "AA0000AA"
            test_logger_main.info(f"\nТест 1: Оновлення часу в'їзду для авторизованого '{existing_authorized_plate}'")
            if handler.find_vehicle_and_update_entry_time(existing_authorized_plate):
                test_logger_main.info("  УСПІХ.")
            else:
                test_logger_main.warning(
                    f"  ПОМИЛКА/НЕ ЗНАЙДЕНО: '{existing_authorized_plate}'. Перевірте номер у таблиці.")

            non_existent_plate = f"NE{int(time.time()) % 10000}XX"
            test_logger_main.info(f"\nТест 2: Пошук неіснуючого номера '{non_existent_plate}' для в'їзду")
            if not handler.find_vehicle_and_update_entry_time(non_existent_plate):
                test_logger_main.info("  УСПІХ (очікувано).")
            else:
                test_logger_main.error("  ПОМИЛКА (неочікувано).")

            unauth_plate = f"UA{int(time.time()) % 10000}YY"
            test_logger_main.info(f"\nТест 3: Логування неавторизованої спроби для '{unauth_plate}'")
            if handler.add_unauthorized_attempt(unauth_plate):
                test_logger_main.info("  УСПІХ.")
            else:
                test_logger_main.warning("  ПОМИЛКА.")

            time.sleep(1)
            exit_plate = f"EX{int(time.time()) % 10000}ZZ"
            test_logger_main.info(f"\nТест 4: Логування виїзду для '{exit_plate}'")
            if handler.log_vehicle_exit(exit_plate):
                test_logger_main.info("  УСПІХ.")
            else:
                test_logger_main.warning("  ПОМИЛКА.")

    test_logger_main.info("\n--- Завершено тестування модуля sheet_handler.py ---")
    test_logger_main.info("Будь ласка, перевірте вашу Google Таблицю для верифікації результатів.")