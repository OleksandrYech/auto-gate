# core/sheet_handler.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials  # Старіший варіант
import logging
from datetime import datetime
import os
from typing import Optional, List  # Додано List для SCOPES

# --- Глобальна конфігурація логування ---
# logger = logging.getLogger(__name__)

# --- Константи ---
DEFAULT_SCOPES: List[str] = ['https://www.googleapis.com/auth/spreadsheets',
                             'https://www.googleapis.com/auth/drive.file']
DEFAULT_CREDENTIALS_FILE: str = 'credentials.json'  # Розмістіть у config/ або передайте шлях

# !!! ВАЖЛИВО: Замініть це на URL вашої Google Таблиці або передайте через конструктор !!!
DEFAULT_SPREADSHEET_URL: str = 'https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0'  # Наприклад, той, що був у sheets.py

# Назва аркуша
DEFAULT_VEHICLES_SHEET_NAME: str = 'Vehicles'

# В'їзд авторизованих: Стовпець A (Номер), Стовпець B (Час останнього в'їзду)
ENTRY_PLATE_COL_A_NUM: int = 1
ENTRY_TIMESTAMP_COL_B_NUM: int = 2
ENTRY_DATA_START_ROW: int = 3

# Неавторизовані спроби: Стовпець D (Номер), Стовпець E (Час спроби)
UNAUTHORIZED_PLATE_COL_D_NUM: int = 4
UNAUTHORIZED_TIMESTAMP_COL_E_NUM: int = 5
UNAUTHORIZED_DATA_START_ROW: int = 3

# Лог виїзду: Стовпець G ("Номер"), Стовпець H ("Останній Виїзд")
EXIT_PLATE_COL_G_NUM: int = 7
EXIT_TIMESTAMP_COL_H_NUM: int = 8
EXIT_DATA_START_ROW: int = 3


class SheetHandler:
    """
    Клас для взаємодії з Google Sheets для авторизації транспортних засобів
    та логування подій в'їзду/виїзду.
    """

    def __init__(self,
                 spreadsheet_url: str = DEFAULT_SPREADSHEET_URL,
                 credentials_file: str = DEFAULT_CREDENTIALS_FILE,
                 scopes: List[str] = None,  # Використовуватиме DEFAULT_SCOPES, якщо None
                 vehicles_sheet_name: str = DEFAULT_VEHICLES_SHEET_NAME):
        """
        Ініціалізація SheetHandler.

        Args:
            spreadsheet_url (str): URL Google Таблиці.
            credentials_file (str): Шлях до файлу облікових даних JSON.
            scopes (List[str]): Список областей доступу для Google API.
            vehicles_sheet_name (str): Назва основного аркуша в таблиці.
        """
        self._logger = logging.getLogger(f"{__name__}.SheetHandler")

        self.spreadsheet_url = spreadsheet_url
        self.credentials_file = credentials_file
        self.scopes = scopes if scopes is not None else DEFAULT_SCOPES  # Використовуємо копію списку за замовчуванням
        self.vehicles_sheet_name = vehicles_sheet_name

        self._client: Optional[gspread.Client] = None
        self._active_worksheet: Optional[gspread.Worksheet] = None  # Можна кешувати активний аркуш

        if self.spreadsheet_url == 'YOUR_SPREADSHEET_URL_HERE':
            self._logger.critical("URL Google Таблиці не встановлено! Будь ласка, встановіть spreadsheet_url.")
            # Можна кинути виняток або залишити як є, але методи не працюватимуть
            # raise ValueError("URL Google Таблиці не встановлено.")

        # Спроба підключитися при ініціалізації
        self._connect_and_authorize()

    def _connect_and_authorize(self) -> bool:
        """Встановлює з'єднання та авторизується з Google Sheets API."""
        if self._client:  # Якщо клієнт вже існує
            return True

        self._logger.info(f"Спроба підключення до Google Sheets API через {self.credentials_file}...")
        try:
            # Перевірка існування файлу облікових даних
            if not os.path.exists(self.credentials_file):
                self._logger.error(
                    f"Файл облікових даних '{self.credentials_file}' не знайдено за шляхом: {os.path.abspath(self.credentials_file)}")
                # Спробувати знайти у директорії config/, якщо шлях не абсолютний
                config_dir_path = os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "config",
                                               self.credentials_file)
                if os.path.exists(config_dir_path):
                    self._logger.info(f"Знайдено файл облікових даних у config/: {config_dir_path}")
                    self.credentials_file = config_dir_path
                else:
                    self._logger.error(f"Файл облікових даних також не знайдено у: {config_dir_path}")
                    return False

            creds = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_file, self.scopes)
            self._client = gspread.authorize(creds)
            self._logger.info("Успішно підключено та авторизовано з Google Sheets API.")
            return True
        except FileNotFoundError:  # Ця помилка вже оброблена вище, але для повноти
            self._logger.error(f"Файл облікових даних '{self.credentials_file}' не знайдено (повторна перевірка).")
            return False
        except Exception as e:
            self._logger.error(f"Не вдалося авторизуватися або підключитися до Google Sheets API: {e}", exc_info=True)
            return False

    def _get_worksheet(self, sheet_name: Optional[str] = None) -> Optional[gspread.Worksheet]:
        """
        Допоміжна функція для отримання конкретного аркуша.
        Якщо sheet_name не вказано, використовує self.vehicles_sheet_name.
        """
        if not self._client:
            self._logger.warning("Клієнт Google Sheets не ініціалізовано. Спроба повторного підключення.")
            if not self._connect_and_authorize():
                return None

        target_sheet_name = sheet_name if sheet_name is not None else self.vehicles_sheet_name

        # Перевірка, чи ми вже маємо цей аркуш (просте кешування)
        if self._active_worksheet and self._active_worksheet.title == target_sheet_name:
            return self._active_worksheet

        try:
            if not self.spreadsheet_url or self.spreadsheet_url == 'YOUR_SPREADSHEET_URL_HERE':
                self._logger.error("URL таблиці не налаштовано. Встановіть значення spreadsheet_url.")
                return None

            spreadsheet = self._client.open_by_url(self.spreadsheet_url)
            worksheet = spreadsheet.worksheet(target_sheet_name)
            self._active_worksheet = worksheet  # Кешуємо
            return worksheet
        except gspread.exceptions.SpreadsheetNotFound:
            self._logger.error(f"Таблицю не знайдено за URL: {self.spreadsheet_url}.")
        except gspread.exceptions.WorksheetNotFound:
            self._logger.error(f"Аркуш '{target_sheet_name}' не знайдено в таблиці.")
        except Exception as e:
            self._logger.error(f"Помилка відкриття аркуша '{target_sheet_name}': {e}", exc_info=True)

        self._active_worksheet = None  # Скидаємо кеш у разі помилки
        return None

    def find_vehicle_and_update_entry_time(self, plate_number: str) -> bool:
        """
        Знаходить номерний знак в аркуші (Стовпець A).
        Якщо знайдено, оновлює час останнього в'їзду в Стовпці B.
        """
        worksheet = self._get_worksheet()  # Використовує self.vehicles_sheet_name
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
                pass  # findall поверне порожній список

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

    def add_unauthorized_attempt(self, plate_number: str) -> bool:
        """
        Додає запис про неавторизовану спробу в'їзду (Стовпці D та E).
        """
        worksheet = self._get_worksheet()
        if not worksheet:
            self._logger.error(
                f"Не вдалося отримати аркуш '{self.vehicles_sheet_name}' для логування неавторизованої спроби '{plate_number}'.")
            return False

        try:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            # Знаходимо наступний порожній рядок
            col_d_values = worksheet.col_values(UNAUTHORIZED_PLATE_COL_D_NUM)
            next_row_to_write = UNAUTHORIZED_DATA_START_ROW

            start_index_for_search = UNAUTHORIZED_DATA_START_ROW - 1
            if start_index_for_search < len(col_d_values):
                for i in range(start_index_for_search, len(col_d_values)):
                    if not col_d_values[i]:
                        next_row_to_write = i + 1
                        break
                else:
                    next_row_to_write = len(col_d_values) + 1

            if next_row_to_write < UNAUTHORIZED_DATA_START_ROW:  # Перестраховка
                next_row_to_write = UNAUTHORIZED_DATA_START_ROW

            worksheet.update_cell(next_row_to_write, UNAUTHORIZED_PLATE_COL_D_NUM, plate_number)
            worksheet.update_cell(next_row_to_write, UNAUTHORIZED_TIMESTAMP_COL_E_NUM, current_datetime)

            self._logger.info(f"Неавторизовану спробу '{plate_number}' залоговано о {current_datetime} "
                              f"в аркуші '{self.vehicles_sheet_name}', рядок {next_row_to_write} (стовпці D,E).")
            return True
        except Exception as e:
            self._logger.error(f"Помилка додавання неавторизованої спроби для '{plate_number}': {e}", exc_info=True)
            return False

    def log_vehicle_exit(self, plate_number: str) -> bool:
        """
        Записує номерний знак та час виїзду автомобіля (Стовпці G та H).
        """
        worksheet = self._get_worksheet()
        if not worksheet:
            self._logger.error(
                f"Не вдалося отримати аркуш '{self.vehicles_sheet_name}' для логування виїзду '{plate_number}'.")
            return False

        try:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
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

            if next_row_to_write < EXIT_DATA_START_ROW:  # Перестраховка
                next_row_to_write = EXIT_DATA_START_ROW

            worksheet.update_cell(next_row_to_write, EXIT_PLATE_COL_G_NUM, plate_number)
            worksheet.update_cell(next_row_to_write, EXIT_TIMESTAMP_COL_H_NUM, current_datetime)

            self._logger.info(f"Виїзд автомобіля залоговано: Номер '{plate_number}', Час {current_datetime} "
                              f"в аркуші '{self.vehicles_sheet_name}', рядок {next_row_to_write} (стовпці G,H).")
            return True
        except Exception as e:
            self._logger.error(f"Помилка логування виїзду авто для номера '{plate_number}': {e}", exc_info=True)
            return False


# --- Блок для тестування модуля ---
if __name__ == '__main__':
    # Налаштування базового логування для виводу в консоль під час тестування
    # У реальному проекті це буде робитися централізовано через logger_config.py
    if not logging.getLogger().hasHandlers():  # Перевіряємо, чи вже є обробники
        logging.basicConfig(
            level=logging.DEBUG,
            format='%(asctime)s - %(name)s - [%(levelname)s] - %(module)s:%(lineno)d - %(message)s',
            handlers=[logging.StreamHandler()]  # Тільки консоль для тесту модуля
        )

    test_logger = logging.getLogger(__name__)  # Використовуємо __name__ для коректного імені логгера
    test_logger.info("--- Тестування модуля sheet_handler.py ---")

    # !!! ВАЖЛИВО: Для тестування потрібно вказати реальний URL вашої таблиці
    # та переконатися, що файл credentials.json існує і налаштований.
    TEST_SPREADSHEET_URL = 'https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0'  # ЗАМІНІТЬ НА ВАШ URL
    TEST_CREDENTIALS_FILE = 'config/credentials.json'  # Припускаємо, що він у config/ відносно кореня проекту

    if TEST_SPREADSHEET_URL == 'YOUR_SPREADSHEET_URL_HERE':
        test_logger.critical("КРИТИЧНО: TEST_SPREADSHEET_URL не встановлено для тестування sheet_handler.py.")
    elif not os.path.exists(TEST_CREDENTIALS_FILE):
        abs_path = os.path.abspath(TEST_CREDENTIALS_FILE)
        test_logger.critical(
            f"КРИТИЧНО: Файл облікових даних '{TEST_CREDENTIALS_FILE}' не знайдено за шляхом: {abs_path}.")
    else:
        test_logger.info(f"Використовується URL таблиці: {TEST_SPREADSHEET_URL}")
        test_logger.info(f"Очікується файл облікових даних: {os.path.abspath(TEST_CREDENTIALS_FILE)}")

        # Створюємо екземпляр SheetHandler
        handler = SheetHandler(
            spreadsheet_url=TEST_SPREADSHEET_URL,
            credentials_file=TEST_CREDENTIALS_FILE
            # Інші параметри (scopes, sheet_name) будуть використані за замовчуванням
        )

        if not handler._client:  # Перевіряємо, чи вдалося підключитися
            test_logger.error("Не вдалося отримати клієнт для Google Sheets. Подальші тести неможливі.")
        else:
            test_logger.info("Клієнт для Google Sheets успішно отримано.")

            # --- Тест: Пошук та оновлення в'їзду ---
            # !!! ЗАМІНІТЬ "EXISTING_PLATE_IN_COL_A" на номер, що ТОЧНО є у Стовпці A (з 3-го рядка) !!!
            test_plate_entry_allowed = "EXISTING_PLATE_IN_COL_A"
            test_logger.info(f"\nТест 1: Пошук та оновлення в'їзду для '{test_plate_entry_allowed}'")
            if handler.find_vehicle_and_update_entry_time(test_plate_entry_allowed):
                test_logger.info(f"  УСПІХ: Час в'їзду для '{test_plate_entry_allowed}' оновлено.")
            else:
                test_logger.warning(
                    f"  ПОМИЛКА/НЕ ЗНАЙДЕНО: '{test_plate_entry_allowed}'. Перевірте номер та права доступу.")

            # --- Тест: Спроба знайти неіснуючий номер для в'їзду ---
            # Генеруємо унікальний неіснуючий номер
            import time as time_module  # Щоб уникнути конфлікту з datetime.time

            test_plate_entry_non_existent = f"NONEXISTENT_{int(time_module.time()) % 10000}"
            test_logger.info(f"\nТест 2: Пошук неіснуючого номера '{test_plate_entry_non_existent}' для в'їзду")
            if not handler.find_vehicle_and_update_entry_time(test_plate_entry_non_existent):
                test_logger.info(f"  УСПІХ (очікувано): Номер '{test_plate_entry_non_existent}' не знайдено.")
            else:
                test_logger.error(f"  ПОМИЛКА (неочікувано): Номер '{test_plate_entry_non_existent}' знайдено.")

            # --- Тест: Додавання неавторизованої спроби ---
            test_plate_unauthorized = f"UNAUTH_{int(time_module.time()) % 10000}"
            test_logger.info(f"\nТест 3: Логування неавторизованої спроби для '{test_plate_unauthorized}'")
            if handler.add_unauthorized_attempt(test_plate_unauthorized):
                test_logger.info(f"  УСПІХ: Спроба '{test_plate_unauthorized}' залогована (Стовпці D,E).")
            else:
                test_logger.warning(f"  ПОМИЛКА: Не вдалося залогувати спробу '{test_plate_unauthorized}'.")

            time_module.sleep(1)  # Щоб уникнути однакових міток часу
            test_plate_unauthorized_2 = f"UNAUTH_NEXT_{int(time_module.time()) % 10000}"
            test_logger.info(f"\nТест 4: Логування ще однієї неавторизованої спроби '{test_plate_unauthorized_2}'")
            if handler.add_unauthorized_attempt(test_plate_unauthorized_2):
                test_logger.info(f"  УСПІХ: Спроба '{test_plate_unauthorized_2}' залогована.")
            else:
                test_logger.warning(f"  ПОМИЛКА: Не вдалося залогувати спробу '{test_plate_unauthorized_2}'.")

            # --- Тест: Логування виїзду автомобіля ---
            test_plate_exit = f"EXITCAR_{int(time_module.time()) % 10000}"
            test_logger.info(f"\nТест 5: Логування виїзду для '{test_plate_exit}'")
            if handler.log_vehicle_exit(test_plate_exit):
                test_logger.info(f"  УСПІХ: Виїзд '{test_plate_exit}' залоговано (Стовпці G,H).")
            else:
                test_logger.warning(f"  ПОМИЛКА: Не вдалося залогувати виїзд '{test_plate_exit}'.")

            time_module.sleep(1)
            test_plate_exit_2 = f"EXITNEXT_{int(time_module.time()) % 10000}"
            test_logger.info(f"\nТест 6: Логування ще одного виїзду '{test_plate_exit_2}'")
            if handler.log_vehicle_exit(test_plate_exit_2):
                test_logger.info(f"  УСПІХ: Виїзд '{test_plate_exit_2}' залоговано.")
            else:
                test_logger.warning(f"  ПОМИЛКА: Не вдалося залогувати виїзд '{test_plate_exit_2}'.")

    test_logger.info("\n--- Завершено тестування модуля sheet_handler.py ---")
    test_logger.info("Будь ласка, перевірте вашу Google Таблицю для верифікації результатів.")