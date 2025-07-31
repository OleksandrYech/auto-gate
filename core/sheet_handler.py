# core/sheet_handler.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import logging
from datetime import datetime
import os
from typing import Optional, List, Tuple

logger = logging.getLogger(__name__)

# --- Константи, що визначають структуру вашої Google Таблиці ---
# Ці значення мають відповідати вашій таблиці
DEFAULT_VEHICLES_SHEET_NAME: str = 'Vehicles' # Назва вашого аркуша

# Починати зчитування даних з цього рядка, щоб пропустити заголовки
ENTRY_DATA_START_ROW: int = 3

# Стовпці для авторизованих номерів (в'їзд)
ENTRY_PLATE_COL_A_NUM: int = 1
ENTRY_TIMESTAMP_COL_B_NUM: int = 2

# Стовпці для неавторизованих спроб
UNAUTHORIZED_PLATE_COL_D_NUM: int = 4
UNAUTHORIZED_TIMESTAMP_COL_E_NUM: int = 5

# Стовпці для логування виїзду
EXIT_PLATE_COL_G_NUM: int = 7
EXIT_TIMESTAMP_COL_H_NUM: int = 8


class SheetHandler:
    """Керує всіма операціями з Google Sheets."""
    def __init__(self, credentials_file_path: str, spreadsheet_url: str):
        self._logger = logging.getLogger(f"{__name__}.SheetHandler")
        self.spreadsheet_url = spreadsheet_url
        self.credentials_file = self._resolve_credentials_path(credentials_file_path)
        self._client: Optional[gspread.Client] = None
        self._active_worksheet: Optional[gspread.Worksheet] = None
        self.CYRILLIC_TO_LATIN_MAP = {
            'А': 'A', 'В': 'B', 'Е': 'E', 'І': 'I', 'К': 'K', 'М': 'M',
            'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C', 'Т': 'T', 'Х': 'X', 'У': 'Y'
        }
        self._connect_and_authorize()

    def _resolve_credentials_path(self, creds_path: str) -> str:
        """Визначає абсолютний шлях до файлу облікових даних."""
        if os.path.isabs(creds_path):
            return creds_path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", os.path.basename(creds_path))
        if os.path.exists(config_path):
            return config_path
        self._logger.warning(f"Файл облікових даних '{creds_path}' не знайдено.")
        return creds_path

    def _connect_and_authorize(self):
        """Підключається та авторизується з Google Sheets API."""
        if not os.path.exists(self.credentials_file):
            self._logger.error(f"Файл облікових даних '{self.credentials_file}' не знайдено.")
            return
        try:
            scopes = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_file, scopes)
            self._client = gspread.authorize(creds)
            self._logger.info("Успішно авторизовано з Google Sheets API.")
        except Exception as e:
            self._logger.error(f"Не вдалося авторизуватися з Google Sheets API: {e}", exc_info=True)

    def _get_worksheet(self) -> Optional[gspread.Worksheet]:
        """Отримує робочий аркуш з таблиці. Кешує результат."""
        if not self._client:
            self._logger.error("Клієнт Google Sheets не авторизований.")
            return None
        if self._active_worksheet:
            return self._active_worksheet
        try:
            spreadsheet = self._client.open_by_url(self.spreadsheet_url)
            self._active_worksheet = spreadsheet.worksheet(DEFAULT_VEHICLES_SHEET_NAME)
            return self._active_worksheet
        except Exception as e:
            self._logger.error(f"Помилка відкриття аркуша '{DEFAULT_VEHICLES_SHEET_NAME}': {e}", exc_info=True)
            return None

    def _transliterate_to_latin(self, text: str) -> str:
        """Перетворює схожі кириличні символи в номері на латинські."""
        if not text:
            return ""
        return "".join([self.CYRILLIC_TO_LATIN_MAP.get(char, char) for char in text.upper()])

    # --- Методи для основної логіки ---

    def find_vehicle_and_update_entry_time(self, plate_number: str) -> bool:
        """Шукає номер у списку, і якщо знайдено, оновлює час у сусідній комірці."""
        worksheet = self._get_worksheet()
        if not worksheet:
            return False
        try:
            recognized_plate_latin = self._transliterate_to_latin(plate_number)
            cell = worksheet.find(recognized_plate_latin, in_column=ENTRY_PLATE_COL_A_NUM)
            if cell:
                current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                worksheet.update_cell(cell.row, ENTRY_TIMESTAMP_COL_B_NUM, current_datetime)
                self._logger.info(f"Авто '{plate_number}' знайдено у рядку {cell.row}. Час в'їзду оновлено.")
                return True
            self._logger.info(f"Номер '{plate_number}' не знайдено у списку авторизованих.")
            return False
        except Exception as e:
            self._logger.error(f"Помилка пошуку/оновлення авто '{plate_number}': {e}", exc_info=True)
            return False

    def add_unauthorized_attempt(self, plate_number: str):
        """Додає запис про неавторизовану спробу в'їзду."""
        worksheet = self._get_worksheet()
        if not worksheet: return
        try:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.append_row(
                [plate_number, current_datetime],
                table_range=f'D{ENTRY_DATA_START_ROW}', # Починати запис з колонки D
                value_input_option='USER_ENTERED'
            )
            self._logger.info(f"Неавторизовану спробу '{plate_number}' залоговано.")
        except Exception as e:
            self._logger.error(f"Помилка логування неавторизованої спроби '{plate_number}': {e}", exc_info=True)

    def log_vehicle_exit(self, plate_number: str):
        """Логує виїзд автомобіля."""
        worksheet = self._get_worksheet()
        if not worksheet: return
        try:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            worksheet.append_row(
                [plate_number, current_datetime],
                table_range=f'G{ENTRY_DATA_START_ROW}', # Починати запис з колонки G
                value_input_option='USER_ENTERED'
            )
            self._logger.info(f"Виїзд '{plate_number}' залоговано.")
        except Exception as e:
            self._logger.error(f"Помилка логування виїзду '{plate_number}': {e}", exc_info=True)

    # --- Нові методи для Telegram-бота ---

    def add_authorized_vehicle(self, plate_number: str) -> Tuple[bool, str]:
        """Додає новий номер у список авторизованих."""
        worksheet = self._get_worksheet()
        if not worksheet:
            return False, "Не вдалося підключитися до таблиці."

        plate_number_upper = self._transliterate_to_latin(plate_number.upper())

        try:
            all_plates_in_col = worksheet.col_values(ENTRY_PLATE_COL_A_NUM)
            if plate_number_upper in all_plates_in_col:
                return False, f"Номер {plate_number_upper} вже є у списку."

            # Знаходимо перший порожній рядок, починаючи з заданого
            next_empty_row = len(all_plates_in_col) + 1
            if next_empty_row < ENTRY_DATA_START_ROW:
                next_empty_row = ENTRY_DATA_START_ROW

            worksheet.update_cell(next_empty_row, ENTRY_PLATE_COL_A_NUM, plate_number_upper)
            self._logger.info(f"Номер {plate_number_upper} успішно додано.")
            return True, f"✅ Номер {plate_number_upper} успішно додано."
        except Exception as e:
            self._logger.error(f"Помилка додавання номера {plate_number_upper}: {e}")
            return False, "Сталася помилка під час роботи з таблицею."

    def delete_authorized_vehicle(self, plate_number: str) -> Tuple[bool, str]:
        """Видаляє номер та час його останнього заїзду зі списку."""
        worksheet = self._get_worksheet()
        if not worksheet:
            return False, "Не вдалося підключитися до таблиці."

        plate_number_upper = self._transliterate_to_latin(plate_number.upper())

        try:
            cell = worksheet.find(plate_number_upper, in_column=ENTRY_PLATE_COL_A_NUM)
            if not cell:
                return False, f"Номер {plate_number_upper} не знайдено у списку."

            # Ефективно очищуємо комірку з номером та часом одним запитом
            worksheet.batch_clear([f'A{cell.row}', f'B{cell.row}'])
            self._logger.info(f"Номер {plate_number_upper} у рядку {cell.row} видалено.")
            return True, f"✅ Номер {plate_number_upper} видалено."
        except gspread.exceptions.CellNotFound:
             return False, f"Номер {plate_number_upper} не знайдено у списку."
        except Exception as e:
            self._logger.error(f"Помилка видалення номера {plate_number_upper}: {e}")
            return False, "Сталася помилка під час роботи з таблицею."

    def get_list_by_columns(self, cols: List[int]) -> List[List[str]]:
        """Допоміжна функція для отримання даних з кількох колонок для звітів."""
        worksheet = self._get_worksheet()
        if not worksheet:
            return []
        try:
            all_data = worksheet.get_all_values()
            # Пропускаємо заголовки
            data_without_header = all_data[ENTRY_DATA_START_ROW - 1:]

            result = []
            for row in data_without_header:
                # Збираємо дані з потрібних колонок, перевіряючи довжину рядка
                row_data = [row[col - 1] for col in cols if len(row) >= col and row[col - 1].strip()]
                # Додаємо, тільки якщо хоча б одна потрібна комірка не порожня
                if len(row_data) > 0 and any(cell for cell in row_data):
                     result.append(row_data)
            return result
        except Exception as e:
            self._logger.error(f"Помилка отримання списку для колонок {cols}: {e}")
            return []
