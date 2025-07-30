# core/sheet_handler.py
import gspread
from oauth2client.service_account import ServiceAccountCredentials
import logging
from datetime import datetime
import os
from typing import Optional, List

logger = logging.getLogger(__name__)

# Перевіряємо, чи запущено проєкт у режимі симуляції
SIMULATION_MODE = os.environ.get('SIMULATION_MODE') == '1'
# Список авторизованих номерів для режиму симуляції
SIMULATION_AUTHORIZED_PLATES = ['ВС2441OM', 'BH5678CE']

# --- Константи, що визначають структуру вашої Google Таблиці ---
DEFAULT_SCOPES: List[str] = ['https://www.googleapis.com/auth/spreadsheets', 'https://www.googleapis.com/auth/drive.file']
DEFAULT_CREDENTIALS_FILE: str = 'credentials.json'
DEFAULT_VEHICLES_SHEET_NAME: str = 'Vehicles'

ENTRY_PLATE_COL_A_NUM: int = 1
ENTRY_TIMESTAMP_COL_B_NUM: int = 2
ENTRY_DATA_START_ROW: int = 3
UNAUTHORIZED_PLATE_COL_D_NUM: int = 4
UNAUTHORIZED_TIMESTAMP_COL_E_NUM: int = 5
EXIT_PLATE_COL_G_NUM: int = 7
EXIT_TIMESTAMP_COL_H_NUM: int = 8


class SheetHandler:
    def __init__(self, credentials_file_path: str, spreadsheet_url: str = ""):
        self._logger = logging.getLogger(f"{__name__}.SheetHandler")

        if SIMULATION_MODE:
            self._logger.warning("SheetHandler працює в РЕЖИМІ СИМУЛЯЦІЇ. Жодних звернень до Google API не буде.")
            self._client = None
            self._active_worksheet = None
        else:
            self.spreadsheet_url = spreadsheet_url
            self.credentials_file = self._resolve_credentials_path(credentials_file_path)
            self._client: Optional[gspread.Client] = None
            self._active_worksheet: Optional[gspread.Worksheet] = None
            self._connect_and_authorize()

        self.CYRILLIC_TO_LATIN_MAP = {
            'А': 'A', 'В': 'B', 'Е': 'E', 'І': 'I', 'К': 'K',
            'М': 'M', 'Н': 'H', 'О': 'O', 'Р': 'P', 'С': 'C',
            'Т': 'T', 'Х': 'X', 'У': 'Y'
        }

    # ... (методи _resolve_credentials_path, _connect_and_authorize, _get_worksheet, _transliterate_to_latin залишаються без змін) ...
    def _resolve_credentials_path(self, creds_path: str) -> str:
        if os.path.isabs(creds_path): return creds_path
        project_root = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        config_path = os.path.join(project_root, "config", os.path.basename(creds_path))
        if os.path.exists(config_path): return config_path
        self._logger.warning(f"Файл облікових даних '{creds_path}' не знайдено.")
        return creds_path

    def _connect_and_authorize(self):
        if not os.path.exists(self.credentials_file):
            self._logger.error(f"Файл облікових даних '{self.credentials_file}' не знайдено.")
            return
        try:
            creds = ServiceAccountCredentials.from_json_keyfile_name(self.credentials_file, DEFAULT_SCOPES)
            self._client = gspread.authorize(creds)
            self._logger.info("Успішно авторизовано з Google Sheets API.")
        except Exception as e:
            self._logger.error(f"Не вдалося авторизуватися з Google Sheets API: {e}", exc_info=True)

    def _get_worksheet(self) -> Optional[gspread.Worksheet]:
        if not self._client: return None
        if self._active_worksheet: return self._active_worksheet
        try:
            spreadsheet = self._client.open_by_url(self.spreadsheet_url)
            self._active_worksheet = spreadsheet.worksheet(DEFAULT_VEHICLES_SHEET_NAME)
            return self._active_worksheet
        except Exception as e:
            self._logger.error(f"Помилка відкриття аркуша: {e}", exc_info=True)
            return None

    def _transliterate_to_latin(self, text: str) -> str:
        if not text: return ""
        text_upper = text.upper()
        return "".join([self.CYRILLIC_TO_LATIN_MAP.get(char, char) for char in text_upper])


    def find_vehicle_and_update_entry_time(self, plate_number: str) -> bool:
        """
        У режимі симуляції перевіряє по фіксованому списку.
        В іншому випадку - шукає номер у колонці A та оновлює час у колонці B.
        """
        if SIMULATION_MODE:
            recognized_plate_latin = self._transliterate_to_latin(plate_number)
            if recognized_plate_latin in SIMULATION_AUTHORIZED_PLATES:
                self._logger.info(f"[СИМУЛЯЦІЯ] Авто '{plate_number}' знайдено у списку авторизованих.")
                return True
            else:
                self._logger.info(f"[СИМУЛЯЦІЯ] Номер '{plate_number}' не знайдено у списку авторизованих.")
                return False

        # --- Логіка для реальної роботи ---
        worksheet = self._get_worksheet()
        if not worksheet: return False
        try:
            recognized_plate_latin = self._transliterate_to_latin(plate_number)
            self._logger.debug(f"Пошук номера '{plate_number}' (трансліт: '{recognized_plate_latin}')")
            authorized_plates_raw = worksheet.col_values(ENTRY_PLATE_COL_A_NUM)
            for i, plate_from_sheet in enumerate(authorized_plates_raw):
                row_number = i + 1
                if row_number < ENTRY_DATA_START_ROW: continue
                plate_from_sheet_latin = self._transliterate_to_latin(plate_from_sheet)
                if recognized_plate_latin == plate_from_sheet_latin:
                    current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
                    worksheet.update_cell(row_number, ENTRY_TIMESTAMP_COL_B_NUM, current_datetime)
                    self._logger.info(f"Авто '{plate_number}' знайдено у рядку {row_number}. Час в'їзду оновлено.")
                    return True
            self._logger.info(f"Номер '{plate_number}' не знайдено у списку авторизованих.")
            return False
        except Exception as e:
            self._logger.error(f"Помилка пошуку/оновлення авто '{plate_number}': {e}", exc_info=True)
            return False

    def add_unauthorized_attempt(self, plate_number: str):
        if SIMULATION_MODE:
            self._logger.info(f"[СИМУЛЯЦІЯ] Записано неавторизовану спробу для номера '{plate_number}'.")
            return

        worksheet = self._get_worksheet()
        # ... (решта коду без змін) ...
        if not worksheet: return
        try:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            col_values = worksheet.col_values(UNAUTHORIZED_PLATE_COL_D_NUM)
            next_row = max(len(col_values) + 1, ENTRY_DATA_START_ROW)
            worksheet.update_cell(next_row, UNAUTHORIZED_PLATE_COL_D_NUM, plate_number)
            worksheet.update_cell(next_row, UNAUTHORIZED_TIMESTAMP_COL_E_NUM, current_datetime)
            self._logger.info(f"Неавторизовану спробу '{plate_number}' залоговано у рядок {next_row}.")
        except Exception as e:
            self._logger.error(f"Помилка логування неавторизованої спроби '{plate_number}': {e}", exc_info=True)


    def log_vehicle_exit(self, plate_number: str):
        if SIMULATION_MODE:
            self._logger.info(f"[СИМУЛЯЦІЯ] Записано виїзд для номера '{plate_number}'.")
            return

        worksheet = self._get_worksheet()
        # ... (решта коду без змін) ...
        if not worksheet: return
        try:
            current_datetime = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
            col_values = worksheet.col_values(EXIT_PLATE_COL_G_NUM)
            next_row = max(len(col_values) + 1, ENTRY_DATA_START_ROW)
            worksheet.update_cell(next_row, EXIT_PLATE_COL_G_NUM, plate_number)
            worksheet.update_cell(next_row, EXIT_TIMESTAMP_COL_H_NUM, current_datetime)
            self._logger.info(f"Виїзд '{plate_number}' залоговано у рядок {next_row}.")
        except Exception as e:
            self._logger.error(f"Помилка логування виїзду '{plate_number}': {e}", exc_info=True)
