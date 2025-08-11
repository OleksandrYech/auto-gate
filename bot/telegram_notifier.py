# bot/telegram_notifier.py
import asyncio
import logging
from typing import Optional

import telegram

logger = logging.getLogger(__name__)

def escape_markdown(text: str) -> str:
    """
    Екранує спеціальні символи для формату Telegram MarkdownV2.
    Перетворює вхідні дані на рядок для уникнення помилок.
    """
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return "".join(f'\\{char}' if char in escape_chars else char for char in str(text))


class TelegramNotifier:
    """
    Клас для надсилання сповіщень про події в Telegram.
    """
    def __init__(self, token: str, chat_id: int):
        self._logger = logging.getLogger(__name__)
        self.bot: Optional[telegram.Bot] = None
        self.chat_id: Optional[int] = None

        if not token or not chat_id:
            self._logger.error("Токен бота або ID чату не вказано. Нотифікатор буде вимкнено.")
            return

        try:
            self.bot = telegram.Bot(token=token)
            self.chat_id = chat_id
            self._logger.info("TelegramNotifier успішно ініціалізовано.")
        except Exception as e:
            self._logger.error(f"Не вдалося ініціалізувати telegram.Bot: {e}", exc_info=True)


    def send_notification(self, photo_path: str, plate_text: str, timestamp: str, status: str):
        """
        Публічний метод для надсилання сповіщення.
        Запускає асинхронну функцію надсилання в новому циклі подій.
        """
        if not self.bot or not self.chat_id:
            self._logger.warning("Спроба надіслати сповіщення, але нотифікатор не ініціалізовано.")
            return

        # Використовуємо asyncio.run для простого запуску async-функції з синхронного коду
        try:
            asyncio.run(self._send_async(photo_path, plate_text, timestamp, status))
        except Exception as e:
             self._logger.error(f"Помилка під час виконання asyncio.run для надсилання сповіщення: {e}", exc_info=True)


    async def _send_async(self, photo_path: str, plate_text: str, timestamp: str, status: str):
        """Асинхронна логіка надсилання фото та тексту."""

        # Екрануємо динамічні частини тексту для безпечного форматування
        safe_status = escape_markdown(status)
        safe_timestamp = escape_markdown(timestamp)

        # Визначаємо емодзі залежно від статусу
        status_emoji = "✅" if status == "Авторизовано" else "❌"

        # Формуємо підпис до фото, використовуючи MarkdownV2
        # Номер `plate_text` не екрануємо, бо він знаходиться всередині `monospace` блоку (`` `...` ``)
        caption = (
            f"{status_emoji} *{safe_status}*\n\n"
            f"🔢 *Номер:* `{plate_text}`\n"
            f"⏰ *Час:* {safe_timestamp}"
        )

        try:
            with open(photo_path, 'rb') as photo_file:
                await self.bot.send_photo(
                    chat_id=self.chat_id,
                    photo=photo_file,
                    caption=caption,
                    parse_mode='MarkdownV2'
                )
            self._logger.info(f"Сповіщення для '{plate_text}' успішно надіслано в Telegram.")
        except FileNotFoundError:
             self._logger.error(f"Файл фото не знайдено за шляхом: {photo_path}")
        except Exception as e:
            self._logger.error(f"Не вдалося надіслати сповіщення в Telegram: {e}", exc_info=True)
