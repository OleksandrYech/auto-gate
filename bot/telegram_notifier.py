# bot/telegram_notifier.py
import telegram
import asyncio
import logging

def escape_markdown(text: str) -> str:
    """Екранує спеціальні символи для формату Telegram MarkdownV2."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return "".join(f'\\{char}' if char in escape_chars else char for char in str(text))


class TelegramNotifier:
    def __init__(self, token: str, chat_id: int):
        self._logger = logging.getLogger(__name__)
        if not token or not chat_id:
            self._logger.error("Токен бота або ID чату не вказано. Нотифікатор вимкнено.")
            self.bot = None
            return

        self.bot = telegram.Bot(token=token)
        self.chat_id = chat_id
        self._logger.info("TelegramNotifier ініціалізовано.")

    def send_notification(self, photo_path: str, plate_text: str, timestamp: str, status: str):
        """Асинхронно надсилає фото з підписом."""
        if not self.bot:
            return
        asyncio.run(self._send_async(photo_path, plate_text, timestamp, status))

    async def _send_async(self, photo_path: str, plate_text: str, timestamp: str, status: str):
        """Асинхронна логіка надсилання."""

        # Застосовуємо екранування до динамічних частин тексту
        safe_status = escape_markdown(status)
        safe_timestamp = escape_markdown(timestamp)
        # plate_text не екрануємо, бо він знаходиться всередині `monospace` блоку (зворотні лапки)

        status_emoji = "✅" if status == "Авторизовано" else "❌"

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
            self._logger.info(f"Сповіщення для '{plate_text}' надіслано в Telegram.")
        except Exception as e:
            self._logger.error(f"Не вдалося надіслати сповіщення в Telegram: {e}", exc_info=True)
