# bot/telegram_notifier.py
import logging
import json
import os
import asyncio
import threading
import telegram
from typing import Set

logger = logging.getLogger(__name__)

AUTHORIZED_USERS_FILE = os.path.join(os.getcwd(), "config", "authorized_users.json")


def escape_markdown(text: str) -> str:
    """Екранує спеціальні символи MarkdownV2."""
    escape_chars = r'_*[]()~`>#+-=|{}.!'
    return ''.join(f'\\{char}' if char in escape_chars else char for char in text)


class AsyncNotifierThread(threading.Thread):
    """
    Окремий потік, який запускає та керує циклом подій asyncio.
    Це необхідно для безпечної інтеграції асинхронного коду в синхронний застосунок.
    """

    def __init__(self, token: str):
        super().__init__()
        self.loop = asyncio.new_event_loop()
        self.bot = telegram.Bot(token)
        self.daemon = True  # Дозволяє програмі завершуватися, навіть якщо цей потік ще працює

    def run(self):
        """Запускає цикл подій у цьому потоці."""
        asyncio.set_event_loop(self.loop)
        self.loop.run_forever()

    def submit_task(self, coro):
        """Безпечно додає асинхронну задачу для виконання в циклі подій."""
        return asyncio.run_coroutine_threadsafe(coro, self.loop)

    def stop(self):
        """Зупиняє цикл подій та потік."""
        self.loop.call_soon_threadsafe(self.loop.stop)


class TelegramNotifier:
    """Клас для надсилання сповіщень, що використовує фоновий потік."""

    def __init__(self, token: str):
        self.notifier_thread = AsyncNotifierThread(token)
        self.notifier_thread.start()
        logger.info("TelegramNotifier та фоновий потік для відправки успішно ініціалізовано.")

    def _load_users_from_file(self) -> Set[int]:
        if not os.path.exists(AUTHORIZED_USERS_FILE): return set()
        try:
            with open(AUTHORIZED_USERS_FILE, 'r') as f:
                return set(json.load(f))
        except (IOError, json.JSONDecodeError):
            return set()

    async def _send_all_async(self, photo_path: str, caption: str, users_to_notify: Set[int]):
        """Асинхронна функція для розсилки повідомлень."""
        tasks = []
        for chat_id in users_to_notify:
            try:
                # ВАЖЛИВО: Файл потрібно відкривати тут, всередині асинхронної функції
                photo_file = open(photo_path, 'rb')
                tasks.append(self.notifier_thread.bot.send_photo(
                    chat_id=chat_id,
                    photo=photo_file,
                    caption=caption,
                    parse_mode='MarkdownV2'
                ))
            except Exception as e:
                logger.error(f"Помилка підготовки до відправки для чату {chat_id}: {e}")

        if not tasks: return

        results = await asyncio.gather(*tasks, return_exceptions=True)

        # Аналізуємо результати та закриваємо файли
        for i, result in enumerate(results):
            chat_id = list(users_to_notify)[i]
            # Закриваємо файл, який передавався у відповідну задачу
            tasks[i].cr_frame.f_locals['photo'].close()
            if isinstance(result, Exception):
                logger.error(f"Помилка надсилання повідомлення до чату {chat_id}: {result}")
            else:
                logger.debug(f"Повідомлення успішно надіслано до чату {chat_id}.")

    def send_notification_to_authorized(self, photo_path: str, plate: str, timestamp: str, status: str):
        """Синхронна функція, яка передає задачу на виконання у фоновий потік."""
        users_to_notify = self._load_users_from_file()
        if not users_to_notify:
            logger.warning("Немає авторизованих користувачів для відправки сповіщень.")
            return

        logger.info(f"Передача задачі на відправку сповіщення для {len(users_to_notify)} користувачів...")
        caption = (f"🚗 *Розпізнано номер:* `{escape_markdown(plate)}`\n"
                   f"🕒 *Час:* {escape_markdown(timestamp)}\n"
                   f"🚦 *Статус:* {escape_markdown(status)}")

        # Передаємо асинхронну функцію на виконання у фоновий потік
        self.notifier_thread.submit_task(
            self._send_all_async(photo_path, caption, users_to_notify)
        )

    def cleanup(self):
        """Коректно зупиняє фоновий потік."""
        logger.info("Зупинка фонового потоку TelegramNotifier...")
        self.notifier_thread.stop()