# bot/bot_main.py
import logging
import os
import sys
from telegram import Update
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
)

# Додаємо шлях до кореневого каталогу, щоб імпортувати модулі з `core` та `utils`
# Це робить скрипт більш стійким до того, звідки його запускають
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.sheet_handler import SheetHandler
# Імпортуємо функцію екранування з telegram_notifier, щоб уникнути дублювання коду
from bot.telegram_notifier import escape_markdown


# --- КОНФІГУРАЦІЯ ---
# Рекомендується завантажувати ці дані зі змінних середовища
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
BOT_PASSWORD = os.getenv("BOT_PASSWORD", "")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0")
SHEETS_CREDENTIALS_PATH = os.path.join(os.getcwd(), "config", "credentials.json")


# --- ГЛОБАЛЬНІ ЗМІННІ ---
# Стан для діалогу авторизації
PASSWORD_ENTRY = 0
# Множина для зберігання ID авторизованих чатів
AUTHORIZED_USERS = set()

# Ініціалізуємо обробник таблиць один раз при старті
sheet_handler = None
try:
    if "YOUR_SPREADSHEET_URL_HERE" in SPREADSHEET_URL or not SPREADSHEET_URL:
        logging.warning("URL для Google Sheets не налаштовано. Бот не зможе працювати з таблицею.")
    else:
        sheet_handler = SheetHandler(
            credentials_file_path=SHEETS_CREDENTIALS_PATH,
            spreadsheet_url=SPREADSHEET_URL
        )
except Exception as e:
    logging.critical(f"Не вдалося ініціалізувати SheetHandler: {e}", exc_info=True)


# --- ЛОГІКА БОТА ---

def auth_required(func):
    """Декоратор для перевірки, чи авторизований користувач."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if update.effective_chat.id not in AUTHORIZED_USERS:
            await update.message.reply_text("⛔️ Доступ заборонено\\. Будь ласка, авторизуйтесь за допомогою /start\\.", parse_mode='MarkdownV2')
            return
        # --- ЗМІНА ---
        # Перевірку на `sheet_handler` тут видалено, оскільки вона тепер відбувається на старті
        # --- КІНЕЦЬ ЗМІНИ ---
        return await func(update, context, *args, **kwargs)
    return wrapper

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє команду /start, розпочинаючи діалог авторизації."""
    if update.effective_chat.id in AUTHORIZED_USERS:
        await update.message.reply_text("✅ Ви вже авторизовані\\. Доступні команди: /help", parse_mode='MarkdownV2')
        return ConversationHandler.END
    await update.message.reply_text("👋 Вітаю\\! Для доступу до системи, будь ласка, введіть пароль\\.", parse_mode='MarkdownV2')
    return PASSWORD_ENTRY

async def check_password(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Перевіряє введений пароль."""
    if update.message.text == BOT_PASSWORD:
        AUTHORIZED_USERS.add(update.effective_chat.id)
        logging.info(f"Користувач {update.effective_user.name} (ID: {update.effective_chat.id}) успішно авторизувався.")
        await update.message.reply_text("✅ Доступ дозволено\\! /help для перегляду списку команд\\.", parse_mode='MarkdownV2')
        return ConversationHandler.END
    else:
        logging.warning(f"Невдала спроба авторизації від користувача {update.effective_user.name} (ID: {update.effective_chat.id}).")
        await update.message.reply_text("⛔️ Неправильний пароль\\. Для повторної спроби введіть /start\\.", parse_mode='MarkdownV2')
        return ConversationHandler.END

@auth_required
async def help_command(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Надсилає довідкове повідомлення з переліком команд."""
    text = (
        "**Доступні команди:**\n\n"
        "/add `НОМЕР` \\- Додати номер у білий список\n"
        "/delete `НОМЕР` \\- Видалити номер з білого списку\n\n"
        "/allowed \\- Показати білий список\n"
        "/unauthorized \\- Показати неавторизовані спроби\n"
        "/departed \\- Показати журнал виїздів"
    )
    await update.message.reply_text(text, parse_mode='MarkdownV2')

@auth_required
async def add_plate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Додає номер у список дозволених."""
    if not context.args:
        await update.message.reply_text("Будь ласка, вкажіть номер\\. Приклад: `/add AA1234BB`", parse_mode='MarkdownV2')
        return
    plate_number = context.args[0]
    success, message = sheet_handler.add_authorized_vehicle(plate_number)
    await update.message.reply_text(escape_markdown(message))

@auth_required
async def delete_plate(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Видаляє номер зі списку дозволених."""
    if not context.args:
        await update.message.reply_text("Будь ласка, вкажіть номер\\. Приклад: `/delete AA1234BB`", parse_mode='MarkdownV2')
        return
    plate_number = context.args[0]
    success, message = sheet_handler.delete_authorized_vehicle(plate_number)
    await update.message.reply_text(escape_markdown(message))

@auth_required
async def get_allowed_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Надсилає список дозволених номерів."""
    data = sheet_handler.get_list_by_columns([1]) # Колонка А
    if not data:
        await update.message.reply_text("Список дозволених номерів порожній\\.", parse_mode='MarkdownV2')
        return

    message = "📄 *Список дозволених номерів:*\n" + "\n".join([f"`{row[0]}`" for row in data])
    await update.message.reply_text(message, parse_mode='MarkdownV2')

@auth_required
async def get_unauthorized_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Надсилає список неавторизованих спроб."""
    data = sheet_handler.get_list_by_columns([4, 5]) # Колонки D, E
    if not data:
        await update.message.reply_text("Список неавторизованих спроб порожній\\.", parse_mode='MarkdownV2')
        return

    message_lines = [f"`{row[0]}` \\- {escape_markdown(row[1])}" for row in data if len(row) == 2]
    message = "📄 *Неавторизовані спроби:*\n" + "\n".join(message_lines)
    await update.message.reply_text(message, parse_mode='MarkdownV2')

@auth_required
async def get_departed_list(update: Update, context: ContextTypes.DEFAULT_TYPE):
    """Надсилає список автомобілів, що виїхали."""
    data = sheet_handler.get_list_by_columns([7, 8]) # Колонки G, H
    if not data:
        await update.message.reply_text("Список виїздів порожній\\.", parse_mode='MarkdownV2')
        return

    message_lines = [f"`{row[0]}` \\- {escape_markdown(row[1])}" for row in data if len(row) == 2]
    message = "📄 *Журнал виїздів:*\n" + "\n".join(message_lines)
    await update.message.reply_text(message, parse_mode='MarkdownV2')

def main():
    """Основна функція, що створює та запускає бота."""

    # --- ЗМІНА ---
    # Перевіряємо критичні залежності ПЕРЕД запуском бота
    if not sheet_handler:
        logging.critical("SheetHandler не ініціалізовано. Бот не може продовжити роботу. Перевірте конфігурацію та доступ до Google Sheets.")
        return # Зупиняємо виконання, якщо немає доступу до таблиці
    # --- КІНЕЦЬ ЗМІНИ ---

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={PASSWORD_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, check_password)]},
        fallbacks=[CommandHandler("start", start)],
    )
    application.add_handler(conv_handler)

    # Реєструємо команди
    application.add_handler(CommandHandler("help", help_command))
    application.add_handler(CommandHandler("add", add_plate))
    application.add_handler(CommandHandler("delete", delete_plate))
    application.add_handler(CommandHandler("allowed", get_allowed_list))
    application.add_handler(CommandHandler("unauthorized", get_unauthorized_list))
    application.add_handler(CommandHandler("departed", get_departed_list))

    logging.info("Бот запускається...")
    application.run_polling()

if __name__ == "__main__":
    logging.basicConfig(
        level=logging.INFO,
        format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s'
    )

    # Перевіряємо, чи встановлені обов'язкові змінні
    if not TELEGRAM_BOT_TOKEN or not BOT_PASSWORD or not SPREADSHEET_URL:
        # --- ЗМІНА ---
        # Видалено escape_markdown для кращої читабельності логів
        logging.error("!!! ПОПЕРЕДЖЕННЯ: Одна або декілька змінних (TELEGRAM_BOT_TOKEN, BOT_PASSWORD, SPREADSHEET_URL) не налаштовані. Бот не може запуститися. !!!")
        # --- КІНЕЦЬ ЗМІНИ ---
    else:
        main()
