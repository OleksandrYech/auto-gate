# bot/bot_main.py
import logging
import os
import sys
import json
import asyncio

from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup, BotCommand
from telegram.ext import (
    Application,
    ApplicationBuilder,  # Додаємо ApplicationBuilder
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler,
)

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.sheet_handler import SheetHandler
from bot.telegram_notifier import escape_markdown
from core.settings_manager import SettingsManager

# --- КОНФІГУРАЦІЯ (без змін) ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "8392356130:AAHlCj5LFqKizWp17KKPTXetvjQiPq30i6U")
BOT_PASSWORD = os.getenv("BOT_PASSWORD", "56xW")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL",
                            "https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0")
CONFIG_DIR = os.path.join(os.getcwd(), "config")
SHEETS_CREDENTIALS_PATH = os.path.join(CONFIG_DIR, "credentials.json")
AUTHORIZED_USERS_FILE = os.path.join(CONFIG_DIR, "authorized_users.json")

# --- ГЛОБАЛЬНІ ЗМІННІ ТА ФУНКЦІЇ (без змін) ---
settings_mgr = SettingsManager()
sheet_handler = None
(PASSWORD_ENTRY, MAIN_MENU, AWAITING_PLATE_TO_ADD, AWAITING_PLATE_TO_DELETE) = range(4)


def load_authorized_users() -> set:
    if not os.path.exists(AUTHORIZED_USERS_FILE): return set()
    try:
        with open(AUTHORIZED_USERS_FILE, 'r', encoding='utf-8') as f:
            user_ids = json.load(f)
            logging.info(f"Завантажено {len(user_ids)} авторизованих користувачів.")
            return set(user_ids)
    except (json.JSONDecodeError, IOError):
        return set()


def save_authorized_users(user_set: set):
    os.makedirs(CONFIG_DIR, exist_ok=True)
    with open(AUTHORIZED_USERS_FILE, 'w', encoding='utf-8') as f:
        json.dump(list(user_set), f, indent=4)
    logging.info(f"Список з {len(user_set)} користувачів збережено.")


AUTHORIZED_USERS = load_authorized_users()


# --- УСІ ОБРОБНИКИ ДІАЛОГІВ (start, check_password, і т.д.) залишаються без змін ---
async def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    is_enabled = settings_mgr.are_notifications_enabled()
    notify_button_text = "❌ Вимкнути сповіщення" if is_enabled else "✅ Увімкнути сповіщення"
    keyboard = [[InlineKeyboardButton("➕ Додати номер", callback_data='add_plate')],
                [InlineKeyboardButton("➖ Вилучити номер", callback_data='delete_plate')],
                [InlineKeyboardButton("📂 Переглянути списки", callback_data='view_lists')],
                [InlineKeyboardButton(notify_button_text, callback_data='toggle_notify')]]
    return InlineKeyboardMarkup(keyboard)


async def get_lists_menu_keyboard() -> InlineKeyboardMarkup:
    keyboard = [[InlineKeyboardButton("Авторизовані", callback_data='list_allowed')],
                [InlineKeyboardButton("Неавторизовані", callback_data='list_unauthorized')],
                [InlineKeyboardButton("Ті, що виїхали", callback_data='list_departed')],
                [InlineKeyboardButton("⬅️ Назад", callback_data='back_to_main')]]
    return InlineKeyboardMarkup(keyboard)


async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.effective_chat.id in AUTHORIZED_USERS:
        reply_markup = await get_main_menu_keyboard()
        await update.message.reply_text("👋 Вітаю! Ви вже авторизовані. Оберіть дію:", reply_markup=reply_markup)
        return MAIN_MENU
    await update.message.reply_text("👋 Вітаю\\! Для доступу до системи, будь ласка, введіть пароль\\.",
                                    parse_mode='MarkdownV2')
    return PASSWORD_ENTRY


async def check_password(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    if update.message.text == BOT_PASSWORD:
        chat_id = update.effective_chat.id
        if chat_id not in AUTHORIZED_USERS:
            AUTHORIZED_USERS.add(chat_id)
            save_authorized_users(AUTHORIZED_USERS)
        logging.info(f"Користувач {update.effective_user.name} (ID: {chat_id}) успішно авторизувався.")
        reply_markup = await get_main_menu_keyboard()
        await update.message.reply_text("✅ Доступ дозволено! Оберіть дію:", reply_markup=reply_markup)
        return MAIN_MENU
    else:
        logging.warning(f"Невдала спроба авторизації від {update.effective_user.name}.")
        await update.message.reply_text("⛔️ Неправильний пароль\\. Спробуйте ще раз або введіть /start\\.",
                                        parse_mode='MarkdownV2')
        return PASSWORD_ENTRY


async def main_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == 'add_plate':
        await query.message.reply_text("Введіть номерний знак для додавання:")
        return AWAITING_PLATE_TO_ADD
    elif query.data == 'delete_plate':
        await query.message.reply_text("Введіть номерний знак для вилучення:")
        return AWAITING_PLATE_TO_DELETE
    elif query.data == 'view_lists':
        reply_markup = await get_lists_menu_keyboard()
        await query.edit_message_text("Оберіть список для перегляду:", reply_markup=reply_markup)
        return MAIN_MENU
    elif query.data == 'toggle_notify':
        current_status = settings_mgr.are_notifications_enabled()
        settings_mgr.set_notifications_enabled(not current_status)
        reply_markup = await get_main_menu_keyboard()
        await query.edit_message_text("Оберіть дію:", reply_markup=reply_markup)
        status_text = "УВІМКНЕНО" if not current_status else "ВИМКНЕНО"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"✅ Сповіщення тепер *{status_text}*",
                                       parse_mode='MarkdownV2')
    return MAIN_MENU


async def lists_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    query = update.callback_query
    await query.answer()
    if query.data == 'back_to_main':
        reply_markup = await get_main_menu_keyboard()
        await query.edit_message_text("Оберіть дію:", reply_markup=reply_markup)
        return MAIN_MENU
    list_map = {'list_allowed': (sheet_handler.get_list_by_columns, [1], "📄 *Список дозволених номерів:*"),
                'list_unauthorized': (sheet_handler.get_list_by_columns, [4, 5], "📄 *Неавторизовані спроби:*"),
                'list_departed': (sheet_handler.get_list_by_columns, [7, 8], "📄 *Журнал виїздів:*")}
    if query.data in list_map:
        func, cols, title = list_map[query.data]
        data = func(cols)
        if not data:
            message_text = f"{title}\n\n_(порожньо)_"
        else:
            if len(cols) == 1:
                message_lines = [f"`{row[0]}`" for row in data]
            else:
                message_lines = [f"`{row[0]}` \\- {escape_markdown(row[1])}" for row in data if len(row) == 2]
            message_text = f"{title}\n" + "\n".join(message_lines)
        await query.message.reply_text(message_text, parse_mode='MarkdownV2')
    return MAIN_MENU


async def process_plate_to_add(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    plate_number = update.message.text.upper()
    success, message = sheet_handler.add_authorized_vehicle(plate_number)
    await update.message.reply_text(escape_markdown(message))
    reply_markup = await get_main_menu_keyboard()
    await update.message.reply_text("Оберіть наступну дію:", reply_markup=reply_markup)
    return MAIN_MENU


async def process_plate_to_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    plate_number = update.message.text.upper()
    success, message = sheet_handler.delete_authorized_vehicle(plate_number)
    await update.message.reply_text(escape_markdown(message))
    reply_markup = await get_main_menu_keyboard()
    await update.message.reply_text("Оберіть наступну дію:", reply_markup=reply_markup)
    return MAIN_MENU


async def menu(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    reply_markup = await get_main_menu_keyboard()
    await update.message.reply_text("Оберіть дію:", reply_markup=reply_markup)
    return MAIN_MENU


# --- КЛЮЧОВА ЗМІНА: Асинхронна функція для налаштувань ---
async def post_init(application: Application):
    """Виконує асинхронні налаштування після ініціалізації бота."""
    await application.bot.set_my_commands([
        BotCommand("start", "🚀 Перезапустити / Авторизуватися"),
        BotCommand("menu", "📃 Виклакати Меню"),
    ])


def main():
    """Синхронна основна функція, що створює та запускає бота."""
    global sheet_handler
    try:
        if "YOUR_SPREADSHEET_URL_HERE" in SPREADSHEET_URL or not SPREADSHEET_URL:
            logging.warning("URL для Google Sheets не налаштовано.")
        else:
            sheet_handler = SheetHandler(credentials_file_path=SHEETS_CREDENTIALS_PATH, spreadsheet_url=SPREADSHEET_URL)
    except Exception as e:
        logging.critical(f"Критична помилка ініціалізації SheetHandler: {e}", exc_info=True)
        return

    # --- КЛЮЧОВА ЗМІНА: Правильний спосіб ініціалізації ---
    application = (
        ApplicationBuilder()
        .token(TELEGRAM_BOT_TOKEN)
        .post_init(post_init)  # Додаємо нашу асинхронну функцію налаштувань
        .build()
    )

    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            PASSWORD_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, check_password)],
            MAIN_MENU: [
                CallbackQueryHandler(main_menu_handler, pattern='^(add_plate|delete_plate|toggle_notify|view_lists)$'),
                CallbackQueryHandler(lists_menu_handler,
                                     pattern='^(list_allowed|list_unauthorized|list_departed|back_to_main)$')
            ],
            AWAITING_PLATE_TO_ADD: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_plate_to_add)],
            AWAITING_PLATE_TO_DELETE: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_plate_to_delete)],
        },
        fallbacks=[CommandHandler("menu", menu), CommandHandler("start", start)],
    )

    application.add_handler(conv_handler)

    logging.info("Бот запускається...")
    # Запускаємо бота в блокуючому режимі. Це найпростіший і найнадійніший спосіб.
    application.run_polling()


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')
    if not TELEGRAM_BOT_TOKEN or not BOT_PASSWORD:
        logging.error("!!! ПОПЕРЕДЖЕННЯ: TELEGRAM_BOT_TOKEN або BOT_PASSWORD не налаштовані !!!")
    else:
        # Просто викликаємо звичайну (синхронну) функцію main
        main()