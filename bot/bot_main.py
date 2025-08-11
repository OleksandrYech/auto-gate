# bot/bot_main.py
import logging
import os
import sys
from telegram import Update, InlineKeyboardButton, InlineKeyboardMarkup
from telegram.ext import (
    Application,
    CommandHandler,
    MessageHandler,
    filters,
    ContextTypes,
    ConversationHandler,
    CallbackQueryHandler, # Додано для обробки кнопок
)

# Додаємо шлях до кореневого каталогу
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from core.sheet_handler import SheetHandler
from bot.telegram_notifier import escape_markdown
from core.settings_manager import SettingsManager

# --- КОНФІГУРАЦІЯ ---
TELEGRAM_BOT_TOKEN = os.getenv("TELEGRAM_BOT_TOKEN", "")
BOT_PASSWORD = os.getenv("BOT_PASSWORD", "")
SPREADSHEET_URL = os.getenv("SPREADSHEET_URL", "https://docs.google.com/spreadsheets/d/1gz5snNdG06sPL0_w2zyWtca3BiAQ7ru8I93LqPVjrC4/edit?gid=0#gid=0")
SHEETS_CREDENTIALS_PATH = os.path.join(os.getcwd(), "config", "credentials.json")


# --- ГЛОБАЛЬНІ ЗМІННІ ТА СТАНИ ДЛЯ ДІАЛОГІВ ---
AUTHORIZED_USERS = set()
settings_mgr = SettingsManager()
sheet_handler = None

# Стани для ConversationHandler
(
    PASSWORD_ENTRY,
    MAIN_MENU,
    AWAITING_PLATE_TO_ADD,
    AWAITING_PLATE_TO_DELETE
) = range(4)


try:
    if "YOUR_SPREADSHEET_URL_HERE" in SPREADSHEET_URL or not SPREADSHEET_URL:
        logging.warning("URL для Google Sheets не налаштовано.")
    else:
        sheet_handler = SheetHandler(
            credentials_file_path=SHEETS_CREDENTIALS_PATH,
            spreadsheet_url=SPREADSHEET_URL
        )
except Exception as e:
    logging.critical(f"Не вдалося ініціалізувати SheetHandler: {e}", exc_info=True)


# --- ЛОГІКА БОТА ---

def auth_required(func):
    """Декоратор для перевірки авторизації."""
    async def wrapper(update: Update, context: ContextTypes.DEFAULT_TYPE, *args, **kwargs):
        if update.effective_chat.id not in AUTHORIZED_USERS:
            await update.message.reply_text("⛔️ Доступ заборонено\\. Будь ласка, авторизуйтесь за допомогою /start\\.", parse_mode='MarkdownV2')
            return
        return await func(update, context, *args, **kwargs)
    return wrapper

# --- ФУНКЦІЇ ДЛЯ ГЕНЕРАЦІЇ КЛАВІАТУР ---

async def get_main_menu_keyboard() -> InlineKeyboardMarkup:
    """Створює та повертає головне меню з кнопками."""
    is_enabled = settings_mgr.are_notifications_enabled()
    notify_button_text = "❌ Вимкнути сповіщення" if is_enabled else "✅ Увімкнути сповіщення"

    keyboard = [
        [InlineKeyboardButton("➕ Додати номер", callback_data='add_plate')],
        [InlineKeyboardButton("➖ Вилучити номер", callback_data='delete_plate')],
        [InlineKeyboardButton("📂 Переглянути списки", callback_data='view_lists')],
        [InlineKeyboardButton(notify_button_text, callback_data='toggle_notify')],
    ]
    return InlineKeyboardMarkup(keyboard)

async def get_lists_menu_keyboard() -> InlineKeyboardMarkup:
    """Створює меню для вибору списків."""
    keyboard = [
        [InlineKeyboardButton("Авторизовані", callback_data='list_allowed')],
        [InlineKeyboardButton("Неавторизовані", callback_data='list_unauthorized')],
        [InlineKeyboardButton("Ті, що виїхали", callback_data='list_departed')],
        [InlineKeyboardButton("⬅️ Назад", callback_data='back_to_main')],
    ]
    return InlineKeyboardMarkup(keyboard)

# --- ОБРОБНИКИ ДІАЛОГІВ ТА КОМАНД ---

async def start(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Розпочинає діалог: перевіряє авторизацію або запитує пароль."""
    if update.effective_chat.id in AUTHORIZED_USERS:
        reply_markup = await get_main_menu_keyboard()
        await update.message.reply_text("👋 Вітаю! Оберіть дію:", reply_markup=reply_markup)
        return MAIN_MENU

    await update.message.reply_text("👋 Вітаю\\! Для доступу до системи, будь ласка, введіть пароль\\.", parse_mode='MarkdownV2')
    return PASSWORD_ENTRY

async def check_password(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Перевіряє пароль та надає доступ до головного меню."""
    if update.message.text == BOT_PASSWORD:
        AUTHORIZED_USERS.add(update.effective_chat.id)
        logging.info(f"Користувач {update.effective_user.name} (ID: {update.effective_chat.id}) успішно авторизувався.")
        reply_markup = await get_main_menu_keyboard()
        await update.message.reply_text("✅ Доступ дозволено! Оберіть дію:", reply_markup=reply_markup)
        return MAIN_MENU
    else:
        logging.warning(f"Невдала спроба авторизації від {update.effective_user.name}.")
        await update.message.reply_text("⛔️ Неправильний пароль\\. Спробуйте ще раз або введіть /start\\.", parse_mode='MarkdownV2')
        return PASSWORD_ENTRY

async def main_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє натискання кнопок головного меню."""
    query = update.callback_query
    await query.answer()

    if query.data == 'add_plate':
        await query.message.reply_text("Введіть номерний знак, який потрібно додати:")
        return AWAITING_PLATE_TO_ADD

    elif query.data == 'delete_plate':
        await query.message.reply_text("Введіть номерний знак, який потрібно вилучити:")
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
        # Опціонально: надіслати підтвердження
        status_text = "УВІМКНЕНО" if not current_status else "ВИМКНЕНО"
        await context.bot.send_message(chat_id=update.effective_chat.id, text=f"✅ Сповіщення тепер *{status_text}*", parse_mode='MarkdownV2')
        return MAIN_MENU

    return MAIN_MENU

async def lists_menu_handler(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє натискання кнопок меню списків."""
    query = update.callback_query
    await query.answer()

    if query.data == 'back_to_main':
        reply_markup = await get_main_menu_keyboard()
        await query.edit_message_text("Оберіть дію:", reply_markup=reply_markup)
        return MAIN_MENU

    # Визначаємо, яку функцію викликати
    list_map = {
        'list_allowed': (sheet_handler.get_list_by_columns, [1], "📄 *Список дозволених номерів:*"),
        'list_unauthorized': (sheet_handler.get_list_by_columns, [4, 5], "📄 *Неавторизовані спроби:*"),
        'list_departed': (sheet_handler.get_list_by_columns, [7, 8], "📄 *Журнал виїздів:*")
    }

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
    """Обробляє введений номер для додавання."""
    plate_number = update.message.text.upper()
    success, message = sheet_handler.add_authorized_vehicle(plate_number)

    await update.message.reply_text(escape_markdown(message))

    # Повертаємо користувача в головне меню
    reply_markup = await get_main_menu_keyboard()
    await update.message.reply_text("Оберіть наступну дію:", reply_markup=reply_markup)
    return MAIN_MENU

async def process_plate_to_delete(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Обробляє введений номер для видалення."""
    plate_number = update.message.text.upper()
    success, message = sheet_handler.delete_authorized_vehicle(plate_number)

    await update.message.reply_text(escape_markdown(message))

    # Повертаємо користувача в головне меню
    reply_markup = await get_main_menu_keyboard()
    await update.message.reply_text("Оберіть наступну дію:", reply_markup=reply_markup)
    return MAIN_MENU

async def cancel(update: Update, context: ContextTypes.DEFAULT_TYPE) -> int:
    """Скасовує поточну операцію та повертає в головне меню."""
    await update.message.reply_text("Дію скасовано.")
    reply_markup = await get_main_menu_keyboard()
    await update.message.reply_text("Оберіть дію:", reply_markup=reply_markup)
    return MAIN_MENU

def main():
    """Основна функція, що створює та запускає бота."""
    if not sheet_handler or not settings_mgr:
        logging.critical("SheetHandler або SettingsManager не ініціалізовано. Бот не може запуститися.")
        return

    application = Application.builder().token(TELEGRAM_BOT_TOKEN).build()

    # Створюємо головний обробник діалогів
    conv_handler = ConversationHandler(
        entry_points=[CommandHandler("start", start)],
        states={
            PASSWORD_ENTRY: [MessageHandler(filters.TEXT & ~filters.COMMAND, check_password)],
            MAIN_MENU: [
                CallbackQueryHandler(main_menu_handler, pattern='^(add_plate|delete_plate|toggle_notify|view_lists)$'),
                CallbackQueryHandler(lists_menu_handler, pattern='^(list_allowed|list_unauthorized|list_departed|back_to_main)$')
            ],
            AWAITING_PLATE_TO_ADD: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_plate_to_add)],
            AWAITING_PLATE_TO_DELETE: [MessageHandler(filters.TEXT & ~filters.COMMAND, process_plate_to_delete)],
        },
        fallbacks=[CommandHandler("cancel", cancel), CommandHandler("start", start)],
        per_message=False
    )

    application.add_handler(conv_handler)

    logging.info("Бот запускається...")
    application.run_polling()

if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - [%(levelname)s] - %(message)s')

    if not TELEGRAM_BOT_TOKEN or not BOT_PASSWORD:
        logging.error("!!! ПОПЕРЕДЖЕННЯ: TELEGRAM_BOT_TOKEN або BOT_PASSWORD не налаштовані. Бот не може запуститися. !!!")
    else:
        main()
