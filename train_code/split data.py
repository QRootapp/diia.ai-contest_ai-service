import random
from pathlib import Path

# --- Налаштування ---
INPUT_FILE = 'label_fixed.txt'  # Вхідний файл з розміткою
TRAIN_FILE = 'train.txt'  # Файл для навчання (90%)
VAL_FILE = 'val.txt'  # Файл для валідації (10%)

SPLIT_RATIO = 0.9  # Відсоток даних для навчання


# --------------------

def split_label_file():
    label_path = Path(INPUT_FILE)

    # 1. Читаємо всі рядки з вашого файлу
    try:
        with open(label_path, 'r', encoding='utf-8') as f:
            all_lines = f.readlines()
    except FileNotFoundError:
        print(f"Помилка: Файл '{INPUT_FILE}' не знайдено.")
        print("   Будь ласка, переконайтеся, що він лежить у тій самій папці.")
        return

    if not all_lines:
        print(f"Помилка: Файл '{INPUT_FILE}' порожній.")
        return

    # 2. Перемішуємо рядки для забезпечення випадкового розподілу даних
    random.shuffle(all_lines)

    # 3. Визначаємо точку розділу
    split_point = int(len(all_lines) * SPLIT_RATIO)

    train_lines = all_lines[:split_point]
    val_lines = all_lines[split_point:]

    # 4. Записуємо тренувальний файл
    with open(TRAIN_FILE, 'w', encoding='utf-8') as f:
        f.writelines(train_lines)
    print(f"Створено '{TRAIN_FILE}' з {len(train_lines)} рядками.")

    # 5. Записуємо файл для перевірки
    with open(VAL_FILE, 'w', encoding='utf-8') as f:
        f.writelines(val_lines)
    print(f"Створено '{VAL_FILE}' з {len(val_lines)} рядками.")

    print("\nГотово. Дані готові до тренування.")


if __name__ == "__main__":
    split_label_file()