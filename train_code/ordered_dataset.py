import os
import shutil
from pathlib import Path

# --- Налаштування ---
# Вхідні шляхи (вихідні файли та папки)
OLD_LABELS_FILE = "labels_full.txt"
OLD_IMAGES_DIR = "plates_full"

# Вихідні шляхи (результат обробки)
NEW_IMAGES_DIR = 'ordered\car_plates'
NEW_LABELS_FILE = 'ordered\label.txt'


# --------------------

def reorder_dataset():
    old_labels_path = Path(OLD_LABELS_FILE)
    old_images_path = Path(OLD_IMAGES_DIR)
    new_images_path = Path(NEW_IMAGES_DIR)
    new_labels_path = Path(NEW_LABELS_FILE)

    # Перевірка
    if not old_labels_path.exists():
        print(f"Файл {OLD_LABELS_FILE} не знайдено.")
        return

    # Створюємо нову папку
    if new_images_path.exists():
        print(f"Увага: Папка {NEW_IMAGES_DIR} вже існує. Видаліть її або змініть налаштування.")
        # shutil.rmtree(new_images_path) # Можна розкоментувати для авто-видалення
        return

    new_images_path.mkdir(parents=True, exist_ok=True)
    print(f"Створено нову папку: {NEW_IMAGES_DIR}")

    # Читаємо старий файл
    with open(old_labels_path, 'r', encoding='utf-8') as f:
        lines = f.readlines()

    new_lines = []
    counter = 1

    print("Початок перейменування та копіювання...")

    for line in lines:
        parts = line.strip().split('\t')
        if len(parts) < 2:
            continue

        # Отримуємо старе ім'я та текст номера
        old_path_str = parts[0]  # напр. "car_plates/old_name.png"
        plate_text = parts[1]  # напр. "AA1234BB"

        old_filename = os.path.basename(old_path_str)  # "old_name.png"
        source_file = old_images_path / old_filename

        if source_file.exists():
            # 1. Генеруємо нове ім'я (7 цифр, наприклад 0000001.png)
            new_filename = f"{counter:07d}.png"
            destination_file = new_images_path / new_filename

            # 2. Копіюємо файл
            shutil.copy2(source_file, destination_file)

            # 3. Формуємо рядок для нового текстового файлу
            # Важливо: зберігаємо префікс папки, як у твоєму прикладі
            new_line = f"car_plates/{new_filename}\t{plate_text}\n"
            new_lines.append(new_line)

            print(f"   {old_filename} -> {new_filename} ({plate_text})")
            counter += 1
        else:
            print(f"   Пропущено (файл не знайдено): {old_filename}")

    # Записуємо новий файл розмітки
    with open(new_labels_path, 'w', encoding='utf-8') as f:
        f.writelines(new_lines)

    print("\n" + "=" * 50)
    print("ГОТОВО.")
    print(f"Новий файл розмітки: {NEW_LABELS_FILE}")
    print(f"Нова папка з фото: {NEW_IMAGES_DIR}")
    print(f"Всього оброблено: {counter - 1} фото")
    print("=" * 50)
    print("\nНаступні кроки:")
    print("1. Перевір нову папку 'car_plates_ordered'.")
    print("2. Якщо все ок, видали стару папку 'car_plates'.")
    print("3. Перейменуй 'car_plates_ordered' у 'car_plates'.")
    print("4. Заміни 'label.txt' на 'label_ordered.txt'.")


if __name__ == "__main__":
    reorder_dataset()