import os
from pathlib import Path

# --- Налаштування ---
# Шлях до текстового файлу з розміткою
LABELS_FILE = "labels_full.txt"

# Шлях до папки з зображеннями
IMAGES_DIR = "plates_full"
# --------------------

def sync_labels_and_images():
    labels_path = Path(LABELS_FILE)
    images_path = Path(IMAGES_DIR)

    # Перевірка існування файлу та папки
    if not labels_path.exists():
        print(f"Файл розмітки {LABELS_FILE} не знайдено.")
        return
    if not images_path.exists():
        print(f"Папку з фото {IMAGES_DIR} не знайдено.")
        return
    if not labels_path.is_file():
        print(f"Помилка: {LABELS_FILE} — це не файл (можливо, це папка?).")
        return

    # 1. Читаємо всі записи з файлу label.txt
    valid_lines = []
    files_in_labels = set()

    print(f"Читання файлу розмітки: {labels_path.name}...")
    with open(labels_path, 'r', encoding='utf-8') as f:
        all_lines = f.readlines()

    # Парсимо файл розмітки
    for line in all_lines:
        parts = line.strip().split('\t')
        if len(parts) >= 1:
            full_path_str = parts[0]
            filename = os.path.basename(full_path_str)

            # Перевіряємо, чи існує такий файл фізично
            if (images_path / filename).exists():
                valid_lines.append(line)
                files_in_labels.add(filename)
            else:
                print(f"   Видаляю з лейблів (немає фото): {filename}")

    # 2. Перезаписуємо label.txt
    if len(valid_lines) < len(all_lines):
        with open(labels_path, 'w', encoding='utf-8') as f:
            f.writelines(valid_lines)
        print(f"Оновлено label.txt: видалено {len(all_lines) - len(valid_lines)} зайвих рядків.")
    else:
        print("label.txt в порядку (всі записи мають фото).")

    # 3. Перевіряємо папку з фото
    print(f"\nПеревірка папки {images_path.name} на зайві фото...")

    all_images = list(images_path.glob('*.*'))
    deleted_images_count = 0

    for img_path in all_images:
        if img_path.name not in files_in_labels:
            try:
                os.remove(img_path)
                print(f"   Видалено фото (немає в лейблах): {img_path.name}")
                deleted_images_count += 1
            except Exception as e:
                print(f"   Помилка видалення {img_path.name}: {e}")

    if deleted_images_count == 0:
        print("Папка в порядку (всі фото є в лейблах).")
    else:
        print(f"Очищено папку: видалено {deleted_images_count} зайвих фото.")

    print("\n" + "=" * 40)
    print(f"Синхронізацію завершено.")
    print(f"Всього пар (фото + текст): {len(valid_lines)}")
    print("=" * 40)


if __name__ == "__main__":
    sync_labels_and_images()