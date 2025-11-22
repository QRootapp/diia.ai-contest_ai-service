import cv2
from pathlib import Path

# --- Налаштування ---

# Папка з вихідними зображеннями
SOURCE_DIR = 'car_detection_processing'

# Папка для збереження нормалізованих файлів
OUTPUT_DIR = 'car_cleaned'

# Цільовий формат зображення (.png - оптимально для OCR, .jpg - для економії місця)
TARGET_FORMAT = '.png'

# Кількість цифр у імені файлу (для формату "0000001")
DIGIT_PADDING = 7


# ---------------------------------

def normalize_images():
    """
    Читає всі зображення з SOURCE_DIR, конвертує у TARGET_FORMAT
    і зберігає з новими іменами (0000001.png, 0000002.png...) у OUTPUT_DIR.
    """

    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)

    # 1. Створюємо вихідну папку, якщо її не існує
    output_path.mkdir(parents=True, exist_ok=True)

    # Знаходимо всі файли зображень за підтримуваними розширеннями
    # Glob у Windows автоматично знайде файли незалежно від регістру
    allowed_extensions = ['.jpg', '.jpeg', '.png', '.webp', '.bmp']
    image_files = []
    for ext in allowed_extensions:
        image_files.extend(list(source_path.glob(f'*{ext}')))

    # Видаляємо дублікати та сортуємо список файлів
    image_files = sorted(list(set(image_files)))

    if not image_files:
        print(f"Помилка: Не знайдено жодних зображень у папці '{SOURCE_DIR}'.")
        print("   Будь ласка, перевірте шлях у 'SOURCE_DIR'.")
        return

    print(f"Знайдено {len(image_files)} зображень. Починаємо обробку...")

    counter = 1
    # 3. Обробляємо кожне зображення
    for file_path in image_files:
        try:
            # Читаємо зображення за допомогою OpenCV
            img = cv2.imread(str(file_path))

            if img is None:
                print(f"   Пропущено (не вдалося прочитати): {file_path.name}")
                continue

            # Генеруємо нове ім'я файлу
            # str(counter).zfill(DIGIT_PADDING) перетворить 1 -> "0000001"
            new_name = f"{str(counter).zfill(DIGIT_PADDING)}{TARGET_FORMAT}"
            new_save_path = output_path / new_name

            # Зберігаємо зображення у новому форматі
            cv2.imwrite(str(new_save_path), img)

            print(f"   [ {counter} ] Оброблено: {file_path.name}  ->  {new_name}")

            # Збільшуємо лічильник
            counter += 1

        except Exception as e:
            print(f"   Помилка обробки {file_path.name}: {e}")

    print("\n" + "=" * 40)
    print(f"Готово. Успішно оброблено {counter - 1} файлів.")
    print(f"   Усі вони збережені у папці: '{OUTPUT_DIR}'")
    print("=" * 40)


if __name__ == "__main__":
    # pip install opencv-python-headless
    normalize_images()