import cv2
from pathlib import Path
from ultralytics import YOLO

# --- Налаштування ---
# Шлях до навченої моделі YOLO для детекції номерних знаків
YOLO_MODEL_PATH = r'..\runs\train\license_plates8\weights\best.pt'

# Папка з чистими фото (вихід попереднього скрипту)
SOURCE_DIR = r'..\car_cleaned'

# Папка для збереження вирізаних номерних знаків
OUTPUT_DIR = r'..\car_plates'

# 3. Мінімальна якість детекції (0.4 = 40%)
CONFIDENCE_THRESHOLD = 0.4

# 4. Мінімальний розмір номера (щоб відсіяти "сміття")
MIN_WIDTH = 40
MIN_HEIGHT = 15


# --------------------

def crop_plates_from_images():
    """
    Проходить по папці SOURCE_DIR, знаходить номери через YOLO,
    і зберігає НАЙКРАЩУ вирізку в OUTPUT_DIR під тією ж назвою.
    """

    print(f"Завантаження моделі YOLO з {YOLO_MODEL_PATH}...")
    try:
        model = YOLO(YOLO_MODEL_PATH)
    except Exception as e:
        print(f"Помилка завантаження моделі YOLO: {e}")
        return

    source_path = Path(SOURCE_DIR)
    output_path = Path(OUTPUT_DIR)

    # 1. Створюємо вихідну папку
    output_path.mkdir(parents=True, exist_ok=True)

    # 2. Знаходимо всі .png файли (з попереднього скрипту)
    image_files = sorted(list(source_path.glob('*.png')))

    if not image_files:
        print(f"Помилка: Не знайдено файлів .png у папці '{SOURCE_DIR}'.")
        print(f"   Перевірте, чи шлях '{SOURCE_DIR}' вказано правильно.")
        print(f"   Також переконайтеся, що скрипт 'normalize_images.py' відпрацював і створив там .png файли.")
        return

    print(f"Знайдено {len(image_files)} зображень. Починаємо обробку...")

    total_cropped = 0
    total_skipped = 0

    # 3. Обробляємо кожне зображення
    for i, file_path in enumerate(image_files):
        try:
            img = cv2.imread(str(file_path))
            if img is None:
                print(f"   Пропущено (не вдалося прочитати): {file_path.name}")
                total_skipped += 1
                continue

            # Запускаємо YOLO
            results = model(img, verbose=False, conf=CONFIDENCE_THRESHOLD)

            best_crop = None
            best_conf = 0.0

            # 4. Шукаємо найкращий номер на фото
            if results and results[0].boxes:
                for box in results[0].boxes:
                    conf = float(box.conf[0])
                    if conf > best_conf:
                        best_conf = conf
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        # Вирізаємо
                        crop = img[y1:y2, x1:x2]

                        # Перевірка якості
                        h, w = crop.shape[:2]
                        if w >= MIN_WIDTH and h >= MIN_HEIGHT:
                            best_crop = crop
                        else:
                            print(
                                f"   [Файл {file_path.name}] Знайдено номер, але він занадто малий ({w}x{h}). Шукаємо далі...")
                            best_conf = 0  # Скидаємо, щоб не зберегти цей малий номер

            # 5. Зберігаємо найкращу знайдену вирізку
            if best_crop is not None:
                # Зберігаємо під тією ж назвою, що й оригінал
                new_save_path = output_path / file_path.name
                cv2.imwrite(str(new_save_path), best_crop)
                print(
                    f"   [ {i + 1}/{len(image_files)} ] Збережено вирізку: {new_save_path.name} (Conf: {best_conf * 100:.1f}%)")
                total_cropped += 1
            else:
                print(
                    f"   [ {i + 1}/{len(image_files)} ] Номери не знайдено (або вони занадто малі) у файлі: {file_path.name}")
                total_skipped += 1

        except Exception as e:
            print(f"   Помилка обробки {file_path.name}: {e}")
            total_skipped += 1

    print("\n" + "=" * 40)
    print(f"Готово.")
    print(f"   Успішно вирізано: {total_cropped} номерів")
    print(f"   Пропущено (не знайдено): {total_skipped} фото")
    print(f"   Усі вони збережені у папці: '{OUTPUT_DIR}'")
    print("=" * 40)


if __name__ == "__main__":
    crop_plates_from_images()