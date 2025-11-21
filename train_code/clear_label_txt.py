import sys
import io

if sys.platform == 'win32':
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')

INPUT_FILE = 'car_plates_photos_ordered_ready_for_learning\label.txt'
OUTPUT_FILE = 'car_plates_photos_ordered_ready_for_learning\label_fixed.txt'

# Словник для заміни кирилиці на латиницю
CYRILLIC_TO_LATIN = {
    'А': 'A', 'В': 'B', 'С': 'C', 'Е': 'E',
    'Н': 'H', 'І': 'I', 'К': 'K', 'М': 'M',
    'О': 'O', 'Р': 'P', 'Т': 'T', 'Х': 'X'
}


def normalize_label_file():
    print(f"Читання '{INPUT_FILE}'...")
    try:
        with open(INPUT_FILE, 'r', encoding='utf-8') as f_in:
            lines = f_in.readlines()
    except FileNotFoundError:
        print(f"Помилка: Файл '{INPUT_FILE}' не знайдено.")
        return

    fixed_lines = 0
    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f_out:
        for line in lines:
            if '\t' not in line:
                continue  # Пропускаємо пошкоджені рядки, якщо є

            path, label = line.strip().split('\t')

            # Нормалізуємо сам лейбл
            new_label = ""
            for char in label:
                # Якщо символ є в словнику заміни, замінюємо
                if char in CYRILLIC_TO_LATIN:
                    new_label += CYRILLIC_TO_LATIN[char]
                    fixed_lines += 1
                else:
                    # Інакше додаємо як є (це буде латиниця або цифра)
                    new_label += char

            # Записуємо виправлений рядок
            f_out.write(f"{path}\t{new_label}\n")

    print(f"Готово! Файл '{OUTPUT_FILE}' створено.")
    print(f"Замінено {fixed_lines} кириличних символів на латинські.")


if __name__ == "__main__":
    normalize_label_file()