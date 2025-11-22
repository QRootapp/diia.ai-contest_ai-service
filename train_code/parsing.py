import os
import time
import random
import requests
from pathlib import Path
from selenium import webdriver
from selenium.webdriver.common.by import By
from selenium.webdriver.chrome.service import Service as ChromeService
from webdriver_manager.chrome import ChromeDriverManager
from selenium.common.exceptions import NoSuchElementException

# --- Налаштування ---
REGIONS_TO_PARSE = [
    "ternopil", "kharkiv", "kherson", "khmelnitskiy", "cherkassy", "chernigov", "chernivtsi"
]
#"zhitomir", "odessa", "vinnytsya", "volyn", "dnipro",    "donetsk", "zakarpattja", "zaporizhzhya", "ivano-frankivsk","kirovograd", "lugansk", "lutsk", "mykolayiv", "poltava", "rivno", "sumy",
DOWNLOAD_DIR = "car_dataset_big"  # Папка для збереження завантажених зображень
CARS_LIMIT_PER_REGION = 200  # Кількість машин для обробки з одного регіону
MAX_PHOTOS_PER_CAR = 8  # Максимальна кількість фото з однієї машини
# --------------------

Path(DOWNLOAD_DIR).mkdir(parents=True, exist_ok=True)
REQUEST_HEADERS = {
    'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/115.0.0.0 Safari/537.36'
}


def download_image(img_url, file_name):
    try:
        if not img_url or not img_url.startswith('http'): return False
        img_data = requests.get(img_url, headers=REQUEST_HEADERS, timeout=15).content
        file_path = Path(file_name)
        # Зберігаємо .webp як .jpg для сумісності
        if file_path.suffix not in ['.jpg', '.webp', '.png']:
            file_path = file_path.with_suffix('.jpg')

        with open(file_path, 'wb') as f:
            f.write(img_data)
        return True
    except Exception:
        return False


def process_single_car_page(driver, car_url, downloaded_ids):
    """Заходить в оголошення і качає HD фото з головного слайдера."""
    try:
        print(f"   Обробка: {car_url}")
        driver.get(car_url)
        time.sleep(2)

        # Шукаємо слайдер
        images = driver.find_elements(By.CSS_SELECTOR, "#photoSlider img, div.gallery-order img")
        if not images:
            images = driver.find_elements(By.TAG_NAME, "img")

        saved_count = 0
        for img in images:
            if saved_count >= MAX_PHOTOS_PER_CAR: break

            try:
                src = img.get_attribute("src")
                if not src: src = img.get_attribute("data-src")

                if src and "riastatic.com/photosnew" in src:
                    final_url = src

                    # Force HD logic
                    formats_to_replace = ["fx.jpg", "bx.jpg", "s.jpg", "m.jpg", "fx.webp", "bx.webp", "s.webp"]
                    for fmt in formats_to_replace:
                        if fmt in final_url:
                            final_url = final_url.replace(fmt, "hd.jpg")
                            break

                    if final_url == src and ".jpg" in final_url:
                        final_url = final_url.replace(".jpg", "hd.jpg")

                    # Конвертація розширення для URL
                    final_url = final_url.replace(".webp", ".jpg")

                    file_id = final_url.split('/')[-1].split('?')[0]

                    if file_id not in downloaded_ids:
                        file_name = Path(DOWNLOAD_DIR) / file_id
                        if download_image(final_url, file_name):
                            downloaded_ids.add(file_id)
                            saved_count += 1
                            # print(f"      +1 фото") # Можна розкоментувати, щоб бачити кожне фото
            except Exception:
                continue

        if saved_count > 0:
            print(f"      Збережено {saved_count} фото.")
        return saved_count

    except Exception as e:
        print(f"   Помилка: {e}")
        return 0


def scrape_region_mass_download(region_slug, downloaded_ids):
    print("\n" + "=" * 50)
    print(f"МАСШТАБНИЙ ПАРСИНГ РЕГІОНУ: {region_slug.upper()}")
    print("=" * 50)

    options = webdriver.ChromeOptions()
    options.add_argument("--disable-blink-features=AutomationControlled")
    options.add_argument("--start-maximized")
    options.add_argument('--ignore-certificate-errors')

    driver = webdriver.Chrome(service=ChromeService(ChromeDriverManager().install()), options=options)

    total_region_photos = 0
    cars_processed_count = 0

    # 1. Відкриваємо регіон
    start_url = f"https://auto.ria.com/uk/city/{region_slug}/"
    driver.get(start_url)
    print("Завантаження...")
    time.sleep(5)

    # 2. Натискаємо "Вживані"
    try:
        print("Фільтр 'Вживані'...")
        xpath_btn = driver.find_elements(By.XPATH, "//label[contains(text(), 'Вживані')]")
        if xpath_btn:
            xpath_btn[0].click()
        else:
            driver.execute_script("document.querySelector(\"label[for='indexName_bu']\").click();")
        time.sleep(5)
    except Exception:
        print("Кнопка не натиснулась, йдемо далі.")

    # 3. ГОЛОВНИЙ ЦИКЛ (ПО СТОРІНКАХ)
    page_number = 1
    while cars_processed_count < CARS_LIMIT_PER_REGION:
        print(
            f"\n--- Сторінка пошуку {page_number} | Оброблено авто: {cars_processed_count}/{CARS_LIMIT_PER_REGION} ---")

        # Скролимо, щоб підвантажити всі машини на поточній сторінці
        for _ in range(5):
            driver.execute_script("window.scrollTo(0, document.body.scrollHeight);")
            time.sleep(1.5)

        # Збираємо посилання
        elements = driver.find_elements(By.XPATH, "//a[contains(@href, '/auto_')]")
        current_page_urls = set()

        for el in elements:
            try:
                url = el.get_attribute("href")
                if url and "newauto" not in url and "auto.ria.com/uk/auto_" in url:
                    current_page_urls.add(url)
            except:
                pass

        print(f"   Знайдено {len(current_page_urls)} машин на цій сторінці.")

        # Обробляємо машини
        for url in current_page_urls:
            if cars_processed_count >= CARS_LIMIT_PER_REGION:
                break

            photos_count = process_single_car_page(driver, url, downloaded_ids)
            if photos_count > 0:
                total_region_photos += photos_count
                cars_processed_count += 1

            # Повертаємось назад на список (Back) - це швидше, ніж відкривати нові вкладки, але стабільніше
            # Використовуємо метод Back для повернення на сторінку пошуку після обробки кожного оголошення
            # Це дозволяє зберегти стан пагінації та уникнути втрати контексту сторінки пошуку

            driver.back()
            # Повернення назад може бути повільним, але це надійний спосіб не втратити пагінацію
            # Після обробки всіх URL з поточної сторінки переходимо на наступну сторінку пошуку

        # === ПЕРЕХІД НА НАСТУПНУ СТОРІНКУ ПОШУКУ ===
        # Оскільки ми ходили по посиланнях і тиснули Back, ми маємо бути на сторінці пошуку.
        # Якщо Back не спрацював ідеально, ми перезавантажимо сторінку з параметром page

        # Формуємо URL наступної сторінки вручну (це найнадійніше)
        page_number += 1
        # Оскільки URL після кліку на "Вживані" динамічний, ми просто шукаємо кнопку "Наступна" або "2", "3"

        try:
            print("   Перехід на наступну сторінку...")
            next_button = driver.find_elements(By.CLASS_NAME, 'js-next')
            if next_button:
                next_button[0].click()
                time.sleep(5)
            else:
                # Якщо кнопки немає, пробуємо магію URL (додаємо ?page=N)
                # Але треба зберегти фільтри.
                # Спробуємо просто знайти пагінацію
                print("   Кнопка 'Далі' не знайдена, завершуємо регіон.")
                break
        except Exception as e:
            print(f"   Неможливо перейти на наступну сторінку: {e}")
            break

    driver.quit()
    return total_region_photos


def run_full_parser():
    existing_files = Path(DOWNLOAD_DIR).glob('*.*')
    downloaded_ids = set(f.name for f in existing_files)
    print(f"ℹ️ Вже є файлів: {len(downloaded_ids)}")

    total_new_images = 0

    for region in REGIONS_TO_PARSE:
        count = scrape_region_mass_download(region, downloaded_ids)
        total_new_images += count

        if REGIONS_TO_PARSE.index(region) < len(REGIONS_TO_PARSE) - 1:
            print(f"Пауза 5 сек...")
            time.sleep(5)

    print("\n" + "=" * 50)
    print(f"ЗАВЕРШЕНО. Завантажено: {total_new_images}")
    print("=" * 50)


if __name__ == "__main__":
    run_full_parser()