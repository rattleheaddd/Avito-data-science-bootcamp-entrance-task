from pathlib import Path
import random
import shutil
import zipfile


TEST_DIR = Path("data/test/images")  # поменяй на свою папку с test
OUTPUT_DIR = Path("test_random_100")
ZIP_PATH = Path("test_random_100.zip")

N = 100
SEED = 42


def main():
    # Собираем изображения
    images = [
        p for p in TEST_DIR.rglob("*")
        if p.suffix.lower() in {
            ".jpg", ".jpeg", ".png", ".webp", ".bmp"
        }
    ]

    print("Total test images:", len(images))

    if len(images) < N:
        raise RuntimeError(
            f"В test только {len(images)} картинок, "
            f"а запрошено {N}"
        )

    # Фиксированный seed:
    # выбор случайный, но его можно воспроизвести
    rng = random.Random(SEED)

    selected = rng.sample(images, N)

    # Пересоздаём папку
    if OUTPUT_DIR.exists():
        shutil.rmtree(OUTPUT_DIR)

    OUTPUT_DIR.mkdir(parents=True)

    # Копируем
    for i, src in enumerate(selected):
        # Добавляем индекс, чтобы одинаковые filename
        # из разных директорий не перезаписались
        dst = OUTPUT_DIR / f"{i:03d}_{src.name}"

        shutil.copy2(src, dst)

    print(f"Copied {len(selected)} images")

    # Создаём zip
    with zipfile.ZipFile(
        ZIP_PATH,
        "w",
        compression=zipfile.ZIP_DEFLATED,
    ) as zf:

        for path in OUTPUT_DIR.iterdir():
            zf.write(
                path,
                arcname=path.name,
            )

    print("Created:", ZIP_PATH)


if __name__ == "__main__":
    main()