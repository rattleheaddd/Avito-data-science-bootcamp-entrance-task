import json
from pathlib import Path


ROOT = Path("data/textocr_dataset")
json_path = ROOT / "TextOCR_0.1_train.json"

with open(json_path, "r", encoding="utf-8") as f:
    data = json.load(f)

print("Top-level type:", type(data))
print("Top-level keys:", data.keys())

for key, value in data.items():
    print("\n" + "=" * 60)
    print("KEY:", key)
    print("TYPE:", type(value))

    try:
        print("LEN:", len(value))
    except TypeError:
        pass

    if isinstance(value, dict):
        if len(value) == 0:
            print("EMPTY DICT")
            continue

        first_key = next(iter(value))

        print("FIRST KEY:", first_key)
        print("FIRST VALUE:")
        print(value[first_key])

    elif isinstance(value, list):
        if len(value) == 0:
            print("EMPTY LIST")
            continue

        print("FIRST ITEM:")
        print(value[0])