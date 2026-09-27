from pathlib import Path

root = Path("data/synthtiger_debug/images/0")

images = list(root.rglob("*.jpg"))

print(len(images))