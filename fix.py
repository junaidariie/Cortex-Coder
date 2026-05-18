import json

path = "Notebook\Qwen2_5_small_finetuning.ipynb"

with open(path, "r", encoding="utf-8") as f:
    nb = json.load(f)

widgets = nb.get("metadata", {}).get("widgets", {})

if "state" not in widgets:
    widgets["state"] = {}

nb["metadata"]["widgets"] = widgets

with open(path, "w", encoding="utf-8") as f:
    json.dump(nb, f, indent=1)

print("Widget state repaired.")