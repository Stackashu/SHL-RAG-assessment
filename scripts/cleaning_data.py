import json

INPUT_FILE = "data/raw_catalog.json"
OUTPUT_FILE = "data/cleaned_catalog.json"

def normalize_test_type(val):
    if not val:
        return ""
    val = val.lower()
    if "knowledge" in val or "skill" in val:
        return "K"
    if "personality" in val or "behavior" in val:
        return "P"
    return ""

def main():
    with open(INPUT_FILE, "r", encoding="utf-8") as f:
        data = json.load(f)

    cleaned = []

    for item in data:
        if not item.get("name"):
            continue

        test_type = normalize_test_type(item.get("test_type", ""))

        text_for_embedding = f"""
        Assessment Name: {item.get('name', '')}
        Description: {item.get('description', '')}
        Test Type: {test_type}
        Duration: {item.get('duration', '')}
        Adaptive: {item.get('adaptive_support', '')}
        Remote: {item.get('remote_support', '')}
        """

        cleaned.append({
            "name": item.get("name", "").strip(),
            "url": item.get("url", "").strip(),
            "test_type": test_type,
            "duration": item.get("duration", ""),
            "adaptive_support": item.get("adaptive_support", ""),
            "remote_support": item.get("remote_support", ""),
            "text": text_for_embedding.strip()
        })

    print("CLEANED ITEMS:", len(cleaned))

    with open(OUTPUT_FILE, "w", encoding="utf-8") as f:
        json.dump(cleaned, f, indent=2, ensure_ascii=False)

if __name__ == "__main__":
    main()
