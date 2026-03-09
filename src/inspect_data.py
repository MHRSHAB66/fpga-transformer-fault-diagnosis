from pathlib import Path
from config import DATA_RAW_DIR, DATA_SAMPLE_DIR

def list_files(directory: Path):
    if not directory.exists():
        print(f"[WARN] Directory does not exist: {directory}")
        return []

    files = [p for p in directory.rglob("*") if p.is_file()]
    print(f"[INFO] Found {len(files)} files in {directory}")
    for f in files[:20]:
        print(f" - {f}")
    return files

if __name__ == "__main__":
    print("Inspecting sample directory...")
    list_files(DATA_SAMPLE_DIR)

    print("\nInspecting raw directory...")
    list_files(DATA_RAW_DIR)