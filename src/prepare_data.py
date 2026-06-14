import os
import re
import json

from config import CHUNK_SIZE, CHUNK_OVERLAP

RAW_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "raw")
PROCESSED_DIR = os.path.join(os.path.dirname(__file__), "..", "data", "processed")
OUTPUT_FILE = os.path.join(PROCESSED_DIR, "chunks.jsonl")

# Wikipedia disambiguation pages (no real content), found by inspecting data/raw
SKIP_FILES = {"BM25.txt", "Tokenization.txt"}

# Wikipedia sections that contain only link lists, not prose worth retrieving
DROP_SECTIONS = {"see also", "references", "further reading", "external links", "notes"}


def clean_text(text):
    """Remove Wikipedia link-list sections and collapse extra blank lines."""
    cleaned_lines = []
    skipping = False
    for line in text.splitlines():
        header = re.match(r"^==+\s*(.+?)\s*==+$", line.strip())
        if header:
            skipping = header.group(1).strip().lower() in DROP_SECTIONS
            continue
        if not skipping:
            cleaned_lines.append(line)
    text = "\n".join(cleaned_lines)
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def chunk_text(text, chunk_size, overlap):
    """Split text into overlapping fixed-size character chunks."""
    chunks = []
    start = 0
    while start < len(text):
        end = start + chunk_size
        chunk = text[start:end].strip()
        if chunk:
            chunks.append(chunk)
        if end >= len(text):
            break
        start = end - overlap
    return chunks


def main():
    os.makedirs(PROCESSED_DIR, exist_ok=True)
    filenames = sorted(
        f for f in os.listdir(RAW_DIR) if f.endswith(".txt") and f not in SKIP_FILES
    )

    total_chunks = 0
    with open(OUTPUT_FILE, "w", encoding="utf-8") as out:
        for filename in filenames:
            source = filename[:-4].replace("_", " ")
            with open(os.path.join(RAW_DIR, filename), "r", encoding="utf-8") as f:
                raw_text = f.read()

            cleaned = clean_text(raw_text)
            for i, chunk in enumerate(chunk_text(cleaned, CHUNK_SIZE, CHUNK_OVERLAP)):
                record = {"id": f"{filename[:-4]}_{i}", "source": source, "text": chunk}
                out.write(json.dumps(record, ensure_ascii=False) + "\n")
                total_chunks += 1

    print(f"Processed {len(filenames)} documents into {total_chunks} chunks.")
    print(f"Output written to {OUTPUT_FILE}")


if __name__ == "__main__":
    main()
