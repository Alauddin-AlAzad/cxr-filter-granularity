from pathlib import Path

lines = [l.strip() for l in open("images_to_download.txt") if l.strip()]

missing = []
for line in lines:
    local = Path("images") / line
    if not local.exists() or local.stat().st_size == 0:
        missing.append(line)

Path("missing.txt").write_text("\n".join(missing) + "\n")

print("First line in list:", lines[0])
print("Expected local path:", Path("images") / lines[0])
print("Total in list:", len(lines))
print("Already downloaded:", len(lines) - len(missing))
print("Still to download:", len(missing))