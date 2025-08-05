import os
import csv

SAVE_ROOT = "submissions"
output_file = "summary.csv"

def parse_metadata(txt_path):
    data = {}
    with open(txt_path, "r") as f:
        for line in f:
            if ":" in line:
                key, val = line.strip().split(":", 1)
                data[key.strip()] = val.strip()
    return data

rows = []
for participant in sorted(os.listdir(SAVE_ROOT)):
    part_dir = os.path.join(SAVE_ROOT, participant)
    if not os.path.isdir(part_dir):
        continue
    for fname in os.listdir(part_dir):
        if fname.endswith(".txt"):
            meta = parse_metadata(os.path.join(part_dir, fname))
            row = {
                "participant_id": meta.get("participant_id", participant),
                "uploaded_at": meta.get("uploaded_at", ""),
                "ink_density": meta.get("ink_density", ""),
                "raw_filename": meta.get("raw_filename", ""),
                "meta_file": fname,
            }
            rows.append(row)

with open(output_file, "w", newline="") as csvfile:
    writer = csv.DictWriter(csvfile, fieldnames=["participant_id", "uploaded_at", "ink_density", "raw_filename", "meta_file"])
    writer.writeheader()
    writer.writerows(rows)

print(f"Wrote summary of {len(rows)} submissions to {output_file}")
