import csv
import os
from typing import List, Dict, Any

def load_existing_csv(path: str) -> List[Dict[str, Any]]:
    """Load existing rows from a CSV file if it exists."""
    if not os.path.exists(path):
        return []
    with open(path, "r", newline='') as csvfile:
        reader = csv.DictReader(csvfile)
        return list(reader)

def get_all_fieldnames(rows: List[Dict[str, Any]]) -> List[str]:
    """Return all unique fieldnames across rows, preserving first-seen order."""
    seen = set()
    ordered_fields = []
    for row in rows:
        for key in row:
            if key not in seen:
                seen.add(key)
                ordered_fields.append(key)
    return ordered_fields

def fill_missing_fields(rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """In-place fill missing fields with empty string in each row."""
    for row in rows:
        for field in fieldnames:
            if field not in row:
                row[field] = ""

def write_csv(path: str, rows: List[Dict[str, Any]], fieldnames: List[str]) -> None:
    """Write all rows with provided fieldnames to CSV."""
    with open(path, "w", newline='') as csvfile:
        writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
        writer.writeheader()
        for row in rows:
            writer.writerow(row)
