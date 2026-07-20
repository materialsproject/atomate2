#!/usr/bin/env python
"""Query most recent calculations."""

from maggma.stores import MongoStore
from datetime import datetime

# Connect to database
store = MongoStore(
    database="atomate2siesta", collection_name="tasks", host="localhost", port=27017
)

store.connect()

print("Recent Calculations")
print("=" * 70)

# Get all documents
docs = list(store.query())

# Sort by completion time (most recent first)
docs_sorted = sorted(docs, key=lambda d: d.get("completed_at", ""), reverse=True)

# Show recent 10
for i, doc in enumerate(docs_sorted[:10], 1):
    output = doc.get("output", {})

    # Handle both dict and list outputs
    if isinstance(output, dict):
        calc_output = output.get("output", {})
        formula = output.get("formula_pretty", "N/A")
        energy = calc_output.get("energy", "N/A")
    else:
        formula = "N/A"
        energy = "N/A"

    # Format date
    date_str = doc.get("completed_at", "N/A")
    try:
        date_obj = datetime.fromisoformat(date_str.replace("Z", "+00:00"))
        date_formatted = date_obj.strftime("%Y-%m-%d %H:%M")
    except (ValueError, AttributeError):
        date_formatted = date_str[:16] if len(date_str) > 16 else date_str

    calc_type = doc.get("name", "N/A")

    print(
        f"{i:2d}. [{date_formatted}] {formula:8s} " f"E={energy:>12} eV  ({calc_type})"
    )

store.close()
