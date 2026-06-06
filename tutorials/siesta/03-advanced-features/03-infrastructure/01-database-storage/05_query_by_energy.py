#!/usr/bin/env python
"""Query calculations by energy range."""

from maggma.stores import MongoStore

# Connect to database
store = MongoStore(
    database="atomate2siesta", collection_name="tasks", host="localhost", port=27017
)

store.connect()

# Query by energy range (example: find low-energy structures)
min_energy = -230.0  # eV
max_energy = -220.0  # eV

print(f"Searching for energies between {min_energy} and {max_energy} eV")
print("-" * 60)

# Query with energy range
docs = list(
    store.query(
        criteria={"output.output.energy": {"$gte": min_energy, "$lte": max_energy}}
    )
)

print(f"Found {len(docs)} calculations\n")


# Sort by energy and show top 10
def get_energy(doc):
    """Safely get energy from document."""
    output = doc.get("output", {})
    if isinstance(output, dict):
        calc_output = output.get("output", {})
        return calc_output.get("energy", float("inf"))
    return float("inf")


docs_sorted = sorted(docs, key=get_energy)

for i, doc in enumerate(docs_sorted[:10], 1):
    output = doc.get("output", {})

    # Handle both dict and list outputs
    if isinstance(output, dict):
        calc_output = output.get("output", {})
        formula = output.get("formula_pretty", "N/A")
        energy = calc_output.get("energy", 0)
    else:
        formula = "N/A"
        energy = 0

    print(
        f"{i}. {formula:10s} " f"E = {energy:10.4f} eV  " f"({doc.get('name', 'N/A')})"
    )

store.close()
