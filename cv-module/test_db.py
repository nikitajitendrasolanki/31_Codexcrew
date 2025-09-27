from db import insert_violation, get_all_violations

# Test ek violation insert karna
sample_data = {
    "vehicle_no": "MH12AB1234",
    "violation_type": "no_helmet",
    "timestamp": "2025-09-26 18:20:00"
}

result = insert_violation(sample_data)
print("Inserted ID:", result.inserted_id)

# Ab check karo ki data aa gaya ya nahi
all_data = get_all_violations()
print("All violations:")
for v in all_data:
    print(v)
