from itertools import combinations

# List of items with their prices
items = [
    ("Honey Roasted Peanuts (2)", 10.24),
    ("Jalapeno Chips", 1.50),
    ("24 Slice Cheese", 2.48),
    ("Vegetable Oil", 3.97),
    ("Coca Cola", 3.00),
    ("Chocolate Milk (2)", 5.46),
    ("Eggs (60)", 13.22),
    ("Fajitas (2)", 4.24),
    ("Laundry Hamper", 3.92),
    ("Diced Tomatoes", 0.96),
    ("Crunchy Peanut Butter", 3.98),
]

target = 46.80

# To handle floating point precision issues, we multiply prices by 100 and work in cents
items_cents = [(name, int(price*100 + 0.5)) for name, price in items]
target_cents = int(target * 100 + 0.5)

# Check all combinations
found = False
for r in range(1, len(items_cents)+1):
    for combo in combinations(items_cents, r):
        total = sum(price for name, price in combo)
        if total == target_cents:
            print("Combination found:")
            for name, price in combo:
                print(f"{name} - ${price/100:.2f}")
            print(f"Total: ${target:.2f}\n")
            found = True

if not found:
    print("No combination sums to the target exactly.")
