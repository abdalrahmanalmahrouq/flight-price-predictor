import requests, random

airlines_economy  = ["IndiGo", "SpiceJet", "GO FIRST", "AirAsia"]
airlines_business = ["Air India", "Vistara"]
cities = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad"]

for _ in range(200):
    # Mirror training distribution — 70% economy, 30% business
    is_business = 1 if random.random() < 0.30 else 0
    city_pair = random.sample(cities, 2)
    airline = (random.choice(airlines_business) if is_business
               else random.choice(airlines_economy))
    is_weekend = random.choice([0, 1])

    requests.post("http://localhost:8000/predict", json={
        "stops_numeric":    random.choice([0, 1, 2]),
        "duration_minutes": random.randint(60, 600),
        "departure_hour":   random.randint(0, 23),
        "arrival_hour":     random.randint(0, 23),
        "month":            random.choice([2, 3]),
        "day":              random.choice([5, 6] if is_weekend else [0,1,2,3,4]),
        "is_weekend":       is_weekend,
        "is_business":      is_business,
        "airline":          airline,
        "from_city":        city_pair[0],
        "to_city":          city_pair[1],
    })

print("Done — 200 predictions logged")
