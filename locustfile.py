import random
from locust import HttpUser, task, between, constant

ECONOMY_AIRLINES = ["IndiGo", "SpiceJet", "GO FIRST", "AirAsia", "StarAir", "Trujet"]
BUSINESS_AIRLINES = ["Air India", "Vistara"]
CITIES = ["Delhi", "Mumbai", "Bangalore", "Chennai", "Kolkata", "Hyderabad"]
MONTHS = [2, 3]
WEEKDAYS = [0, 1, 2, 3, 4]    # Monday–Friday → is_weekend=0
WEEKEND_DAYS = [5, 6]          # Saturday, Sunday → is_weekend=1


def random_city_pair():
    from_city = random.choice(CITIES)
    to_city   = random.choice([c for c in CITIES if c != from_city])
    return from_city, to_city


def economy_payload():
    from_city, to_city = random_city_pair()
    is_weekend = random.choice([0, 1])
    day = random.choice(WEEKEND_DAYS if is_weekend else WEEKDAYS)
    return {
        "stops_numeric":    random.choice([0, 1, 2]),
        "duration_minutes": random.randint(60, 600),
        "departure_hour":   random.randint(0, 23),
        "arrival_hour":     random.randint(0, 23),
        "month":            random.choice(MONTHS),
        "day":              day,
        "is_weekend":       is_weekend,
        "is_business":      0,
        "airline":          random.choice(ECONOMY_AIRLINES),
        "from_city":        from_city,
        "to_city":          to_city,
    }


def business_payload():
    from_city, to_city = random_city_pair()
    is_weekend = random.choice([0, 1])
    day = random.choice(WEEKEND_DAYS if is_weekend else WEEKDAYS)
    return {
        "stops_numeric":    random.choice([0, 1, 2]),
        "duration_minutes": random.randint(60, 600),
        "departure_hour":   random.randint(0, 23),
        "arrival_hour":     random.randint(0, 23),
        "month":            random.choice(MONTHS),
        "day":              day,
        "is_weekend":       is_weekend,
        "is_business":      1,
        "airline":          random.choice(BUSINESS_AIRLINES),
        "from_city":        from_city,
        "to_city":          to_city,
    }


class EconomyUser(HttpUser):
    weight = 6
    wait_time = between(1, 3)

    @task(4)
    def predict_economy(self):
        self.client.post(
            "/predict",
            json=economy_payload(),
            name="/predict [economy]",
        )

    @task(1)
    def health_check(self):
        self.client.get("/health", name="/health")


class BusinessUser(HttpUser):
    weight = 2
    wait_time = between(2, 5)

    @task
    def predict_business(self):
        self.client.post(
            "/predict",
            json=business_payload(),
            name="/predict [business]",
        )


class BatchUser(HttpUser):
    weight = 2
    wait_time = between(1, 3)    # ← was constant(0), now has wait time

    @task(1)
    def predict_small_batch(self):
        self.client.post(
            "/predict/batch",
            json={"flights": [economy_payload() for _ in range(5)]},
            name="/predict/batch [5 flights]",
        )

    @task(1)
    def predict_large_batch(self):
        self.client.post(
            "/predict/batch",
            json={"flights": [economy_payload() for _ in range(50)]},
            name="/predict/batch [50 flights]",
        )