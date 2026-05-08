### Phase 2 — Data Ingestion & EDA ✅
**Status:** Complete

#### Tasks completed:
- [x] Loaded business.csv (93,487 rows) and economy.csv (206,772 rows)
- [x] Verified zero null values in both datasets
- [x] Dropped 2 duplicates from economy
- [x] Converted price from string with commas to float
- [x] Converted date to datetime, extracted month, day_of_week, is_weekend
- [x] Cleaned stop column (dirty \n\t values), encoded to 0/1/2
- [x] Converted time_taken to duration_minutes integer
- [x] Extracted departure_hour and arrival_hour from dep_time/arr_time
- [x] Dropped irrelevant columns: ch_code, num_code, date, dep_time, arr_time, time_taken, stop
- [x] Added is_business column (1=business, 0=economy)
- [x] Merged both datasets → 300,259 rows × 12 columns
- [x] Saved to data/processed/flights_cleaned.csv

#### Key findings:
- Business prices: 12,000 → 123,071 (mean 52,540)
- Economy prices: 1,105 → 42,349 (mean 6,571)
- Only Air India and Vistara operate business class
- More stops = higher price in both classes
- Month affects economy prices strongly (Feb 2x March) but not business
- 6 cities: Delhi, Mumbai, Bangalore, Kolkata, Hyderabad, Chennai
- Dataset covers only 49 days (Feb 11 → Mar 31 2022) — limited seasonality

#### Decisions made:
- Used business.csv + economy.csv instead of Clean_Dataset.csv — to keep raw date and engineer features ourselves
- Kept apparent post-merge duplicates — they represent real flights, not data errors
- Did NOT encode or normalize — must happen after train/test split to prevent data leakage
