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


### Phase 3 — Preprocessing ✅
**Status:** Complete

#### Tasks completed:
- [x] Loaded flights_cleaned.csv (300,259 rows × 12 columns)
- [x] Split into train (64%), val (16%), test (20%) before any encoding
- [x] Applied Target Encoding on airline, from, to using y_train means only
- [x] Dropped original string columns after encoding
- [x] Confirmed zero nulls after encoding
- [x] No scaling applied — XGBoost/LightGBM are scale invariant
- [x] Saved all 6 splits to data/processed/

#### Key decisions:
- Split before encoding to prevent data leakage
- Used simple target encoding over LeaveOneOut — negligible difference at 300K rows
- Skipped scaling — unnecessary for tree based models
- Validation set created for early stopping during model training

#### Final feature set (11 columns):
- stops_numeric, duration_minutes, departure_hour, arrival_hour
- month, day, is_weekend, is_business
- airline_encoded, from_encoded, to_encoded
