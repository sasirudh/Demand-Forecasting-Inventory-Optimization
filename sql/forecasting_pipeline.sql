-- =====================================================================
-- Retail demand forecasting: SQL data pipeline  (SQLite dialect)
--
-- Expects a raw table `transactions` with columns:
--   transaction_id, transaction_date (ISO 'YYYY-MM-DD' text), store_id,
--   unit_price, quantity_sold, promotion_applied (0/1), holiday_indicator (0/1)
--
-- Steps:
--   1. store_daily       : transaction -> store x day aggregation
--   2. store_daily_full  : gap-filled calendar spine (missing days = 0 sales)
--   3. model_features    : lag, rolling-window and calendar features
--   4. safety_stock      : Q* = D_hat + Z * sigma_e per store (95% service level)
-- =====================================================================

DROP VIEW IF EXISTS store_daily;
DROP VIEW IF EXISTS store_daily_full;
DROP VIEW IF EXISTS model_features;
DROP VIEW IF EXISTS safety_stock;

-- ---------------------------------------------------------------------
-- 1. Store x day aggregation
-- ---------------------------------------------------------------------
CREATE VIEW store_daily AS
SELECT
    store_id,
    transaction_date,
    SUM(quantity_sold)               AS total_volume,
    SUM(unit_price * quantity_sold)  AS total_revenue,
    COUNT(transaction_id)            AS transaction_count,
    MAX(promotion_applied)           AS promotion_applied,
    MAX(holiday_indicator)           AS holiday_indicator
FROM transactions
GROUP BY store_id, transaction_date;

-- ---------------------------------------------------------------------
-- 2. Complete calendar spine: every store gets every day in the range.
--    Days with no transactions become zero-sales rows, so lags and
--    rolling windows below are true calendar-day offsets, not row offsets.
-- ---------------------------------------------------------------------
CREATE VIEW store_daily_full AS
WITH RECURSIVE bounds AS (
    SELECT MIN(transaction_date) AS d0, MAX(transaction_date) AS d1
    FROM transactions
),
calendar(d) AS (
    SELECT d0 FROM bounds
    UNION ALL
    SELECT date(d, '+1 day') FROM calendar, bounds WHERE d < d1
),
stores AS (
    SELECT DISTINCT store_id FROM transactions
)
SELECT
    s.store_id,
    c.d                                         AS transaction_date,
    COALESCE(sd.total_volume, 0)                AS total_volume,
    COALESCE(sd.total_revenue, 0)               AS total_revenue,
    COALESCE(sd.transaction_count, 0)           AS transaction_count,
    COALESCE(sd.promotion_applied, 0)           AS promotion_applied,
    COALESCE(sd.holiday_indicator, 0)           AS holiday_indicator
FROM stores s
CROSS JOIN calendar c
LEFT JOIN store_daily sd
       ON sd.store_id = s.store_id
      AND sd.transaction_date = c.d;

-- ---------------------------------------------------------------------
-- 3. Feature table for the models.
--    All lag / rolling features use ONLY past days (no leakage):
--    the rolling windows end at 1 PRECEDING, i.e. they exclude today.
--    SQLite has no STDDEV, so sample std is built from sums:
--        var = (sum(x^2) - sum(x)^2 / n) / (n - 1)
-- ---------------------------------------------------------------------
CREATE VIEW model_features AS
WITH lagged AS (
    SELECT
        store_id,
        transaction_date,
        total_revenue,
        total_volume,
        promotion_applied,
        holiday_indicator,

        LAG(total_revenue, 1)  OVER w AS revenue_lag_1,
        LAG(total_revenue, 7)  OVER w AS revenue_lag_7,
        LAG(total_revenue, 28) OVER w AS revenue_lag_28,

        AVG(total_revenue) OVER w7  AS rolling_mean_7d,
        SUM(total_revenue * total_revenue) OVER w7 AS sum_sq_7d,
        SUM(total_revenue) OVER w7  AS sum_7d,
        COUNT(total_revenue) OVER w7 AS n_7d
    FROM store_daily_full
    WINDOW
        w   AS (PARTITION BY store_id ORDER BY transaction_date),
        w7  AS (PARTITION BY store_id ORDER BY transaction_date
                ROWS BETWEEN 7 PRECEDING AND 1 PRECEDING)
)
SELECT
    store_id,
    transaction_date,
    total_revenue,
    total_volume,
    promotion_applied,
    holiday_indicator,

    -- calendar features (0 = Monday ... 6 = Sunday, matching pandas dayofweek)
    (CAST(strftime('%w', transaction_date) AS INTEGER) + 6) % 7  AS day_of_week,
    CASE WHEN (CAST(strftime('%w', transaction_date) AS INTEGER) + 6) % 7 >= 5
         THEN 1 ELSE 0 END                                       AS is_weekend,
    CAST(strftime('%m', transaction_date) AS INTEGER)            AS month,
    CAST(strftime('%d', transaction_date) AS INTEGER)            AS day_of_month,

    revenue_lag_1,
    revenue_lag_7,
    revenue_lag_28,
    rolling_mean_7d,
    CASE WHEN n_7d >= 2
         THEN SQRT(MAX((sum_sq_7d - sum_7d * sum_7d / n_7d) / (n_7d - 1), 0))
    END                                                          AS rolling_std_7d
FROM lagged
WHERE revenue_lag_28 IS NOT NULL      -- drop warm-up rows without full history
  AND n_7d = 7;

-- ---------------------------------------------------------------------
-- 4. Safety stock / order-up-to level.
--    Requires a table of model output written back from Python:
--      lgbm_forecasts(store_id, transaction_date, actual, d_hat)
--    e.g. eval_df[['store_id','Date','Actual','D_hat']]
--           .rename(columns={'Date':'transaction_date','Actual':'actual',
--                            'D_hat':'d_hat'})
--           .to_sql('lgbm_forecasts', conn, if_exists='replace', index=False)
--
--    sigma_e = sample std of forecast errors (actual - d_hat) per store.
--    Z = 1.645 for a 95% service level (one-sided normal).
-- ---------------------------------------------------------------------
CREATE VIEW safety_stock AS
WITH errs AS (
    SELECT store_id, transaction_date, actual, d_hat,
           actual - d_hat AS err
    FROM lgbm_forecasts
),
sigma AS (
    SELECT
        store_id,
        COUNT(*) AS n,
        SQRT(MAX((SUM(err * err) - SUM(err) * SUM(err) / COUNT(*))
                 / (COUNT(*) - 1), 0)) AS sigma_e
    FROM errs
    GROUP BY store_id
    HAVING COUNT(*) > 1
)
SELECT
    e.store_id,
    e.transaction_date,
    e.actual,
    e.d_hat,
    s.sigma_e,
    1.645 * s.sigma_e                                   AS safety_stock,
    MAX(e.d_hat + 1.645 * s.sigma_e, 0)                 AS q_star,
    CASE WHEN e.actual <= MAX(e.d_hat + 1.645 * s.sigma_e, 0)
         THEN 1 ELSE 0 END                              AS covered
FROM errs e
JOIN sigma s USING (store_id);

-- ---------------------------------------------------------------------
-- Example queries
-- ---------------------------------------------------------------------
-- Achieved service level per store (target = 0.95):
--   SELECT store_id, ROUND(AVG(covered), 3) AS achieved_service_level,
--          ROUND(AVG(q_star), 0) AS avg_order_level
--   FROM safety_stock GROUP BY store_id;
