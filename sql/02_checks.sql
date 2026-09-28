-- Data checks. Every row here should show 0 in "bad_rows".
-- If any check fails, the Python script stops.

SELECT 'row count is 64,000'              AS check_name, abs(count(*) - 64000) AS bad_rows FROM customers
UNION ALL
SELECT 'no missing values',
       count(*) FILTER (WHERE arm IS NULL OR recency IS NULL OR history IS NULL
                           OR visit IS NULL OR conversion IS NULL OR spend IS NULL)
FROM customers
UNION ALL
SELECT 'bought something but did not visit', count(*) FILTER (WHERE conversion = 1 AND visit = 0)      FROM customers
UNION ALL
SELECT 'spend > 0 but no purchase',          count(*) FILTER (WHERE spend > 0 AND conversion = 0)      FROM customers
UNION ALL
SELECT 'purchase but spend = 0',             count(*) FILTER (WHERE conversion = 1 AND spend = 0)      FROM customers
UNION ALL
SELECT 'recency outside 1-12',               count(*) FILTER (WHERE recency NOT BETWEEN 1 AND 12)      FROM customers
UNION ALL
SELECT 'negative spend or history',          count(*) FILTER (WHERE spend < 0 OR history < 0)          FROM customers
UNION ALL
SELECT 'bought neither mens nor womens',     count(*) FILTER (WHERE mens = 0 AND womens = 0)           FROM customers;
