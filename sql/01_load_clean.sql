-- Load the raw CSV and make one clean table.
-- The raw file has no customer id, so I add one from the row number.

CREATE OR REPLACE TABLE customers AS
SELECT
    row_number() OVER ()                         AS customer_id,
    CASE segment
        WHEN 'No E-Mail'     THEN 'control'
        WHEN 'Mens E-Mail'   THEN 'mens_email'
        WHEN 'Womens E-Mail' THEN 'womens_email'
    END                                          AS arm,
    recency,                                     -- months since last purchase (1-12)
    history,                                     -- $ spent in the past year
    history_segment,
    mens,                                        -- bought men's items last year (0/1)
    womens,                                      -- bought women's items last year (0/1)
    CASE
        WHEN mens = 1 AND womens = 1 THEN 'both'
        WHEN mens = 1                THEN 'mens_only'
        ELSE                              'womens_only'
    END                                          AS buyer_type,
    -- the raw file spells it "Surburban"
    replace(zip_code, 'Surburban', 'Suburban')   AS zip_area,
    newbie,                                      -- new customer in the past 12 months (0/1)
    channel,                                     -- how they bought last year
    visit,                                       -- visited the site in the 2 weeks after (0/1)
    conversion,                                  -- bought in the 2 weeks after (0/1)
    spend                                        -- $ spent in the 2 weeks after
FROM read_csv_auto('data/raw/hillstrom.csv');
