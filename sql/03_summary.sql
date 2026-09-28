-- Results per test group (arm). This is the table the A/B test starts from.

CREATE OR REPLACE TABLE arm_summary AS
SELECT
    arm,
    count(*)                         AS customers,
    sum(visit)::INT                  AS visits,
    sum(conversion)::INT             AS orders,
    round(sum(spend), 2)             AS revenue,
    round(avg(visit), 4)             AS visit_rate,
    round(avg(conversion), 4)        AS conversion_rate,
    round(avg(spend), 4)             AS spend_per_customer
FROM customers
GROUP BY arm
ORDER BY arm;

-- Same thing, split by what the customer bought last year.
-- This is where the Womens email problem shows up.

CREATE OR REPLACE TABLE buyer_type_summary AS
SELECT
    buyer_type,
    arm,
    count(*)                         AS customers,
    round(avg(visit), 4)             AS visit_rate,
    round(avg(conversion), 4)        AS conversion_rate,
    round(avg(spend), 4)             AS spend_per_customer
FROM customers
GROUP BY buyer_type, arm
ORDER BY buyer_type, arm;
