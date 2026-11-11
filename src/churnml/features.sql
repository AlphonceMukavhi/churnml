-- Same features as churnml.features.build_features, computed inside PostgreSQL.
SELECT
    c.customer_id,
    c.plan,
    c.country,
    c.age,
    c.monthly_fee::float8                                   AS monthly_fee,
    (%(as_of)s::date - c.signup_date)                       AS tenure_days,
    (AVG(u.sessions) FILTER (
        WHERE u.month >= %(as_of)s::date - INTERVAL '3 months'))::float8
                                                            AS sessions_last3,
    (AVG(u.sessions) FILTER (
        WHERE u.month <  %(as_of)s::date - INTERVAL '3 months'))::float8
                                                            AS sessions_prev3,
    SUM(u.support_tickets)::int                             AS tickets_total,
    SUM(u.payment_failures)::int                            AS failures_total
FROM customers AS c
JOIN usage_monthly AS u ON u.customer_id = c.customer_id
WHERE u.month < %(as_of)s::date
GROUP BY c.customer_id
ORDER BY c.customer_id
