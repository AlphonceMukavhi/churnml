-- churnml schema. Safe to re-run: it drops and recreates the tables.
DROP TABLE IF EXISTS labels, usage_monthly, customers CASCADE;

CREATE TABLE customers (
    customer_id  integer PRIMARY KEY,
    signup_date  date NOT NULL,
    plan         text NOT NULL CHECK (plan IN ('basic', 'standard', 'premium')),
    country      text NOT NULL,
    age          integer NOT NULL CHECK (age BETWEEN 16 AND 100),
    monthly_fee  numeric(8, 2) NOT NULL CHECK (monthly_fee >= 0)
);

CREATE TABLE usage_monthly (
    customer_id       integer NOT NULL REFERENCES customers (customer_id),
    month             date NOT NULL,
    sessions          integer NOT NULL CHECK (sessions >= 0),
    support_tickets   integer NOT NULL CHECK (support_tickets >= 0),
    payment_failures  integer NOT NULL CHECK (payment_failures >= 0),
    PRIMARY KEY (customer_id, month)
);

CREATE TABLE labels (
    customer_id  integer PRIMARY KEY REFERENCES customers (customer_id),
    churned      boolean NOT NULL
);

CREATE INDEX usage_monthly_month_idx ON usage_monthly (month);
