-- SQL schema for the Fraud Detection API in Supabase

-- Enable Row Level Security (RLS)
CREATE EXTENSION IF NOT EXISTS "uuid-ossp";

-- Transactions table
CREATE TABLE IF NOT EXISTS transactions (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    transaction_id TEXT UNIQUE,
    amount DECIMAL NOT NULL,
    merchant_category TEXT NOT NULL,
    merchant_name TEXT NOT NULL,
    transaction_type TEXT NOT NULL,
    card_present BOOLEAN NOT NULL,
    country TEXT NOT NULL,
    timestamp TIMESTAMPTZ NOT NULL,
    is_fraud BOOLEAN DEFAULT FALSE,
    prediction_score DECIMAL,
    model_version TEXT,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Model versions table
CREATE TABLE IF NOT EXISTS model_versions (
    id TEXT PRIMARY KEY,
    version TEXT NOT NULL UNIQUE,
    algorithm TEXT NOT NULL,
    performance_metrics JSONB NOT NULL,
    feature_importance JSONB,
    last_trained TIMESTAMPTZ NOT NULL,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- Users table
CREATE TABLE IF NOT EXISTS users (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    email TEXT UNIQUE NOT NULL,
    name TEXT,
    is_admin BOOLEAN DEFAULT FALSE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    updated_at TIMESTAMPTZ DEFAULT NOW()
);

-- API keys table
CREATE TABLE IF NOT EXISTS api_keys (
    id UUID PRIMARY KEY DEFAULT uuid_generate_v4(),
    user_id UUID REFERENCES users(id) ON DELETE CASCADE,
    key TEXT UNIQUE NOT NULL,
    name TEXT,
    is_active BOOLEAN DEFAULT TRUE,
    created_at TIMESTAMPTZ DEFAULT NOW(),
    expires_at TIMESTAMPTZ
);

-- Create indexes
CREATE INDEX IF NOT EXISTS idx_transactions_timestamp ON transactions(timestamp);
CREATE INDEX IF NOT EXISTS idx_transactions_merchant ON transactions(merchant_category, merchant_name);
CREATE INDEX IF NOT EXISTS idx_model_versions_created ON model_versions(created_at);

-- RLS Policies
ALTER TABLE transactions ENABLE ROW LEVEL SECURITY;
ALTER TABLE model_versions ENABLE ROW LEVEL SECURITY;
ALTER TABLE users ENABLE ROW LEVEL SECURITY;
ALTER TABLE api_keys ENABLE ROW LEVEL SECURITY;

-- Allow authenticated users to select transactions
CREATE POLICY transactions_select_policy ON transactions
    FOR SELECT USING (auth.role() = 'authenticated');

-- Allow authenticated users to insert transactions
CREATE POLICY transactions_insert_policy ON transactions
    FOR INSERT WITH CHECK (auth.role() = 'authenticated');

-- Allow authenticated users to select model versions
CREATE POLICY model_versions_select_policy ON model_versions
    FOR SELECT USING (auth.role() = 'authenticated');

-- Allow only admins to insert model versions
CREATE POLICY model_versions_insert_policy ON model_versions
    FOR INSERT WITH CHECK (
        auth.role() = 'authenticated' AND
        EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid() AND users.is_admin = TRUE
        )
    );

-- Allow users to see their own information
CREATE POLICY users_select_policy ON users
    FOR SELECT USING (auth.uid() = id OR 
        EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid() AND users.is_admin = TRUE
        )
    );

-- Allow only admins to manage users
CREATE POLICY users_insert_policy ON users
    FOR INSERT WITH CHECK (
        auth.role() = 'authenticated' AND
        EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid() AND users.is_admin = TRUE
        )
    );

-- Allow users to see their own API keys
CREATE POLICY api_keys_select_policy ON api_keys
    FOR SELECT USING (auth.uid() = user_id OR 
        EXISTS (
            SELECT 1 FROM users
            WHERE users.id = auth.uid() AND users.is_admin = TRUE
        )
    );

-- Allow authenticated users to create API keys for themselves
CREATE POLICY api_keys_insert_policy ON api_keys
    FOR INSERT WITH CHECK (
        auth.role() = 'authenticated' AND
        user_id = auth.uid()
    );

-- Demo seed function (for development only)
CREATE OR REPLACE FUNCTION reset_demo()
RETURNS void AS $$
BEGIN
    -- Clear existing data
    DELETE FROM transactions;
    DELETE FROM model_versions;
    
    -- Add demo model version
    INSERT INTO model_versions (id, version, algorithm, performance_metrics, last_trained)
    VALUES (
        'v2023.1.1',
        'v2023.1.1',
        'LightGBM',
        '{"accuracy": 0.95, "precision": 0.92, "recall": 0.85, "f1": 0.88, "auc": 0.97}',
        NOW()
    );
    
    -- Add demo transactions (would be more in a real implementation)
    INSERT INTO transactions (transaction_id, amount, merchant_category, merchant_name, transaction_type, card_present, country, timestamp, is_fraud)
    VALUES
        ('tx_demo_1', 123.45, 'retail', 'ACME Store', 'purchase', TRUE, 'US', NOW() - INTERVAL '1 day', FALSE),
        ('tx_demo_2', 1500.00, 'electronics', 'TechGiant', 'purchase', TRUE, 'US', NOW() - INTERVAL '2 days', FALSE),
        ('tx_demo_3', 50.00, 'restaurant', 'Good Eats', 'purchase', TRUE, 'US', NOW() - INTERVAL '3 days', FALSE),
        ('tx_demo_4', 899.99, 'travel', 'Luxury Hotels', 'purchase', FALSE, 'FR', NOW() - INTERVAL '5 days', TRUE),
        ('tx_demo_5', 25.00, 'entertainment', 'Cinema World', 'purchase', TRUE, 'UK', NOW() - INTERVAL '1 week', FALSE);
END;
$$ LANGUAGE plpgsql; 