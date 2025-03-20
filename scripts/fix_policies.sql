-- Drop existing policies
DROP POLICY IF EXISTS transactions_select_policy ON transactions;
DROP POLICY IF EXISTS transactions_insert_policy ON transactions;
DROP POLICY IF EXISTS model_versions_select_policy ON model_versions;
DROP POLICY IF EXISTS model_versions_insert_policy ON model_versions;
DROP POLICY IF EXISTS users_select_policy ON users;
DROP POLICY IF EXISTS users_insert_policy ON users;
DROP POLICY IF EXISTS api_keys_select_policy ON api_keys;
DROP POLICY IF EXISTS api_keys_insert_policy ON api_keys;

-- Simplified policies for development
CREATE POLICY "Enable read access for all users"
ON transactions FOR SELECT
USING (true);

CREATE POLICY "Enable insert access for all users"
ON transactions FOR INSERT
WITH CHECK (true);

CREATE POLICY "Enable read access for all users"
ON model_versions FOR SELECT
USING (true);

CREATE POLICY "Enable insert access for all users"
ON model_versions FOR INSERT
WITH CHECK (true);

CREATE POLICY "Enable read access for all users"
ON users FOR SELECT
USING (true);

CREATE POLICY "Enable insert access for all users"
ON users FOR INSERT
WITH CHECK (true);

CREATE POLICY "Enable read access for all users"
ON api_keys FOR SELECT
USING (true);

CREATE POLICY "Enable insert access for all users"
ON api_keys FOR INSERT
WITH CHECK (true);

-- Note: In production, you would want more restrictive policies
-- This is just for development purposes 