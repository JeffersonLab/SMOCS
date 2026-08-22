-- Select database to use
CREATE DATABASE IF NOT EXISTS agentdb;

USE agentdb;

-- Create a tables

CREATE TABLE IF NOT EXISTS agent_information (
    id INT AUTO_INCREMENT PRIMARY KEY, 
    registered_id VARCHAR(50) NOT NULL, 
    agent_name VARCHAR(50), 
    config BLOB, 
    info BLOB
);

-- context is a single, fixed BLOB column shared by every agent type, rather than a
-- set of individually-typed, per-agent columns: it holds whatever numpy float array
-- an agent's ingest thread chooses to record as that row's "operating context" (see
-- DBManager.record_sensor_data / _compute_block_id in smocs/db/mysql_api_v0.py),
-- serialized the same way as state/prediction, or is simply NULL for any agent with
-- no notion of context at all (for example, the plain autoencoder or the RL control
-- agent). Because this column's meaning is entirely opaque to the database - only
-- the owning agent interprets the floats inside it - no per-agent schema migration
-- is ever needed to add it: it already exists, for every agent, from this file
-- alone. The two indexes below are declared inline, as part of this same statement,
-- rather than as separate CREATE INDEX statements after it, specifically because
-- this entire init.sql script is re-run by the agent container's entrypoint on
-- every container start, not only on first-ever provisioning - a bare CREATE INDEX
-- would fail with a duplicate-key-name error on the second and every subsequent
-- run, whereas indexes declared inline are simply skipped, along with the rest of
-- this statement, once IF NOT EXISTS finds the table already present.
CREATE TABLE IF NOT EXISTS agent_inferences (
    id INT AUTO_INCREMENT PRIMARY KEY,
    state_source_timestamp DATETIME(6) NOT NULL,
    state_received_timestamp DATETIME(6) NOT NULL,
    state BLOB NOT NULL,
    prediction_timestamp DATETIME(6),
    prediction BLOB,
    block_id INT NOT NULL DEFAULT 0,
    context BLOB,
    INDEX idx_state_source_timestamp (state_source_timestamp),
    INDEX idx_block_id (block_id)
);

CREATE TABLE IF NOT EXISTS agent_replay (
    id INT AUTO_INCREMENT PRIMARY KEY,
    state_id INT NOT NULL,
    action_success BOOL,
    reward BLOB NOT NULL,
    next_state_source_timestamp DATETIME(6) NOT NULL,
    next_state_received_timestamp DATETIME(6) NOT NULL,
    next_state BLOB NOT NULL,
    terminate BOOL NOT NULL,
    truncate BOOL NOT NULL,
    info BLOB,
    FOREIGN KEY (state_id) REFERENCES agent_inferences(id)
);;


-- Optional: grant privileges to non-root user
-- GRANT ALL PRIVILEGES ON agentdb.* TO 'admin'@'localhost';