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

CREATE TABLE IF NOT EXISTS agent_inferences (
    id INT AUTO_INCREMENT PRIMARY KEY, 
    state_source_timestamp DATETIME(6) NOT NULL, 
    state_received_timestamp DATETIME(6) NOT NULL,
    state BLOB NOT NULL, 
    prediction_timestamp DATETIME(6), 
    prediction BLOB
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