#!/bin/bash
set -e

# Check if AGENT_TYPE is set
if [ -z "$AGENT_TYPE" ]; then
    echo "[ERROR] AGENT_TYPE environment variable is not set"
    echo "[ERROR] Available agent types should have corresponding {type}-requirements.txt files"
    exit 1
fi

echo "[INFO] Starting agent container for type: $AGENT_TYPE"

# Install agent-specific requirements
REQUIREMENTS_FILE="${AGENT_TYPE}-requirements.txt"
if [ -f "$REQUIREMENTS_FILE" ]; then
    echo "[INFO] Installing $AGENT_TYPE specific requirements..."
    pip3 install --no-cache-dir -r "$REQUIREMENTS_FILE"
else
    echo "[ERROR] Requirements file not found: $REQUIREMENTS_FILE"
    echo "[ERROR] Available requirements files:"
    ls -la *-requirements.txt 2>/dev/null || echo "[ERROR] No requirements files found"
    exit 1
fi

echo "[INFO] Starting MySQL..."
service mysql start

echo "[INFO] Waiting for MySQL to be ready..."
until mysqladmin ping -h localhost --silent; do
    sleep 1
done

echo "[INFO] Fixing root authentication..."
mysql -u root -p"$MYSQL_ROOT_PASSWORD" -e \
  "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '$MYSQL_ROOT_PASSWORD'; FLUSH PRIVILEGES;"

if [ -f /docker-entrypoint-initdb.d/init.sql ]; then
    echo "[INFO] Running init.sql..."
    mysql -u root -p"$MYSQL_ROOT_PASSWORD" < /docker-entrypoint-initdb.d/init.sql
fi

echo "[INFO] Running ${AGENT_TYPE} Agent..."
exec python3 -m "smocs.agents.${AGENT_TYPE}_agent" --agent_config "$AGENT_CONFIG"