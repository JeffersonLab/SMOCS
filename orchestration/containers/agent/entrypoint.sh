#!/bin/bash
set -e

echo "[INFO] Starting MySQL..."
service mysql start

echo "[INFO] Waiting for MySQL to be ready..."
until mysqladmin ping -h localhost --silent; do
    sleep 1
done

# Optional: initialize schema
echo "[INFO] Fixing root authentication..."
mysql -u root -p"$MYSQL_ROOT_PASSWORD" -e \
  "ALTER USER 'root'@'localhost' IDENTIFIED WITH mysql_native_password BY '$MYSQL_ROOT_PASSWORD'; FLUSH PRIVILEGES;"

if [ -f /docker-entrypoint-initdb.d/init.sql ]; then
    echo "[INFO] Running init.sql..."
    mysql -u root -p"$MYSQL_ROOT_PASSWORD"< /docker-entrypoint-initdb.d/init.sql
fi

# echo "[PRINT] MySQL Root Password: $MYSQL_ROOT_PASSWORD"
echo "[INFO] Running Agent..."
python3 /app/smocs/agents/db_demo_agent.py


tail -f /dev/null