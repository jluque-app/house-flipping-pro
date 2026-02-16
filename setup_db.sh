#!/bin/bash

# Start Postgres
echo "Starting PostgreSQL..."
brew services start postgresql@17
sleep 5

# Create User
echo "Creating user 'bcn'..."
createuser -s bcn || echo "User 'bcn' likely exists"
psql -d postgres -c "ALTER USER bcn WITH PASSWORD 'bcn';"
psql -d postgres -c "ALTER USER bcn WITH PASSWORD 'bcn';"

# Create DB
echo "Creating database 'bcn'..."
createdb -O bcn bcn || echo "Database 'bcn' likely exists"

# Enable PostGIS
echo "Enabling PostGIS..."
psql -d bcn -c "CREATE EXTENSION IF NOT EXISTS postgis;"

# Run Migrations
echo "Running migrations..."
cd backend
npm run migrate

# Import Data
echo "Importing data..."
python3 scripts/import_excel.py

echo "Database setup complete."
