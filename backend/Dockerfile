FROM node:20-bookworm

# Python para el import_excel.py
RUN apt-get update && apt-get install -y python3 python3-pip && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy dependency definitions
COPY backend/package.json /app/backend/package.json
COPY backend/tsconfig.json /app/backend/tsconfig.json

WORKDIR /app/backend
RUN npm install

# Install Python dependencies for data import script
COPY backend/scripts/requirements.txt /app/backend/scripts/requirements.txt
# Using --break-system-packages because we are in a container and want these available globally
RUN pip3 install --break-system-packages -r /app/backend/scripts/requirements.txt

WORKDIR /app
COPY . /app

WORKDIR /app/backend

# Default command for development
CMD ["npm", "run", "dev"]
