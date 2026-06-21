FROM python:3.11-slim

WORKDIR /app

RUN apt-get update \
    && apt-get install -y --no-install-recommends gcc \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# App code + models (no traffic.db — using Supabase now)
COPY . .

# Remove any stale SQLite db that may have been bundled
RUN rm -f traffic.db

RUN chmod +x entrypoint.sh

EXPOSE 5000

CMD ["./entrypoint.sh"]
