#!/usr/bin/env bash
# Starts the ML service (port 8000) and the Spring Boot backend + UI (port 8080) on this machine.
# Prerequisites: Java 17, Python 3.10+ with ml-service/requirements.txt installed, PostgreSQL on localhost:5432
# with database deepfake_db / user deepfake / password deepfake (override with DB_URL, DB_USER, DB_PASSWORD).
set -euo pipefail
ROOT="$(cd "$(dirname "$0")" && pwd)"
LOGS="$ROOT/.logs"; mkdir -p "$LOGS"

cleanup() { echo; echo "stopping..."; kill "${ML_PID:-}" "${BE_PID:-}" 2>/dev/null || true; }
trap cleanup EXIT INT TERM

echo "▶ ML service  → http://localhost:8000  (log: .logs/ml-service.log)"
( cd "$ROOT/ml-service" && DEVICE="${DEVICE:-cpu}" python3 -m uvicorn app.main:app --host 0.0.0.0 --port 8000 ) >"$LOGS/ml-service.log" 2>&1 &
ML_PID=$!

echo "▶ Backend+UI  → http://localhost:8080  (log: .logs/backend.log)"
# PostgreSQL < 12 needs Hibernate's legacy dialect (profile "legacypg"); detect the server version automatically.
PG_MAJOR=$(PGPASSWORD="${DB_PASSWORD:-deepfake}" psql -h localhost -p 5432 -U "${DB_USER:-deepfake}" -d "${DB_NAME:-deepfake_db}" -Atc "show server_version" 2>/dev/null | cut -d. -f1)
if [ -n "$PG_MAJOR" ] && [ "$PG_MAJOR" -lt 12 ]; then export SPRING_PROFILES_ACTIVE="${SPRING_PROFILES_ACTIVE:-legacypg}"; echo "  (PostgreSQL $PG_MAJOR detected → profile legacypg)"; fi
( cd "$ROOT/backend" && ./mvnw -q spring-boot:run ) >"$LOGS/backend.log" 2>&1 &
BE_PID=$!

for i in $(seq 1 90); do
  ml=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8000/health || true)
  be=$(curl -s -o /dev/null -w '%{http_code}' http://localhost:8080/api/health || true)
  [ "$ml" = "200" ] && [ "$be" = "200" ] && break
  sleep 2
done
echo "ML: $(curl -s http://localhost:8000/health)"
echo "Backend: $(curl -s http://localhost:8080/api/health | head -c 300)"
echo; echo "Open http://localhost:8080 — press Ctrl+C to stop both services."
wait
