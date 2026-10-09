#!/usr/bin/env bash
# Start the Chronos Transit Engine on http://127.0.0.1:8765 (first run installs dependencies and trains the CNN).
set -euo pipefail
cd "$(dirname "$0")"

if [ ! -x .venv/bin/python ]; then
  python3 -m venv .venv
  .venv/bin/pip install -q --upgrade pip
  .venv/bin/pip install -q -r requirements.txt
fi
.venv/bin/python manage.py migrate --noinput >/dev/null
if [ ! -f "${CTE_DATA_DIR:-data}/models/cte_cnn.pt" ]; then
  echo "No CNN checkpoint found - training (about 10 minutes, one time)..."
  .venv/bin/python manage.py train_model
fi
echo "Chronos Transit Engine -> http://127.0.0.1:${CTE_PORT:-8765}"
exec .venv/bin/python manage.py runserver "127.0.0.1:${CTE_PORT:-8765}" --noreload
