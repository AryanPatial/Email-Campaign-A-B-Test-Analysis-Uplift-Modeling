#!/bin/sh
# Runs the whole project from the raw CSV. Takes about 20 seconds.
set -e
cd "$(dirname "$0")"
python src/01_build_db.py
python src/02_ab_test.py
python src/03_uplift_model.py
python src/04_charts.py
