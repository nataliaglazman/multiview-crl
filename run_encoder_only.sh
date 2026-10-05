python -m unittest tests.test_encoder_runtime -v &&
python scripts/compare_encoders.py train --output-dir "$OUT" --only-seed 42 &&
python scripts/compare_encoders.py evaluate --output-dir "$OUT" --only-seed 42 &&
python scripts/compare_encoders.py summarize --output-dir "$OUT" --only-seed 42