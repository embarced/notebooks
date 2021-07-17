curl -d '{"instances": [[50.0, 122.0], [48, 100], [30, 150]]}' \
    -X POST http://localhost:8501/v1/models/insurance:predict