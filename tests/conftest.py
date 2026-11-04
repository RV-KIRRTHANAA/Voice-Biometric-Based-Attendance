import os

# Ensure JWT_SECRET_KEY is defined during pytest runs without requiring external environment setup
os.environ.setdefault("JWT_SECRET_KEY", "test_suite_jwt_secret_key_for_testing_purposes_only_32b")

