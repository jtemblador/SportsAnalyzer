"""
File: src/nfl/db/config.py

Database configuration loaded from environment variables / .env file.
"""

import os
from pathlib import Path
from dotenv import load_dotenv

# Load .env from project root
load_dotenv(Path(__file__).parent.parent.parent.parent / '.env')

DB_CONFIG = {
    'dbname': os.getenv('DB_NAME', 'nfl_predictions'),
    'user': os.getenv('DB_USER', 'j0e'),
    'password': os.getenv('DB_PASSWORD', 'nfl'),
    'host': os.getenv('DB_HOST', 'localhost'),
    'port': os.getenv('DB_PORT', '5432'),
}
