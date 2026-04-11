"""
File: src/nfl/db/connection.py

Database connection functions for the NFL predictions database.

Usage:
    from src.nfl.db.connection import get_connection, get_engine

    # For raw SQL queries (psycopg2)
    conn = get_connection()
    cur = conn.cursor()
    cur.execute("SELECT * FROM weekly_stats WHERE player_id = %s", (player_id,))
    conn.close()

    # For pandas operations (SQLAlchemy)
    engine = get_engine()
    df.to_sql('weekly_stats', engine, if_exists='append', index=False)
    df = pd.read_sql("SELECT * FROM games WHERE season = 2024", engine)
"""

import psycopg2
from sqlalchemy import create_engine
from src.nfl.db.config import DB_CONFIG


def get_connection():
    """
    Get a psycopg2 connection to the NFL predictions database.
    Caller is responsible for closing the connection.

    Returns:
        psycopg2 connection object
    """
    return psycopg2.connect(**DB_CONFIG)


def get_engine():
    """
    Get a SQLAlchemy engine for the NFL predictions database.
    Used by pandas for to_sql() and read_sql() operations.

    Returns:
        SQLAlchemy Engine object
    """
    url = f"postgresql://{DB_CONFIG['user']}:{DB_CONFIG['password']}@{DB_CONFIG['host']}:{DB_CONFIG['port']}/{DB_CONFIG['dbname']}"
    return create_engine(url)
