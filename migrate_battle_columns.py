"""
migrate_battle_columns.py
=========================
Migration script to add the 3 new Byte Battle columns to the `user` table:
  - battle_xp      INTEGER DEFAULT 0
  - battle_wins     INTEGER DEFAULT 0
  - battle_losses   INTEGER DEFAULT 0

Usage (local or on Render shell):
    python migrate_battle_columns.py

It is SAFE to run multiple times — it checks whether each column already
exists before attempting to add it (no duplicate-column errors).
"""

import os
from dotenv import load_dotenv

load_dotenv()

# ── Import app so we get the configured db / engine ──────────────────────────
from app import app, db

NEW_COLUMNS = [
    ("battle_xp",     "INTEGER DEFAULT 0"),
    ("battle_wins",   "INTEGER DEFAULT 0"),
    ("battle_losses", "INTEGER DEFAULT 0"),
]

def column_exists(conn, table: str, column: str) -> bool:
    """Return True if *column* already exists in *table*."""
    # Works for both PostgreSQL and SQLite
    db_type = conn.engine.dialect.name
    if db_type == "postgresql":
        result = conn.execute(
            db.text(
                "SELECT 1 FROM information_schema.columns "
                "WHERE table_name = :tbl AND column_name = :col"
            ),
            {"tbl": table, "col": column},
        )
    else:  # sqlite
        result = conn.execute(db.text(f"PRAGMA table_info({table})"))
        rows = result.fetchall()
        return any(row[1] == column for row in rows)
    return result.fetchone() is not None


def run_migration():
    with app.app_context():
        with db.engine.connect() as conn:
            for col_name, col_def in NEW_COLUMNS:
                if column_exists(conn, "user", col_name):
                    print(f"  ✅  Column '{col_name}' already exists — skipping.")
                else:
                    print(f"  ➕  Adding column '{col_name}' ({col_def}) …")
                    conn.execute(
                        db.text(f'ALTER TABLE "user" ADD COLUMN {col_name} {col_def}')
                    )
                    print(f"  ✅  Column '{col_name}' added successfully.")
            conn.commit()
        print("\n🎉  Migration complete. All battle columns are present.")


if __name__ == "__main__":
    run_migration()
