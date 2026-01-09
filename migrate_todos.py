import sqlite3
import os

DB_PATH = 'instance/StudyVerse.db'

def migrate():
    if not os.path.exists(DB_PATH):
        # Fallback for old structure if instance folder not used
        db_path_alt = 'StudyVerse.db'
        if os.path.exists(db_path_alt):
            conn = sqlite3.connect(db_path_alt)
        else:
            print("Database not found!")
            return
    else:
        conn = sqlite3.connect(DB_PATH)
    
    cursor = conn.cursor()
    
    try:
        # Add group_id column
        cursor.execute("ALTER TABLE todo ADD COLUMN group_id INTEGER DEFAULT NULL REFERENCES 'group'(id)")
        print("Added group_id column.")
    except sqlite3.OperationalError as e:
        print(f"Skipping group_id: {e}")

    try:
        # Add completed_by column
        cursor.execute("ALTER TABLE todo ADD COLUMN completed_by TEXT DEFAULT ''")
        print("Added completed_by column.")
    except sqlite3.OperationalError as e:
        print(f"Skipping completed_by: {e}")

    conn.commit()
    conn.close()
    print("Migration complete.")

if __name__ == "__main__":
    migrate()
