import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT

try:
    # Connect to the default 'postgres' database
    conn = psycopg2.connect(
        dbname="postgres",
        user="postgres",
        password="Daksh@007",
        host="localhost",
        port="5432"
    )

    conn.set_isolation_level(ISOLATION_LEVEL_AUTOCOMMIT)
    cur = conn.cursor()

    # Check if database exists
    cur.execute("SELECT 1 FROM pg_database WHERE datname = 'StudyVerse_Final'")
    exists = cur.fetchone()

    if not exists:
        print("Database does not exist. Creating...")
        cur.execute('CREATE DATABASE "StudyVerse_Final";')
        print("Database 'StudyVerse_Final' created successfully.")
    else:
        print("Database 'StudyVerse_Final' already exists.")

except Exception as e:
    print(f"Error: {e}")
finally:
    if 'cur' in locals() and cur:
        cur.close()
    if 'conn' in locals() and conn:
        conn.close()
