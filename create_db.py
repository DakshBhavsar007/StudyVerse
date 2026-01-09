import psycopg2
from psycopg2.extensions import ISOLATION_LEVEL_AUTOCOMMIT
import os
from dotenv import load_dotenv

load_dotenv()

# Get DB URL from env or use default local
DATABASE_URL = os.getenv("DATABASE_URL")

def create_database():
    try:
        if DATABASE_URL:
            # If using Render/Production URL, we usually don't need to create DB as it's provided
            # But if we are running locally with postgres
            print(f"Using DATABASE_URL: {DATABASE_URL}")
            return

        # Local development fallback
        print("Connecting to local Postgres...")
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
            
        cur.close()
        conn.close()

    except Exception as e:
        print(f"Error creating database: {e}")

if __name__ == "__main__":
    create_database()
