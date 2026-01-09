# PostgreSQL Migration Guide

## Overview
This guide will help you migrate from SQLite to PostgreSQL for the StudyVerse application.

## Prerequisites
1. **Install PostgreSQL**: Download and install PostgreSQL from https://www.postgresql.org/download/
2. **Install pgAdmin 4**: Usually comes with PostgreSQL installation
3. **Install Python PostgreSQL driver**: Already added to requirements.txt

---

## Step-by-Step Migration Process

### **Step 1: Install PostgreSQL Driver**

Run this command to install the PostgreSQL adapter:

```bash
pip install psycopg2-binary==2.9.9
```

Or install all requirements:
```bash
pip install -r requirements.txt
```

---

### **Step 2: Set Up PostgreSQL Database**

#### **Option A: Using pgAdmin 4 (GUI)**

1. **Open pgAdmin 4**
2. **Create a new database:**
   - Right-click on "Databases" → "Create" → "Database"
   - Database name: `StudyVerse_db`
   - Owner: `postgres` (or your username)
   - Click "Save"

3. **Note down your credentials:**
   - Host: `localhost`
   - Port: `5432` (default)
   - Database: `StudyVerse_db`
   - Username: `postgres` (or your username)
   - Password: (the password you set during PostgreSQL installation)

#### **Option B: Using Command Line (psql)**

```bash
# Login to PostgreSQL
psql -U postgres

# Create database
CREATE DATABASE StudyVerse_db;

# Create a dedicated user (optional but recommended)
CREATE USER StudyVerse_user WITH PASSWORD 'your_secure_password';

# Grant privileges
GRANT ALL PRIVILEGES ON DATABASE StudyVerse_db TO StudyVerse_user;

# Exit
\q
```

---

### **Step 3: Update .env Configuration**

Replace the SQLite connection string with PostgreSQL:

**Current (SQLite):**
```
DATABASE_URL=sqlite:///StudyVerse.db
```

**New (PostgreSQL):**
```
DATABASE_URL=postgresql://username:password@localhost:5432/StudyVerse_db
```

**Example configurations:**

**If using default postgres user:**
```
DATABASE_URL=postgresql://postgres:your_password@localhost:5432/StudyVerse_db
```

**If you created a dedicated user:**
```
DATABASE_URL=postgresql://StudyVerse_user:your_secure_password@localhost:5432/StudyVerse_db
```

**For production (e.g., hosted PostgreSQL):**
```
DATABASE_URL=postgresql://user:password@your-host.com:5432/database_name
```

---

### **Step 4: Export Data from SQLite (Optional)**

If you want to migrate existing data from SQLite to PostgreSQL:

1. **Create a backup script** (`migrate_data.py`):

```python
import sqlite3
import json
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

# Connect to SQLite
sqlite_conn = sqlite3.connect('StudyVerse.db')
sqlite_cursor = sqlite_conn.cursor()

# Get all table names
sqlite_cursor.execute("SELECT name FROM sqlite_master WHERE type='table';")
tables = sqlite_cursor.fetchall()

# Export data
for table in tables:
    table_name = table[0]
    if table_name != 'sqlite_sequence':
        print(f"Exporting {table_name}...")
        sqlite_cursor.execute(f"SELECT * FROM {table_name}")
        rows = sqlite_cursor.fetchall()
        # Save to JSON file
        with open(f'backup_{table_name}.json', 'w') as f:
            json.dump([dict(row) for row in rows], f, indent=2)

sqlite_conn.close()
print("Export complete!")
```

2. **Run the backup script:**
```bash
python migrate_data.py
```

---

### **Step 5: Initialize PostgreSQL Database**

1. **Stop the running Flask app** (if running)

2. **Delete the old SQLite database** (after backing up if needed):
   - The file `StudyVerse.db` will no longer be used

3. **Run the Flask app** - it will automatically create tables in PostgreSQL:
```bash
python app.py
```

The app will:
- Connect to PostgreSQL
- Create all tables automatically
- Run any migrations defined in the code

---

### **Step 6: Verify the Migration**

1. **Check in pgAdmin 4:**
   - Expand your database → Schemas → public → Tables
   - You should see all your tables (user, todo, friendship, etc.)

2. **Test the application:**
   - Sign up with a new account
   - Create todos, send messages, etc.
   - Verify everything works correctly

---

## Connection String Format Reference

```
postgresql://[user]:[password]@[host]:[port]/[database]
```

**Components:**
- `user`: PostgreSQL username (e.g., `postgres`, `StudyVerse_user`)
- `password`: User's password
- `host`: Database server address (e.g., `localhost`, `192.168.1.100`)
- `port`: PostgreSQL port (default: `5432`)
- `database`: Database name (e.g., `StudyVerse_db`)

**Examples:**

Local development:
```
postgresql://postgres:mypass123@localhost:5432/StudyVerse_db
```

Remote server:
```
postgresql://myuser:securepass@db.example.com:5432/production_db
```

Cloud hosting (Heroku, Render, etc.):
```
postgresql://user:pass@ec2-xxx.compute.amazonaws.com:5432/dbname
```

---

## Troubleshooting

### **Error: "psycopg2" not found**
```bash
pip install psycopg2-binary
```

### **Error: "could not connect to server"**
- Make sure PostgreSQL service is running
- Windows: Check Services → PostgreSQL
- Linux/Mac: `sudo systemctl status postgresql`

### **Error: "password authentication failed"**
- Double-check your username and password
- Verify credentials in pgAdmin 4

### **Error: "database does not exist"**
- Create the database first using pgAdmin or psql

### **Connection timeout**
- Check if PostgreSQL is listening on the correct port
- Verify firewall settings

---

## Advantages of PostgreSQL Over SQLite

✅ **Concurrent Access**: Multiple users can write simultaneously
✅ **Scalability**: Better performance with large datasets
✅ **Advanced Features**: JSON support, full-text search, advanced queries
✅ **Data Integrity**: Better ACID compliance
✅ **Production Ready**: Industry standard for web applications
✅ **Remote Access**: Can connect from multiple servers
✅ **Better Security**: User roles and permissions

---

## Keeping Both Databases (Optional)

You can use SQLite for development and PostgreSQL for production by using environment variables:

**.env.development:**
```
DATABASE_URL=sqlite:///StudyVerse.db
```

**.env.production:**
```
DATABASE_URL=postgresql://user:pass@host:5432/StudyVerse_db
```

Then load the appropriate .env file based on environment.

---

## Next Steps After Migration

1. ✅ Test all features thoroughly
2. ✅ Set up regular database backups
3. ✅ Configure connection pooling for better performance
4. ✅ Monitor database performance
5. ✅ Consider using database migrations tool (Alembic)

---

## Support

If you encounter any issues during migration, common solutions:
- Restart PostgreSQL service
- Verify connection string format
- Check PostgreSQL logs in pgAdmin
- Ensure database user has proper permissions

Good luck with your migration! 🚀
