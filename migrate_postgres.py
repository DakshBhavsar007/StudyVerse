<<<<<<< HEAD
from app import app, db
import sqlalchemy

def migrate():
    with app.app_context():
        # Using text() for raw SQL execution in SQLAlchemy
        from sqlalchemy import text
        
        try:
            # Add group_id column
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE todo ADD COLUMN group_id INTEGER DEFAULT NULL REFERENCES \"group\"(id);"))
                conn.commit()
            print("Added group_id column.")
        except Exception as e:
            print(f"Skipping group_id: {e}")

        try:
            # Add completed_by column
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE todo ADD COLUMN completed_by TEXT DEFAULT '';"))
                conn.commit()
            print("Added completed_by column.")
        except Exception as e:
            print(f"Skipping completed_by: {e}")

        print("Migration complete.")

if __name__ == "__main__":
    migrate()
=======
from app import app, db
import sqlalchemy

def migrate():
    with app.app_context():
        # Using text() for raw SQL execution in SQLAlchemy
        from sqlalchemy import text
        
        try:
            # Add group_id column
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE todo ADD COLUMN group_id INTEGER DEFAULT NULL REFERENCES \"group\"(id);"))
                conn.commit()
            print("Added group_id column.")
        except Exception as e:
            print(f"Skipping group_id: {e}")

        try:
            # Add completed_by column
            with db.engine.connect() as conn:
                conn.execute(text("ALTER TABLE todo ADD COLUMN completed_by TEXT DEFAULT '';"))
                conn.commit()
            print("Added completed_by column.")
        except Exception as e:
            print(f"Skipping completed_by: {e}")

        print("Migration complete.")

if __name__ == "__main__":
    migrate()
>>>>>>> f5971551499e078e14bc7548b7a15e1f97eb6644
