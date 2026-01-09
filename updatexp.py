from app import app, db, User, GamificationService

def update_xp():
    with app.app_context():
        # Find users starting with Daksh (Case insensitive search if possible, or usually just Capitalized)
        users = User.query.filter(User.first_name.like('Daksh%')).all()
        
        if not users:
            print("No users found starting with 'Daksh'")
            return

        for user in users:
            print(f"Updating user: {user.first_name} {user.last_name} (Current XP: {user.total_xp})")
            user.total_xp += 1000000
            
            # Recalculate level
            new_level = GamificationService.calculate_level(user.total_xp)
            if new_level > user.level:
                print(f"  Level Up! {user.level} -> {new_level}")
                user.level = new_level
                
            print(f"  New XP: {user.total_xp}")
            
        db.session.commit()
        print("Update successful!")

if __name__ == "__main__":
    update_xp()
