"""
🧹 StudyVerse Project Cleanup Script
This script safely archives unused files to keep your project clean.
"""

import os
import shutil
from datetime import datetime

# Create archive folder with timestamp
archive_folder = f"archived_files_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
os.makedirs(archive_folder, exist_ok=True)

print("🧹 Starting cleanup process...")
print(f"📦 Archive folder: {archive_folder}\n")

# Files to archive
files_to_archive = [
    # Test files
    "test_api.py",
    "test_final_gemini.py",
    "test_gemini.py",
    "test_gemini_models.py",
    "test_gemini_simple.py",
    "test_simple_gemini.py",
    
    # Utility scripts (one-time use)
    "add_column.py",
    "check_db.py",
    "fix_password_hash_schema.py",
    "updatexp.py",
    
    # Unused Python modules
    "auth.py",
    "database.py",
    "models.py",
    "schemas.py",
    "ai_service.py",
    "websocket_service.py",
    "pdf_processor.py",
    "app_config.py",
    "list_models.py",
    
    # Firebase/Redundant docs
    "FIREBASE_SETUP.md",
    "README_GEMINI.md",
    "TEST_AI.md",
    "GOOGLE_GEMINI_SETUP.md",
    
    # Config example (optional - keep if you want)
    # "config.example.py",
]

# Folders to archive
folders_to_archive = [
    "inspirationui",  # Design mockups only
]

# Move files
moved_count = 0
for file in files_to_archive:
    if os.path.exists(file):
        try:
            shutil.move(file, os.path.join(archive_folder, file))
            print(f"✅ Archived: {file}")
            moved_count += 1
        except Exception as e:
            print(f"❌ Error moving {file}: {e}")
    else:
        print(f"⚠️  Not found: {file}")

# Move folders
for folder in folders_to_archive:
    if os.path.exists(folder):
        try:
            shutil.move(folder, os.path.join(archive_folder, folder))
            print(f"✅ Archived folder: {folder}")
            moved_count += 1
        except Exception as e:
            print(f"❌ Error moving {folder}: {e}")
    else:
        print(f"⚠️  Not found: {folder}")

print(f"\n🎉 Cleanup complete!")
print(f"📊 {moved_count} items archived to: {archive_folder}")
print(f"\n💡 To restore files: move them back from {archive_folder}/")
print(f"💡 To delete permanently: delete the {archive_folder}/ folder")
