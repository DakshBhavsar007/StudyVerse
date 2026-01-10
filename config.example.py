# Configuration Example File
# Copy this file and rename it to config.py (or use .env), then add your API keys

# ============================================
# Database Configuration
# ============================================
# Local PostgreSQL
# DATABASE_URL = "postgresql://postgres:password@localhost:5432/StudyVerse_Final"

# ============================================
# AI API CONFIGURATION
# ============================================
# Google Gemini
AI_API_KEY = "your-google-api-key-here"
AI_API_TYPE = "google"
AI_MODEL = "models/gemini-2.5-flash"

# ============================================
# Google OAuth Configuration
# ============================================
GOOGLE_CLIENT_ID = "your-google-client-id"
GOOGLE_CLIENT_SECRET = "your-google-client-secret"

# ============================================
# Email Configuration (SMTP)
# ============================================
MAIL_SERVER = "smtp.gmail.com"
MAIL_PORT = 587
MAIL_USERNAME = "your-email@gmail.com"
MAIL_PASSWORD = "your-app-password"
MAIL_DEFAULT_SENDER = "StudyVerse <your-email@gmail.com>"

# ============================================
# Flask Configuration
# ============================================
SECRET_KEY = "your-secret-key-here-change-this-in-production"

