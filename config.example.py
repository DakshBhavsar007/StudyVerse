# Configuration Example File
# Copy this file and rename it to config.py, then add your API keys

# ============================================
# AI API CONFIGURATION - PASTE YOUR API KEY HERE
# ============================================

# Option 1: OpenAI
AI_API_KEY = "sk-your-openai-api-key-here"
AI_API_TYPE = "openai"
AI_MODEL = "gpt-4"  # or "gpt-3.5-turbo"

# Option 2: Anthropic Claude
# AI_API_KEY = "sk-ant-your-anthropic-api-key-here"
# AI_API_TYPE = "anthropic"
# AI_MODEL = "claude-3-opus-20240229"

# Option 3: Lovable AI Gateway (from original project)
# AI_API_KEY = "your-lovable-api-key-here"
# AI_API_TYPE = "lovable"
# AI_MODEL = "google/gemini-2.5-flash"

# Option 4: Google Gemini
# AI_API_KEY = "your-google-api-key-here"
# AI_API_TYPE = "google"
# AI_MODEL = "gemini-pro"

# ============================================
# Flask Configuration
# ============================================
SECRET_KEY = "your-secret-key-here-change-this-in-production"

# ============================================
# Database (SQLite - no configuration needed)
# ============================================
# SQLite database will be created automatically as StudyVerse.db

