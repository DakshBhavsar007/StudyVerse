# AI API Setup Guide

This guide shows you exactly where to paste your AI API key to make the chat feature work.

## Quick Setup (3 Steps)

### Step 1: Get Your API Key

Choose one AI provider:

1. **OpenAI** (Recommended for beginners)
   - Sign up at https://platform.openai.com
   - Get your API key from https://platform.openai.com/api-keys
   - Model: `gpt-4` or `gpt-3.5-turbo`

2. **Anthropic Claude**
   - Sign up at https://console.anthropic.com
   - Get your API key from https://console.anthropic.com/settings/keys
   - Model: `claude-3-opus-20240229` or `claude-3-sonnet-20240229`

3. **Lovable AI Gateway** (Original project used this)
   - Get your key from Lovable.dev dashboard
   - Model: `google/gemini-2.5-flash`

4. **Google Gemini**
   - Get API key from https://aistudio.google.com/app/apikey
   - Model: `gemini-pro`

### Step 2: Add Your API Key

**Method A: Environment Variable (Recommended for Production)**

On Windows (PowerShell):
```powershell
$env:AI_API_KEY="sk-your-api-key-here"
$env:AI_API_TYPE="openai"
$env:AI_MODEL="gpt-4"
python app.py
```

On Linux/Mac:
```bash
export AI_API_KEY="sk-your-api-key-here"
export AI_API_TYPE="openai"
export AI_MODEL="gpt-4"
python app.py
```

**Method B: Direct in app.py (Easy for Testing)**

Open `app.py` and find these lines (around line 15-20):

```python
# Method 2: Direct configuration (alternative - paste your key here)
# Uncomment and paste your API key below:
# AI_API_KEY = "sk-your-api-key-here"
# AI_API_TYPE = "openai"
# AI_MODEL = "gpt-4"
```

Uncomment and paste your key:
```python
AI_API_KEY = "sk-your-actual-api-key-here"
AI_API_TYPE = "openai"  # Change to "anthropic", "lovable", or "google" if needed
AI_MODEL = "gpt-4"  # Change model name if needed
```

### Step 3: Run the Application

```bash
python app.py
```

Open http://localhost:5000 and test the chat feature!

## File Locations

- **Main file to edit**: `app.py` (lines 15-20 for API key, lines 222-280 for chat endpoint)
- **Chat endpoint function**: `chat_message()` in `app.py` (line 222)
- **AI API function**: `call_ai_api()` in `app.py` (line 251)

## Configuration Options

| Setting | Description | Example Values |
|---------|-------------|----------------|
| `AI_API_KEY` | Your API key from the provider | `sk-...` (OpenAI), `sk-ant-...` (Anthropic) |
| `AI_API_TYPE` | Which AI service to use | `openai`, `anthropic`, `lovable`, `google` |
| `AI_MODEL` | Which model to use | `gpt-4`, `gpt-3.5-turbo`, `claude-3-opus-20240229` |

## Troubleshooting

**Error: "AI API key not configured"**
- Make sure you've set `AI_API_KEY` in environment variable OR uncommented it in `app.py`

**Error: "Rate limits exceeded"**
- You've hit your API usage limit. Wait a bit or upgrade your plan.

**Error: "Payment required"**
- Your API account needs billing setup. Add payment method to your OpenAI/Anthropic account.

**Chat not responding**
- Check your internet connection
- Verify API key is correct
- Check console for error messages

## Notes

- SQLite database is already configured (no setup needed)
- SECRET_KEY is set to a default value (change it for production)
- All API keys are handled securely (never commit them to git!)

