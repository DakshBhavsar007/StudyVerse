import eventlet
eventlet.monkey_patch()

from flask import Flask, render_template, request, session, redirect, url_for, Response, flash, jsonify
from flask_sqlalchemy import SQLAlchemy
from flask_login import LoginManager, login_user, logout_user, login_required, current_user, UserMixin
from flask_socketio import SocketIO, join_room, emit
from werkzeug.security import generate_password_hash, check_password_hash
from werkzeug.utils import secure_filename
from datetime import datetime, timedelta
from authlib.integrations.flask_client import OAuth
import base64
import io
import json
import os
import requests
from email_service import init_mail, send_welcome_email
from pytz import timezone, utc

import os
import sys
import time
import re
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Allow OAuth over HTTP for local testing
os.environ['OAUTHLIB_INSECURE_TRANSPORT'] = '1'

# Check for required API Key
if not os.getenv("AI_API_KEY"):
    print("\n" + "!" * 80)
    print(" ERROR: AI_API_KEY is missing!")
    print("!" * 80)
    print(" Please create a .env file in the project root directory with the following content:")
    print(" AI_API_KEY=your_api_key_here")
    print(" AI_API_TYPE=google  # or openai, anthropic, lovable")
    print("!" * 80 + "\n")

try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False


from werkzeug.middleware.proxy_fix import ProxyFix

app = Flask(__name__)
# Fix for Render (and other proxies) to handle HTTPS correctly
app.wsgi_app = ProxyFix(app.wsgi_app, x_for=1, x_proto=1, x_host=1, x_prefix=1)

app.config['SECRET_KEY'] = os.getenv('SECRET_KEY', 'dev-secret-key-change-in-production')
app.config['SQLALCHEMY_DATABASE_URI'] = os.getenv('DATABASE_URL', 'sqlite:///StudyVerse.db')
app.config['SQLALCHEMY_TRACK_MODIFICATIONS'] = False
# Optimize connection pooling for Render
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
}
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# Google OAuth Config - Use environment variables
app.config['GOOGLE_CLIENT_ID'] = os.getenv('GOOGLE_CLIENT_ID')
app.config['GOOGLE_CLIENT_SECRET'] = os.getenv('GOOGLE_CLIENT_SECRET')

oauth = OAuth(app)
google = oauth.register(
    name='google',
    client_id=app.config['GOOGLE_CLIENT_ID'],
    client_secret=app.config['GOOGLE_CLIENT_SECRET'],
    server_metadata_url='https://accounts.google.com/.well-known/openid-configuration',
    client_kwargs={'scope': 'openid email profile'}
)

# Session Configuration - Fix persistent login issues
app.config['PERMANENT_SESSION_LIFETIME'] = timedelta(days=7)
# Use secure cookies in production (when https is used)
app.config['SESSION_COOKIE_SECURE'] = os.environ.get('RENDER') is not None
app.config['SESSION_COOKIE_HTTPONLY'] = True
app.config['SESSION_COOKIE_SAMESITE'] = 'Lax'
app.config['REMEMBER_COOKIE_DURATION'] = timedelta(days=30)
app.config['REMEMBER_COOKIE_SECURE'] = os.environ.get('RENDER') is not None
app.config['REMEMBER_COOKIE_HTTPONLY'] = True

os.makedirs(app.config['UPLOAD_FOLDER'], exist_ok=True)

# Initialize SocketIO
socketio = SocketIO(app)

# AI API Configuration
AI_API_KEY = os.getenv("AI_API_KEY", "")
AI_API_TYPE = os.getenv("AI_API_TYPE", "google")

# IST Timezone
IST = timezone('Asia/Kolkata')

# Timezone Helper: Convert UTC datetime to IST formatted time
def to_ist_time(utc_datetime):
    """Convert UTC datetime to IST and return formatted 12-hour time string."""
    if not utc_datetime:
        return ""
    
    # Ensure datetime is timezone-aware (UTC)
    if utc_datetime.tzinfo is None:
        utc_datetime = utc.localize(utc_datetime)
    
    # Convert to IST
    ist_datetime = utc_datetime.astimezone(IST)
    
    # Format as 12-hour time with AM/PM
    return ist_datetime.strftime('%I:%M %p')

AI_MODEL = os.getenv("AI_MODEL", "models/gemini-2.5-flash")


db = SQLAlchemy(app)

# Initialize email service
mail = init_mail(app)

login_manager = LoginManager()
login_manager.init_app(app)
login_manager.login_view = 'auth'

# Jinja Template Filter for IST Time (must be after app is configured)
@app.template_filter('ist_time')
def ist_time_filter(utc_datetime):
    """Jinja filter to convert UTC datetime to IST time string."""
    return to_ist_time(utc_datetime)

# Also add to_ist_time as a global function for templates
app.jinja_env.globals.update(to_ist_time=to_ist_time)



# Database Models
class User(UserMixin, db.Model):
    id = db.Column(db.Integer, primary_key=True)
    email = db.Column(db.String(120), unique=True, nullable=False)
    password_hash = db.Column(db.String(255), nullable=True) # Nullable for OAuth users
    first_name = db.Column(db.String(50))
    last_name = db.Column(db.String(50))
    google_id = db.Column(db.String(100), nullable=True)
    profile_image = db.Column(db.String(255), nullable=True)
    cover_image = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # Gamification Fields
    total_xp = db.Column(db.Integer, default=0)
    level = db.Column(db.Integer, default=1)
    current_streak = db.Column(db.Integer, default=0)
    longest_streak = db.Column(db.Integer, default=0)
    last_activity_date = db.Column(db.Date, nullable=True)
    about_me = db.Column(db.Text, nullable=True)
    
    def get_avatar(self, size=200):
        if self.profile_image and "ui-avatars.com" not in self.profile_image:
            return self.profile_image
        
        # Robust initial extraction
        f_name = (self.first_name or '').strip()
        l_name = (self.last_name or '').strip()
        
        f = f_name[0] if f_name else ''
        l = l_name[0] if l_name else ''
        
        # Force at least one character
        initials = f"{f}{l}".upper()
        if not initials:
            initials = "U"
            
        return f"https://ui-avatars.com/api/?name={initials}&background=0ea5e9&color=fff&size={size}"
    
    def to_dict(self):
        return {
            'id': self.id,
            'first_name': self.first_name,
            'last_name': self.last_name,
            'level': self.level,
            'total_xp': self.total_xp,
            'rank': GamificationService.get_rank(self.level),
            'avatar': self.get_avatar(),
            'is_public': self.is_public_profile
        }

    # Privacy & Status
    is_public_profile = db.Column(db.Boolean, default=True)
    last_seen = db.Column(db.DateTime, default=datetime.utcnow)

class Friendship(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    friend_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    status = db.Column(db.String(20), default='pending') # pending, accepted, rejected
    created_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    # We can use backrefs in User if preferred, or just query directly
    # user = db.relationship('User', foreign_keys=[user_id], backref=db.backref('friendships_sent', lazy='dynamic'))
    # friend = db.relationship('User', foreign_keys=[friend_id], backref=db.backref('friendships_received', lazy='dynamic'))

class Badge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    description = db.Column(db.String(255), nullable=False)
    icon = db.Column(db.String(50), default='fa-medal') # FontAwesome class
    criteria_type = db.Column(db.String(50)) # e.g., 'streak', 'xp', 'wins'
    criteria_value = db.Column(db.Integer)

class UserBadge(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    badge_id = db.Column(db.Integer, db.ForeignKey('badge.id'), nullable=False)
    earned_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    user = db.relationship('User', backref='badges')
    badge = db.relationship('Badge')

class XPHistory(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    source = db.Column(db.String(50), nullable=False) # battle, task, focus
    amount = db.Column(db.Integer, nullable=False)
    timestamp = db.Column(db.DateTime, default=datetime.utcnow)

class GamificationService:
    """Central logic for XP, Levels, Ranks, and Badges."""
    
    RANKS = {
        (1, 5): ('Bronze', 'fa-shield-halved', '#CD7F32'),
        (6, 10): ('Silver', 'fa-shield-halved', '#C0C0C0'),
        (11, 20): ('Gold', 'fa-shield-halved', '#FFD700'),
        (21, 35): ('Platinum', 'fa-gem', '#E5E4E2'),
        (36, 50): ('Diamond', 'fa-gem', '#b9f2ff'),
        (51, 75): ('Heroic', 'fa-crown', '#ff4d4d'),
        (76, 100): ('Master', 'fa-crown', '#ff0000'),
        (101, 9999): ('Grandmaster', 'fa-dragon', '#800080')
    }

    @staticmethod
    def calculate_level(total_xp):
        # Level = floor(total_xp / 500) + 1
        return max(1, int(total_xp / 500) + 1)

    @staticmethod
    def get_rank(level):
        for (min_lvl, max_lvl), (name, icon, color) in GamificationService.RANKS.items():
            if min_lvl <= level <= max_lvl:
                return {'name': name, 'icon': icon, 'color': color}
        return {'name': 'Bronze', 'icon': 'fa-shield-halved', 'color': '#CD7F32'}

    @staticmethod
    def add_xp(user_id, source, amount):
        user = User.query.get(user_id)
        if not user:
            return

        # Cap Focus XP daily
        if source == 'focus':
            today_start = datetime.utcnow().replace(hour=0, minute=0, second=0, microsecond=0)
            daily_focus_xp = db.session.query(db.func.sum(XPHistory.amount))\
                .filter(XPHistory.user_id == user.id, XPHistory.source == 'focus', XPHistory.timestamp >= today_start)\
                .scalar() or 0
            
            # Simple daily cap logic: max 500 XP from focus per day
            if daily_focus_xp >= 500:
                return {'earned': 0, 'message': 'Daily Focus XP cap reached!'}
            
            if daily_focus_xp + amount > 500:
                amount = 500 - daily_focus_xp

        if amount <= 0:
            return

        user.total_xp += amount
        
        # Level Up Check
        new_level = GamificationService.calculate_level(user.total_xp)
        leveled_up = False
        if new_level > user.level:
            user.level = new_level
            leveled_up = True
            
        # Log History
        log = XPHistory(user_id=user.id, source=source, amount=amount)
        db.session.add(log)
        
        # Update Streak (if not already updated today)
        GamificationService.update_streak(user)
        
        # Check Badges
        GamificationService.check_badges(user)
        
        db.session.commit()
        
        return {
            'earned': amount, 
            'new_total': user.total_xp, 
            'leveled_up': leveled_up,
            'new_level': user.level,
            'rank': GamificationService.get_rank(user.level)
        }

    @staticmethod
    def update_streak(user):
        today = datetime.utcnow().date()
        if user.last_activity_date == today:
            return # Already active today
        
        if user.last_activity_date == today - timedelta(days=1):
            user.current_streak += 1
        else:
            user.current_streak = 1 # Reset if missed a day (or first time)
            
        user.last_activity_date = today
        if user.current_streak > user.longest_streak:
            user.longest_streak = user.current_streak

    @staticmethod
    def check_badges(user):
        # 1. Streak Badges
        if user.current_streak >= 30:
            GamificationService.award_badge(user, 'Consistency King')
        
        # 2. XP Badges (Level based roughly)
        if user.level >= 10:
             GamificationService.award_badge(user, 'Rising Star')
        if user.level >= 50:
             GamificationService.award_badge(user, 'Dedicated Scholar')
        if user.level >= 100:
             GamificationService.award_badge(user, 'Centurion')
             
        # More rules can be added here
        
    @staticmethod
    def award_badge(user, badge_name):
        badge = Badge.query.filter_by(name=badge_name).first()
        if not badge:
            # Create default if missing (lazy init)
            if badge_name == 'Consistency King':
                badge = Badge(name='Consistency King', description='Achieve a 30-day streak', icon='fa-fire', criteria_type='streak', criteria_value=30)
            elif badge_name == 'Rising Star':
                badge = Badge(name='Rising Star', description='Reach Level 10', icon='fa-star', criteria_type='level', criteria_value=10)
            elif badge_name == 'Dedicated Scholar':
                badge = Badge(name='Dedicated Scholar', description='Reach Level 50', icon='fa-book-open', criteria_type='level', criteria_value=50)
            elif badge_name == 'Centurion':
                badge = Badge(name='Centurion', description='Reach Level 100', icon='fa-crown', criteria_type='level', criteria_value=100)
            else:
                return 
            db.session.add(badge)
            db.session.commit()
            
        if not UserBadge.query.filter_by(user_id=user.id, badge_id=badge.id).first():
            ub = UserBadge(user_id=user.id, badge_id=badge.id)
            db.session.add(ub)



class Todo(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    title = db.Column(db.String(200), nullable=False)
    completed = db.Column(db.Boolean, default=False)
    priority = db.Column(db.String(20), default='medium')
    due_date = db.Column(db.String(50))
    category = db.Column(db.String(50))
    is_group = db.Column(db.Boolean, default=False)
    group_id = db.Column(db.Integer, db.ForeignKey('group.id'), nullable=True)
    completed_by = db.Column(db.Text, default="")  # Comma-separated user_ids
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class ChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    is_group = db.Column(db.Boolean, default=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class Subject(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    name = db.Column(db.String(100), nullable=False)
    color = db.Column(db.String(20), default='#4ade80')
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class StudySession(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    subject_id = db.Column(db.Integer, db.ForeignKey('subject.id'), nullable=True)
    duration = db.Column(db.Integer, nullable=False)  # in minutes
    mode = db.Column(db.String(20), default='focus')  # focus, shortBreak, longBreak
    completed_at = db.Column(db.DateTime, default=datetime.utcnow)
    
    subject = db.relationship('Subject')

class TopicProficiency(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    topic_name = db.Column(db.String(200), nullable=False)
    proficiency = db.Column(db.Integer, default=0)
    completed = db.Column(db.Boolean, default=False)
    updated_at = db.Column(db.DateTime, default=datetime.utcnow)

class Group(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    name = db.Column(db.String(100), nullable=False)
    admin_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    invite_code = db.Column(db.String(10), nullable=False, unique=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

class GroupMember(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.Integer, db.ForeignKey('group.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    joined_at = db.Column(db.DateTime, default=datetime.utcnow)

    __table_args__ = (
        db.UniqueConstraint('group_id', 'user_id', name='uq_group_member'),
    )

class GroupChatMessage(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    group_id = db.Column(db.Integer, db.ForeignKey('group.id'), nullable=False)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=True)
    role = db.Column(db.String(20), nullable=False)  # 'user' or 'assistant'
    content = db.Column(db.Text, nullable=False)
    file_path = db.Column(db.String(255), nullable=True)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)

    user = db.relationship('User', backref='group_messages')

class SyllabusDocument(db.Model):
    id = db.Column(db.Integer, primary_key=True)
    user_id = db.Column(db.Integer, db.ForeignKey('user.id'), nullable=False)
    filename = db.Column(db.String(100), nullable=False)
    extracted_text = db.Column(db.Text, nullable=False)
    created_at = db.Column(db.DateTime, default=datetime.utcnow)


# ------------------------------
# Data Structures (DS) Utilities
# ------------------------------

class Stack:
    """Simple LIFO stack.

    Used for: Undo delete in Todos.
    """

    def __init__(self):
        self._items = []

    def push(self, item):
        self._items.append(item)

    def pop(self):
        if not self._items:
            return None
        return self._items.pop()

    def is_empty(self):
        return len(self._items) == 0


class LRUCache:
    """LRU Cache using dict + list (simplified).

    DS concept:
    - Hash map (dict) for O(1) key lookup
    - List to track usage order for eviction
    """

    def __init__(self, capacity=50):
        self.capacity = capacity
        self.cache = {}
        self.order = []  # most recent at end

    def get(self, key):
        if key not in self.cache:
            return None
        if key in self.order:
            self.order.remove(key)
        self.order.append(key)
        return self.cache[key]

    def put(self, key, value):
        if key in self.cache:
            self.cache[key] = value
            if key in self.order:
                self.order.remove(key)
            self.order.append(key)
            return

        if len(self.cache) >= self.capacity:
            oldest = self.order.pop(0)
            del self.cache[oldest]

        self.cache[key] = value
        self.order.append(key)


# ------------------------------
# OOP Services
# ------------------------------

class AuthService:
    """Authentication service (OOP abstraction around auth logic)."""

    @staticmethod
    def create_user(email: str, password: str, first_name: str, last_name: str) -> "User":
        if User.query.filter_by(email=email).first():
            raise ValueError("Email already registered")

        user = User(
            email=email,
            password_hash=generate_password_hash(password),
            first_name=first_name,
            last_name=last_name,
        )
        db.session.add(user)
        db.session.commit()
        return user

    @staticmethod
    def authenticate(email: str, password: str):
        user = User.query.filter_by(email=email).first()
        
        # Check if user exists
        if not user:
            return None
        
        # Check if user has a password (not a Google OAuth user)
        if not user.password_hash:
            # User signed up with Google OAuth, no password set
            return None
        
        # Verify password
        if check_password_hash(user.password_hash, password):
            return user
        
        return None


class GroupService:
    """Group operations: create, join, membership check."""

    @staticmethod
    def _generate_invite_code(length: int = 6) -> str:
        import random
        import string

        alphabet = string.ascii_uppercase + string.digits
        return ''.join(random.choice(alphabet) for _ in range(length))

    @staticmethod
    def create_group(admin_user_id: int, name: str) -> Group:
        invite_code = GroupService._generate_invite_code()
        while Group.query.filter_by(invite_code=invite_code).first() is not None:
            invite_code = GroupService._generate_invite_code()

        group = Group(name=name, admin_id=admin_user_id, invite_code=invite_code)
        db.session.add(group)
        db.session.commit()

        db.session.add(GroupMember(group_id=group.id, user_id=admin_user_id))
        db.session.commit()
        return group

    @staticmethod
    def join_group(user_id: int, invite_code: str) -> Group:
        group = Group.query.filter_by(invite_code=invite_code).first()
        if not group:
            raise ValueError("Invalid invite code")

        existing = GroupMember.query.filter_by(group_id=group.id, user_id=user_id).first()
        if existing:
            return group

        db.session.add(GroupMember(group_id=group.id, user_id=user_id))
        db.session.commit()
        return group

    @staticmethod
    def get_user_group(user_id: int):
        membership = GroupMember.query.filter_by(user_id=user_id).order_by(GroupMember.joined_at.desc()).first()
        if not membership:
            return None
        return Group.query.get(membership.group_id)


class SyllabusService:
    """PDF syllabus upload + extraction and retrieval (simple)."""

    @staticmethod
    def save_syllabus(user_id: int, filename: str, extracted_text: str) -> SyllabusDocument:
        existing = SyllabusDocument.query.filter_by(user_id=user_id).first()
        if existing:
            db.session.delete(existing)
            db.session.commit()

        doc = SyllabusDocument(user_id=user_id, filename=filename, extracted_text=extracted_text)
        db.session.add(doc)
        db.session.commit()
        return doc

    @staticmethod
    def get_syllabus_text(user_id: int) -> str:
        doc = SyllabusDocument.query.filter_by(user_id=user_id).first()
        return doc.extracted_text if doc else ""

    @staticmethod
    def extract_tasks_from_pdf(pdf_bytes: bytes) -> list:
        if not AI_API_KEY:
            raise ValueError("AI_API_KEY not configured")

        model_id = os.environ.get("GEMINI_PDF_MODEL", "models/gemini-2.5-flash")
        if "/" in model_id:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/{model_id}:generateContent?key={AI_API_KEY}"
        else:
            endpoint = f"https://generativelanguage.googleapis.com/v1beta/models/{model_id}:generateContent?key={AI_API_KEY}"

        prompt = (
            "You are a study assistant. Analyze the attached PDF notes. "
            "Identify the main chapters and key topics. "
            "Output ONLY valid JSON in this exact format: "
            "{\"tasks\": [{\"title\": \"Chapter Name\", \"subtasks\": [\"Topic 1\", \"Topic 2\"]}]}"
        )

        pdf_data = base64.b64encode(pdf_bytes).decode("utf-8")
        payload = {
            "contents": [
                {
                    "parts": [
                        {"text": prompt},
                        {
                            "inline_data": {
                                "mime_type": "application/pdf",
                                "data": pdf_data,
                            }
                        },
                    ]
                }
            ],
            "generationConfig": {
                "response_mime_type": "application/json",
                "temperature": 0.2,
            },
        }

        response = requests.post(endpoint, json=payload, headers={'Content-Type': 'application/json'}, timeout=60)
        result_data = response.json() if response.text else {}

        if response.status_code != 200:
            error_msg = result_data.get("error", {}).get("message", response.text or "Unknown error")
            raise ValueError(f"Google Error: {error_msg}")

        if "error" in result_data:
            error_msg = result_data["error"].get("message", "Unknown API Error")
            raise ValueError(f"Google Error: {error_msg}")

        raw = ""
        if "candidates" in result_data and result_data["candidates"]:
            raw = result_data["candidates"][0]["content"]["parts"][0].get("text", "")
        if not raw:
            raise ValueError("AI could not read this PDF")

        raw = raw.strip()
        if raw.startswith("```"):
            raw = raw.strip("`")
            if raw.lower().startswith("json"):
                raw = raw[4:].lstrip()

        try:
            parsed = json.loads(raw)
        except Exception:
            start = raw.find("{")
            end = raw.rfind("}")
            if start == -1 or end == -1 or end <= start:
                raise
            parsed = json.loads(raw[start : end + 1])

        tasks = parsed.get("tasks", [])
        if not isinstance(tasks, list):
            raise ValueError("Invalid AI response: tasks is not a list")
        return tasks

    @staticmethod
    def build_chapters_from_todos(user_id: int) -> list:
        todos = (
            Todo.query
            .filter_by(user_id=user_id, is_group=False)
            .order_by(Todo.created_at.asc())
            .all()
        )

        chapters = {}
        for t in todos:
            if not t.category:
                continue
            chapters.setdefault(t.category, []).append(t)

        result = []
        for category, items in chapters.items():
            total = len(items)
            completed = sum(1 for x in items if x.completed)
            percent = int((completed / total) * 100) if total else 0
            result.append({
                'name': category,
                'todos': items,
                'total': total,
                'completed': completed,
                'percent': percent,
            })

        result.sort(key=lambda x: x['name'].lower())
        return result


class ChatService:
    """Personal + group AI chat.

    Uses DS concept:
    - LRU cache to avoid repeated calls for same query.
    """

    _cache = LRUCache(capacity=100)

    @staticmethod
    def build_system_prompt(user: User, syllabus_text: str) -> str:
        base = (
            "You are StudyVerse, an expert AI Study Coach and academic mentor. "
            "Your goal is to help students learn effectively, stay motivated, and organize their studies.\n"
            "Guidelines:\n"
            "1. Be encouraging, structured, and clear. Use Markdown (bold, lists) for readability.\n"
            "2. If the user asks about their syllabus, refer to the provided syllabus text.\n"
            "3. Keep responses concise but helpful. Avoid long monologues unless necessary.\n"
            "4. Remember the context of the conversation."
        )
        if syllabus_text:
            base += "\n\nSyllabus Context (extracted from uploaded PDF):\n" + syllabus_text[:3000]
        return base

    @staticmethod
    def personal_reply(user: User, message: str) -> str:
        # Legacy wrapper
        return ChatService.generate_chat_response(user, message)

    @staticmethod
    def generate_chat_response(user: User, message: str) -> str:
        # 1. Get History (Last 10 messages)
        history = ChatMessage.query.filter_by(user_id=user.id, is_group=False)\
            .order_by(ChatMessage.created_at.desc())\
            .limit(10)\
            .all()
        history.reverse() # Oldest first

        # 2. Build Messages List for API
        syllabus_text = SyllabusService.get_syllabus_text(user.id)
        system_prompt = ChatService.build_system_prompt(user, syllabus_text)
        
        api_messages = [{'role': 'system', 'content': system_prompt}]
        
        for msg in history:
            role = 'assistant' if msg.role == 'assistant' else 'user'
            api_messages.append({'role': role, 'content': msg.content})
            
        # Add current message
        api_messages.append({'role': 'user', 'content': message})

        # 3. Call API
        reply = call_ai_api(api_messages)
        return reply

@login_manager.user_loader
def load_user(user_id):
    return User.query.get(int(user_id))

# Routes
@app.route('/')
def index():
    try:
        if current_user.is_authenticated:
            return redirect(url_for('dashboard'))
    except:
        pass
    return redirect(url_for('auth'))

@app.route('/auth')
def auth():
    # Check if user is authenticated
    if current_user.is_authenticated:
        print(f"Auth Check: User {current_user.id} is authenticated. Redirecting to dashboard.")
        return redirect(url_for('dashboard'))
    return render_template('auth.html')

@app.route('/signup', methods=['POST'])
def signup():
    """Create account using standard HTML form POST (no JSON)."""
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')
    first_name = request.form.get('first_name', '').strip()
    last_name = request.form.get('last_name', '').strip()

    if not email or not password:
        flash('Email and password are required.', 'error')
        return redirect(url_for('auth'))

    try:
        user = AuthService.create_user(email, password, first_name, last_name)
        
        # Send welcome email to new user
        try:
            send_welcome_email(user.email, user.first_name, user.last_name)
        except Exception as e:
            print(f"Failed to send welcome email: {e}")
            # Continue even if email fails
            
    except ValueError as e:
        flash(str(e), 'error')
        return redirect(url_for('auth'))

    login_user(user, remember=True)  # Enable remember me for persistent sessions
    session.permanent = True
    return redirect(url_for('dashboard'))

@app.route('/signin', methods=['POST'])
def signin():
    """Sign in using standard HTML form POST (no JSON)."""
    email = request.form.get('email', '').strip().lower()
    password = request.form.get('password', '')

    # Check if user exists first
    user = User.query.filter_by(email=email).first()
    
    if user and not user.password_hash:
        # User signed up with Google OAuth
        flash('This account was created with Google Sign-In. Please use the "Sign in with Google" button.', 'error')
        return redirect(url_for('auth'))
    
    # Authenticate with password
    user = AuthService.authenticate(email, password)
    if not user:
        flash('Invalid email or password.', 'error')
        return redirect(url_for('auth'))

    login_user(user, remember=True)  # Enable remember me for persistent sessions
    session.permanent = True
    return redirect(url_for('dashboard'))

@app.route('/logout', methods=['GET', 'POST'])
@login_required
def logout():
    print(f"Logging out user: {current_user.id}")
    # Set last_seen to past to ensure immediate offline status
    try:
        current_user.last_seen = datetime.utcnow() - timedelta(minutes=15)
        db.session.commit()
    except:
        pass
        
    logout_user()
    session.clear()
    
    # Create response to clear cookies explicitly
    response = redirect(url_for('auth'))
    
    # Clear Flask-Login 'remember me' cookie
    cookie_name = app.config.get('REMEMBER_COOKIE_NAME', 'remember_token')
    response.delete_cookie(cookie_name)
    
    # Clear session cookie
    response.delete_cookie('session')
    
    flash('You have been logged out.', 'success')
    return response

@app.route('/login/google')
def login_google():
    redirect_uri = url_for('google_callback', _external=True)
    print(f"[GOOGLE AUTH] Initiating OAuth flow")
    print(f"[GOOGLE AUTH] Redirect URI: {redirect_uri}")
    return google.authorize_redirect(redirect_uri)

@app.route('/login/google/callback')
def google_callback():
    print(f"[GOOGLE AUTH] Callback received")
    print(f"[GOOGLE AUTH] Request args: {request.args}")
    
    # Handle error or missing code
    if 'error' in request.args:
        flash(f"Google login failed: {request.args.get('error')}", 'error')
        return redirect(url_for('auth'))
        
    if 'code' not in request.args:
        flash("Google login failed: No authorization code received.", 'error')
        return redirect(url_for('auth'))

    try:
        print(f"[GOOGLE AUTH] Attempting to authorize access token...")
        token = google.authorize_access_token()
        print(f"[GOOGLE AUTH] Token received successfully")
        user_info = google.parse_id_token(token, nonce=None)
        
        email = user_info.get('email')
        google_id = user_info.get('sub')
        name = user_info.get('name', '')
        picture = user_info.get('picture')

        # Check if user exists
        user = User.query.filter_by(email=email).first()
        is_new_user = False

        if not user:
            is_new_user = True
            # Create new user
            names = name.split(' ', 1) if name else ['', '']
            first_name = names[0]
            last_name = names[1] if len(names) > 1 else ''
            
            user = User(
                email=email,
                first_name=first_name,
                last_name=last_name,
                google_id=google_id,
                profile_image=picture,
                password_hash=None # No password for Google users
            )
            db.session.add(user)
            db.session.commit()
            
            # Send welcome email to new Google user
            try:
                send_welcome_email(user.email, user.first_name, user.last_name)
            except Exception as e:
                print(f"Failed to send welcome email: {e}")
                # Continue even if email fails
        else:
            # Update existing user info
            if not user.google_id:
                user.google_id = google_id
            if picture:
                user.profile_image = picture
            db.session.commit()

        # Log the user in
        login_user(user, remember=True)
        session.permanent = True
        
        return redirect(url_for('dashboard'))
    
    except Exception as e:
        import traceback
        error_details = traceback.format_exc()
        print(f"=" * 80)
        print(f"ERROR during Google Auth:")
        print(f"Exception Type: {type(e).__name__}")
        print(f"Exception Message: {str(e)}")
        print(f"Full Traceback:")
        print(error_details)
        print(f"=" * 80)
        flash(f"Google authentication failed: {str(e)}", "error")
        return redirect(url_for('auth'))

@app.route('/api/auth/google', methods=['POST'])
def google_auth():
    """Handle Google OAuth sign-in from Firebase."""
    import uuid
    
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400
    
    email = data.get('email')
    display_name = data.get('displayName', '')
    
    if not email:
        return jsonify({'status': 'error', 'message': 'Email is required'}), 400
    
    # Check if user exists
    user = User.query.filter_by(email=email).first()
    is_new_user = False
    
    if not user:
        is_new_user = True
        # Create new user from Google account
        names = display_name.split(' ', 1) if display_name else ['', '']
        first_name = names[0] if names else ''
        last_name = names[1] if len(names) > 1 else ''
        
        user = User(
            email=email,
            password_hash=generate_password_hash(str(uuid.uuid4())),  # Random password for OAuth users
            first_name=first_name,
            last_name=last_name
        )
        db.session.add(user)
        db.session.commit()
        
        # Send welcome email to new Google user
        try:
            send_welcome_email(user.email, user.first_name, user.last_name)
        except Exception as e:
            print(f"Failed to send welcome email: {e}")
            # Continue even if email fails
    
    # Log in the user
    login_user(user, remember=True)
    session.permanent = True
    
    return jsonify({'status': 'success', 'message': 'Authentication successful'})


@app.route('/dashboard')
@login_required
def dashboard():
    total_todos = Todo.query.filter_by(user_id=current_user.id).count()
    completed_todos = Todo.query.filter_by(user_id=current_user.id, completed=True).count()
    remaining_todos = max(total_todos - completed_todos, 0)

    week_ago = datetime.utcnow() - timedelta(days=7)
    weekly_minutes = (
        db.session.query(db.func.coalesce(db.func.sum(StudySession.duration), 0))
        .filter(StudySession.user_id == current_user.id)
        .filter(StudySession.completed_at >= week_ago)
        .scalar()
    )
    weekly_hours = round((weekly_minutes or 0) / 60.0, 1)

    completion_percent = int((completed_todos / total_todos) * 100) if total_todos else 0

    topic_rows = (
        TopicProficiency.query
        .filter_by(user_id=current_user.id)
        .order_by(TopicProficiency.updated_at.desc())
        .limit(6)
        .all()
    )
    avg_proficiency = (
        db.session.query(db.func.coalesce(db.func.avg(TopicProficiency.proficiency), 0))
        .filter(TopicProficiency.user_id == current_user.id)
        .scalar()
    )
    avg_proficiency = int(round(avg_proficiency or 0))
    topics_covered = TopicProficiency.query.filter_by(user_id=current_user.id).count()

    recent_todos = (
        Todo.query
        .filter_by(user_id=current_user.id)
        .order_by(Todo.created_at.desc())
        .limit(5)
        .all()
    )
    upcoming_todos = (
        Todo.query
        .filter_by(user_id=current_user.id, completed=False)
        .filter(Todo.due_date.isnot(None))
        .filter(Todo.due_date != '')
        .order_by(Todo.due_date.asc())
        .limit(5)
        .all()
    )

    return render_template(
        'dashboard.html',
        total_todos=total_todos,
        completed_todos=completed_todos,
        remaining_todos=remaining_todos,
        weekly_hours=weekly_hours,
        completion_percent=completion_percent,
        avg_proficiency=avg_proficiency,
        topics_covered=topics_covered,
        topic_rows=topic_rows,
        recent_todos=recent_todos,
        upcoming_todos=upcoming_todos,
    )

@app.route('/chat')
@login_required
def chat():
    messages = ChatMessage.query.filter_by(user_id=current_user.id, is_group=False).order_by(ChatMessage.created_at.asc()).limit(50).all()
    return render_template('chat.html', chat_messages=messages)

@app.route('/chat/send', methods=['POST'])
@login_required
def chat_send():
    """Personal chat send (AJAX supported)."""
    
    # Handle JSON (AJAX)
    if request.is_json:
        data = request.get_json()
        content = data.get('message', '').strip()
        if not content:
            return jsonify({'status': 'error', 'message': 'Empty message'}), 400
            
        # Store user message
        user_msg = ChatMessage(user_id=current_user.id, role='user', content=content, is_group=False)
        db.session.add(user_msg)
        db.session.commit()

        # Generate AI response (Context Aware)
        reply = ChatService.generate_chat_response(current_user, content)
        
        # Store AI response
        ai_msg = ChatMessage(user_id=current_user.id, role='assistant', content=reply, is_group=False)
        db.session.add(ai_msg)
        db.session.commit()
        
        # Return response with IST timestamps
        return jsonify({
            'status': 'success',
            'reply': reply,
            'user_timestamp': to_ist_time(user_msg.created_at),
            'ai_timestamp': to_ist_time(ai_msg.created_at)
        })


    # Legacy Form Post
    content = request.form.get('message', '').strip()
    if not content:
        return redirect(url_for('chat'))

    # Store user message
    db.session.add(ChatMessage(user_id=current_user.id, role='user', content=content, is_group=False))
    db.session.commit()

    # Generate AI response
    reply = ChatService.personal_reply(current_user, content)
    db.session.add(ChatMessage(user_id=current_user.id, role='assistant', content=reply, is_group=False))
    db.session.commit()

    return redirect(url_for('chat'))

@app.route('/group')
@login_required
def group_chat():
    group = GroupService.get_user_group(current_user.id)
    messages = []
    members = []
    online_count = 0
    if group:
        # Load messages and join with User to get names
        messages = (
            GroupChatMessage.query
            .filter_by(group_id=group.id)
            .order_by(GroupChatMessage.created_at.asc())
            .limit(100)
            .all()
        )
        # Join (DBMS concept): membership table join with user table
        members = (
            db.session.query(User)
            .join(GroupMember, GroupMember.user_id == User.id)
            .filter(GroupMember.group_id == group.id)
            .all()
        )
        
        # Attach online status (Active within last 5 minutes)
        now = datetime.utcnow()

        for m in members:
            # If last_seen is None, assume offline.
            # 5 minutes threshold
            if m.last_seen and (now - m.last_seen).total_seconds() < 300:
                m.is_online_status = True
                online_count += 1
            else:
                m.is_online_status = False

    return render_template('group_chat.html', group=group, group_messages=messages, group_members=members, online_count=online_count)

@app.route('/group/create', methods=['POST'])
@login_required
def group_create():
    name = request.form.get('name', '').strip()
    if not name:
        flash('Group name is required.', 'error')
        return redirect(url_for('group_chat'))

    GroupService.create_group(current_user.id, name)
    return redirect(url_for('group_chat'))

@app.route('/group/join', methods=['POST'])
@login_required
def group_join():
    invite_code = request.form.get('invite_code', '').strip().upper()
    if not invite_code:
        flash('Invite code is required.', 'error')
        return redirect(url_for('group_chat'))

    try:
        GroupService.join_group(current_user.id, invite_code)
    except ValueError as e:
        flash(str(e), 'error')
    return redirect(url_for('group_chat'))

@app.route('/group/leave', methods=['POST'])
@login_required
def group_leave():
    """Leave the current group."""
    membership = GroupMember.query.filter_by(user_id=current_user.id).first()
    if membership:
        db.session.delete(membership)
        db.session.commit()
        flash('You have left the group.', 'success')
    return redirect(url_for('group_chat'))

@app.route('/group/send', methods=['POST'])
@login_required
def group_send():
    group = GroupService.get_user_group(current_user.id)
    if not group:
        flash('Join or create a group first.', 'error')
        return redirect(url_for('group_chat'))

    content = request.form.get('message', '').strip()
    if not content:
        return redirect(url_for('group_chat'))

    db.session.add(GroupChatMessage(group_id=group.id, user_id=current_user.id, role='user', content=content))
    db.session.commit()

    # Group AI: trigger if user mentions @StudyVerse
    if '@StudyVerse' in content.lower() or '@assistant' in content.lower():
        reply = ChatService.personal_reply(current_user, content)
        db.session.add(GroupChatMessage(group_id=group.id, user_id=None, role='assistant', content=reply))
        db.session.commit()

    return redirect(url_for('group_chat'))

@app.route('/todos')
@login_required
def todos():
    personal = Todo.query.filter_by(user_id=current_user.id, is_group=False).order_by(Todo.created_at.desc()).all()
    
    group = None
    group_tod = []
    user_group = GroupService.get_user_group(current_user.id)
    if user_group:
        group_tod = Todo.query.filter_by(group_id=user_group.id, is_group=True).order_by(Todo.created_at.desc()).all()
        group = user_group

    return render_template('todos.html', personal_todos=personal, group_todos=group_tod, current_group=group)

@app.route('/todos/add', methods=['POST'])
@login_required
def todos_add():
    title = request.form.get('title', '').strip()
    is_group = request.form.get('is_group') == '1'
    if not title:
        return redirect(url_for('todos'))

    group_id = None
    if is_group:
        user_group = GroupService.get_user_group(current_user.id)
        if user_group:
            group_id = user_group.id
        else:
            # Fallback if user tries to add group task without being in a group
            is_group = False 

    todo = Todo(
        user_id=current_user.id,
        title=title,
        completed=False,
        priority=request.form.get('priority', 'medium'),
        due_date=request.form.get('due_date'),
        category=request.form.get('category'),
        is_group=is_group,
        group_id=group_id
    )
    db.session.add(todo)
    db.session.commit()
    
    next_url = request.form.get('next') or request.args.get('next')
    if next_url:
        return redirect(next_url)
    return redirect(url_for('todos'))

@app.route('/todos/edit/<int:todo_id>', methods=['POST'])
@login_required
def todos_edit(todo_id):
    todo = Todo.query.get_or_404(todo_id)
    if todo.user_id != current_user.id:
        return jsonify({'error': 'Unauthorized'}), 403
    
    todo.title = request.form.get('title', todo.title)
    todo.category = request.form.get('category', todo.category)
    todo.priority = request.form.get('priority', todo.priority)
    todo.due_date = request.form.get('due_date', todo.due_date)
    
    db.session.commit()
    
    next_url = request.form.get('next') or request.referrer
    if next_url:
        return redirect(next_url)
    next_url = request.form.get('next') or request.referrer
    if next_url:
        return redirect(next_url)
    return redirect(url_for('calendar'))

@app.route('/todos/delete_category', methods=['POST'])
@login_required
def todos_delete_category():
    category = request.form.get('category')
    is_group = request.form.get('is_group') == '1'
    
    if not category:
        return redirect(url_for('todos'))

    # delete() with synchronization is safer/standard for bulk deletes
    Todo.query.filter_by(
        user_id=current_user.id, 
        category=category, 
        is_group=is_group
    ).delete()
    
    db.session.commit()
    return redirect(url_for('todos'))

@app.route('/todos/add_batch', methods=['POST'])
@login_required
def todos_add_batch():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400

    category = data.get('category', '').strip()
    priority = data.get('priority', 'Medium')
    due_date = data.get('due_date')
    is_group = data.get('is_group') == '1' or data.get('is_group') is True
    subtasks = data.get('subtasks', [])

    if not subtasks:
        return jsonify({'error': 'No subtasks provided'}), 400

    created_count = 0
    for sub_title in subtasks:
        sub_title = str(sub_title).strip()
        if not sub_title:
            continue
        
        todo = Todo(
            user_id=current_user.id,
            title=sub_title,
            completed=False,
            priority=priority,
            due_date=due_date,
            category=category, # The "Task Title" acts as the category/project name
            is_group=is_group,
        )
        db.session.add(todo)
        created_count += 1

    if created_count > 0:
        db.session.commit()
        return jsonify({'status': 'success', 'count': created_count})
    
    return jsonify({'status': 'no_tasks_created'})

@app.context_processor
def inject_gamification():
    if current_user.is_authenticated:
        rank_info = GamificationService.get_rank(current_user.level)
        # XP to next level = 500 * level (simplified based on formula floor(total/500))
        # actually formula is level = floor(total/500) + 1
        # so next level at: level * 500
        next_level_xp = current_user.level * 500
        progress_percent = int(((current_user.total_xp % 500) / 500) * 100)
        
        return dict(
            rank_name=rank_info['name'],
            rank_icon=rank_info['icon'],
            rank_color=rank_info['color'],
            next_level_xp=next_level_xp,
            level_progress=progress_percent
        )
    return dict()

@app.route('/todos/toggle/<int:todo_id>', methods=['POST'])
@login_required
def todos_toggle(todo_id):
    # Fetch task (check both ownership or group membership)
    todo = Todo.query.get_or_404(todo_id)
    
    # Verify access
    has_access = False
    if todo.user_id == current_user.id:
        has_access = True
    elif todo.is_group and todo.group_id:
        # Check if user is in the group
        user_group = GroupService.get_user_group(current_user.id)
        if user_group and user_group.id == todo.group_id:
            has_access = True
            
    if not has_access:
        return jsonify({'error': 'Unauthorized'}), 403

    if todo.is_group and todo.group_id:
        # Shared Group Logic
        completed_ids = todo.completed_by.split(',') if todo.completed_by else []
        completed_ids = [uid for uid in completed_ids if uid] # Clean list
        
        str_id = str(current_user.id)
        if str_id in completed_ids:
            completed_ids.remove(str_id)
            # If was globally complete, un-complete it
            if todo.completed:
                todo.completed = False
        else:
            completed_ids.append(str_id)
            # Check global completion
            group_members = GroupMember.query.filter_by(group_id=todo.group_id).all()
            if len(completed_ids) >= len(group_members):
                todo.completed = True
                
        todo.completed_by = ",".join(completed_ids)
        
        # Award XP only if marking as done individualy (simple logic for now)
        if str_id in completed_ids:
             GamificationService.add_xp(current_user.id, 'task', 10)
             flash('Group Task marked finished! +10 XP', 'success')
             
    else:
        # Personal Task Logic
        todo.completed = not todo.completed
        if todo.completed:
            GamificationService.add_xp(current_user.id, 'task', 10)
            flash('Task completed! +10 XP', 'success')
        
    db.session.commit()
    
    next_url = request.form.get('next') or request.args.get('next')
    if next_url:
        return redirect(next_url)

    return redirect(url_for('todos'))

@app.route('/todos/delete/<int:todo_id>', methods=['POST'])
@login_required
def todos_delete(todo_id):
    todo = Todo.query.get_or_404(todo_id)

    # Authorization Check
    authorized = False
    if not todo.is_group:
        if todo.user_id == current_user.id:
            authorized = True
    else:
        # Only Group Admin can delete group tasks
        if todo.group_id:
            group = Group.query.get(todo.group_id)
            if group and group.admin_id == current_user.id:
                authorized = True
        elif todo.user_id == current_user.id: # Fallback for old loose group tasks
             authorized = True
             
    if not authorized:
        flash('You do not have permission to delete this task.', 'error')
        return redirect(url_for('todos'))

    # DS concept: push deleted item into undo stack stored in session
    undo_stack = session.get('todo_undo_stack', [])
    undo_stack.append({
        'title': todo.title,
        'priority': todo.priority,
        'due_date': todo.due_date,
        'category': todo.category,
        'is_group': bool(todo.is_group),
    })
    session['todo_undo_stack'] = undo_stack[-20:]  # cap stack size

    db.session.delete(todo)
    db.session.commit()

    next_url = request.form.get('next') or request.args.get('next')
    if next_url:
        return redirect(next_url)

    return redirect(url_for('todos'))

@app.route('/todos/undo', methods=['POST'])
@login_required
def todos_undo():
    undo_stack = Stack()
    for item in session.get('todo_undo_stack', []):
        undo_stack.push(item)

    last = undo_stack.pop()
    if last is None:
        return redirect(url_for('todos'))

    # Write the updated stack back to session
    remaining = []
    while not undo_stack.is_empty():
        remaining.append(undo_stack.pop())
    session['todo_undo_stack'] = list(reversed(remaining))

    db.session.add(Todo(
        user_id=current_user.id,
        title=last['title'],
        completed=False,
        priority=last.get('priority', 'medium'),
        due_date=last.get('due_date'),
        category=last.get('category'),
        is_group=last.get('is_group', False),
    ))
    db.session.commit()
    return redirect(url_for('todos'))

@app.route('/pomodoro')
@login_required
def pomodoro():
    return render_template('pomodoro.html')

@app.route('/pomodoro/sessions', methods=['POST'])
@login_required
def pomodoro_save_session():
    """Save completed Pomodoro session to database."""
    duration = request.form.get('duration', type=int)
    mode = request.form.get('mode', 'focus')
    subject_id = request.form.get('subject_id', type=int)
    
    if duration:
        study_session = StudySession(
            user_id=current_user.id,
            duration=duration,
            mode=mode,
            subject_id=subject_id,
            completed_at=datetime.utcnow()
        )
        db.session.add(study_session)
        
        # Award XP: 1 XP per minute of focus
        if mode == 'focus':
            xp_amount = duration
            result = GamificationService.add_xp(current_user.id, 'focus', xp_amount)
            if result:
                 # We can return this to UI if we want a popup
                 pass
        
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Session saved'})
    
    return jsonify({'status': 'error', 'message': 'Invalid duration'}), 400

@app.route('/pomodoro/goals', methods=['GET'])
@login_required
def pomodoro_get_goals():
    """Fetch session goals (Todos with category='Session Goal')."""
    goals = Todo.query.filter_by(user_id=current_user.id, category='Session Goal').order_by(Todo.created_at.asc()).all()
    return jsonify([
        {
            'id': g.id,
            'title': g.title,
            'completed': g.completed
        } for g in goals
    ])

@app.route('/pomodoro/goals', methods=['POST'])
@login_required
def pomodoro_add_goal():
    """Add a new session goal."""
    data = request.get_json()
    title = data.get('title', '').strip()
    if not title:
        return jsonify({'error': 'Title is required'}), 400

    goal = Todo(
        user_id=current_user.id,
        title=title,
        completed=False,
        category='Session Goal',
        priority='medium',
        is_group=False
    )
    db.session.add(goal)
    db.session.commit()
    
    return jsonify({
        'id': goal.id,
        'title': goal.title,
        'completed': goal.completed
    })

@app.route('/pomodoro/goals/<int:goal_id>/toggle', methods=['POST'])
@login_required
def pomodoro_toggle_goal(goal_id):
    """Toggle completion status of a session goal."""
    goal = Todo.query.filter_by(id=goal_id, user_id=current_user.id, category='Session Goal').first_or_404()
    goal.completed = not goal.completed
    
    if goal.completed:
        # Mini reward for session goals
        GamificationService.add_xp(current_user.id, 'session_goal', 5)
        
    db.session.commit()
    return jsonify({'status': 'success', 'completed': goal.completed})

@app.route('/pomodoro/goals/<int:goal_id>/update', methods=['POST'])
@login_required
def pomodoro_update_goal(goal_id):
    """Update title of a session goal."""
    goal = Todo.query.filter_by(id=goal_id, user_id=current_user.id, category='Session Goal').first_or_404()
    
    data = request.get_json()
    new_title = data.get('title', '').strip()
    
    if new_title:
        goal.title = new_title
        db.session.commit()
        return jsonify({'status': 'success', 'title': goal.title})
    
    return jsonify({'error': 'Empty title'}), 400

@app.route('/pomodoro/goals/<int:goal_id>/delete', methods=['POST'])
@login_required
def pomodoro_delete_goal(goal_id):
    """Delete a session goal."""
    goal = Todo.query.filter_by(id=goal_id, user_id=current_user.id, category='Session Goal').first_or_404()
    db.session.delete(goal)
    db.session.commit()
    return jsonify({'status': 'success'})


@app.route('/syllabus')
@login_required
def syllabus():
    doc = SyllabusDocument.query.filter_by(user_id=current_user.id).first()
    chapters = SyllabusService.build_chapters_from_todos(current_user.id)
    total_topics = sum(c['total'] for c in chapters)
    completed_topics = sum(c['completed'] for c in chapters)
    avg_completion = int((completed_topics / total_topics) * 100) if total_topics else 0
    return render_template(
        'syllabus.html',
        syllabus_doc=doc,
        chapters=chapters,
        chapters_count=len(chapters),
        topics_count=total_topics,
        completed_count=completed_topics,
        avg_completion=avg_completion,
    )

@app.route('/syllabus/upload', methods=['POST'])
@login_required
def syllabus_upload():
    """Upload and extract PDF syllabus."""
    uploaded = request.files.get('pdf')
    if not uploaded:
        flash('Please select a PDF file.', 'error')
        return redirect(url_for('syllabus'))

    filename = uploaded.filename or 'syllabus.pdf'

    pdf_bytes = uploaded.read()
    if not pdf_bytes:
        flash('Uploaded PDF was empty.', 'error')
        return redirect(url_for('syllabus'))

    # AI: Extract tasks directly from the PDF using Gemini (real API).
    tasks = []
    try:
        tasks = SyllabusService.extract_tasks_from_pdf(pdf_bytes)
    except Exception as e:
        flash(f'AI task extraction failed: {str(e)}', 'error')

    # Extract text with PyPDF2 (used as context for chat). Some PDFs (scans) may yield no text.
    extracted = ""
    try:
        from PyPDF2 import PdfReader
        reader = PdfReader(io.BytesIO(pdf_bytes))
        parts = []
        for page in reader.pages:
            text = page.extract_text() or ''
            parts.append(text)
        extracted = "\n".join(parts).strip()
    except Exception:
        extracted = ""

    if not extracted:
        # Keep a non-empty placeholder because SyllabusDocument.extracted_text is NOT NULL.
        extracted = f"(No text could be extracted from this PDF. It may be a scanned document.)\nFilename: {filename}"
        flash('PDF uploaded, but no text could be extracted (might be scanned image). Tasks can still be generated by AI.', 'error')

    SyllabusService.save_syllabus(current_user.id, filename, extracted)

    # Persist AI-generated tasks into real Todos.
    created_count = 0
    if tasks:
        for task in tasks:
            chapter = str(task.get("title", "")).strip() or "Chapter"
            subtasks = task.get("subtasks", [])
            chapter_category = chapter[:50]

            # Create a parent todo for the chapter
            chapter_title = chapter[:200]
            exists = Todo.query.filter_by(user_id=current_user.id, title=chapter_title, category=chapter_category, is_group=False).first()
            if not exists:
                db.session.add(Todo(
                    user_id=current_user.id,
                    title=chapter_title,
                    completed=False,
                    priority='medium',
                    due_date=None,
                    category=chapter_category,
                    is_group=False,
                ))
                created_count += 1

            if isinstance(subtasks, list):
                for sub in subtasks:
                    sub_title = str(sub).strip()
                    if not sub_title:
                        continue
                    sub_title = sub_title[:200]
                    exists = Todo.query.filter_by(user_id=current_user.id, title=sub_title, category=chapter_category, is_group=False).first()
                    if exists:
                        continue
                    db.session.add(Todo(
                        user_id=current_user.id,
                        title=sub_title,
                        completed=False,
                        priority='medium',
                        due_date=None,
                        category=chapter_category,
                        is_group=False,
                    ))
                    created_count += 1

        db.session.commit()

    if created_count > 0:
        flash(f'Created {created_count} tasks from PDF using Gemini!', 'success')
    else:
        flash('PDF uploaded and processed successfully!', 'success')
    return redirect(url_for('syllabus'))

@app.route('/progress')
@login_required
def progress():
    total_todos = Todo.query.filter_by(user_id=current_user.id).count()
    completed_todos = Todo.query.filter_by(user_id=current_user.id, completed=True).count()
    completion_percent = int((completed_todos / total_todos) * 100) if total_todos else 0

    week_ago = datetime.utcnow() - timedelta(days=7)
    weekly_minutes = (
        db.session.query(db.func.coalesce(db.func.sum(StudySession.duration), 0))
        .filter(StudySession.user_id == current_user.id)
        .filter(StudySession.completed_at >= week_ago)
        .scalar()
    )
    weekly_hours = round((weekly_minutes or 0) / 60.0, 1)
    sessions_week = StudySession.query.filter_by(user_id=current_user.id).filter(StudySession.completed_at >= week_ago).count()

    # Consecutive-day streak based on completed sessions.
    streak = 0
    today = datetime.utcnow().date()
    days_with_sessions = {
        str(r[0])
        for r in (
            db.session.query(db.func.date(StudySession.completed_at))
            .filter(StudySession.user_id == current_user.id)
            .distinct()
            .all()
        )
        if r and r[0]
    }
    while (today - timedelta(days=streak)).isoformat() in days_with_sessions:
        streak += 1

    # Study hours per day for the last 7 days (for simple bar chart).
    daily = []
    max_hours = 0.0
    for i in range(6, -1, -1):
        day = (today - timedelta(days=i))
        start_dt = datetime.combine(day, datetime.min.time())
        end_dt = datetime.combine(day, datetime.max.time())
        minutes = (
            db.session.query(db.func.coalesce(db.func.sum(StudySession.duration), 0))
            .filter(StudySession.user_id == current_user.id)
            .filter(StudySession.completed_at >= start_dt)
            .filter(StudySession.completed_at <= end_dt)
            .scalar()
        )
        hours = round((minutes or 0) / 60.0, 1)
        max_hours = max(max_hours, hours)
        
        # Format label (e.g. "1.5h" or "45m")
        total_minutes = int(minutes or 0)
        if total_minutes < 60:
            display_val = f"{total_minutes}m"
        else:
            h = total_minutes // 60
            m = total_minutes % 60
            if m > 0:
                display_val = f"{h}h {m}m"
            else:
                display_val = f"{h}h"
                
        daily.append({
            'label': day.strftime('%a'),
            'hours': hours,
            'minutes': total_minutes,
            'display': display_val
        })

    for d in daily:
        # Use minutes for more precise percentage relative to max
        # If max_hours is 0, percent is 0.
        # max_minutes = max_hours * 60
        max_minutes = max([x['minutes'] for x in daily]) if daily else 0
        
        if max_minutes > 0:
            d['percent'] = int((d['minutes'] / max_minutes) * 100)
            # Ensure at least a sliver is visible if there's any time
            if d['minutes'] > 0 and d['percent'] < 5:
                d['percent'] = 5
        else:
            d['percent'] = 0

    top_topics = (
        TopicProficiency.query
        .filter_by(user_id=current_user.id)
        .order_by(TopicProficiency.proficiency.desc())
        .limit(5)
        .all()
    )

    return render_template(
        'progress.html',
        total_todos=total_todos,
        completed_todos=completed_todos,
        completion_percent=completion_percent,
        weekly_hours=weekly_hours,
        sessions_week=sessions_week,
        day_streak=streak,
        daily_hours=daily,
        top_topics=top_topics,
    )

@app.route('/settings')
@login_required
def settings():
    return profile(current_user.id)

@app.route('/profile/<int:user_id>')
@login_required
def profile(user_id):
    user = User.query.get_or_404(user_id)
    
    # Ensure badges are up to date
    GamificationService.check_badges(user)
    db.session.commit()
    
    badges = UserBadge.query.filter_by(user_id=user.id).all()
    # Calculate stats for the target user
    total_focus_minutes = db.session.query(db.func.sum(StudySession.duration))\
        .filter(StudySession.user_id == user.id).scalar() or 0
    total_focus_hours = round(total_focus_minutes / 60, 1)

    # Fetch tasks for Calendar (only if viewing own profile)
    calendar_events = []
    if current_user.is_authenticated and user.id == current_user.id:
        calendar_events = Todo.query.filter_by(user_id=user.id, completed=False)\
            .filter(Todo.due_date.isnot(None))\
            .filter(Todo.due_date != '')\
            .order_by(Todo.due_date.asc())\
            .all()

    return render_template('profile.html', user=user, badges=badges, total_focus_hours=total_focus_hours, calendar_events=calendar_events)

@app.route('/calendar')
@login_required
def calendar_view():
    # Only show uncompleted tasks with due dates for the calendar
    calendar_events = Todo.query.filter_by(user_id=current_user.id, completed=False)\
        .filter(Todo.due_date.isnot(None))\
        .filter(Todo.due_date != '')\
        .order_by(Todo.due_date.asc())\
        .all()
    
    return render_template('calendar.html', calendar_events=calendar_events)

@app.route('/settings/public-profile', methods=['POST'])
@login_required
def update_public_profile():
    data = request.get_json()
    if not data:
        return jsonify({'error': 'No data provided'}), 400
        
    is_public = data.get('is_public', True)
    current_user.is_public_profile = bool(is_public)
    db.session.commit()
    
    return jsonify({'status': 'success'})

@app.route('/settings/update', methods=['POST'])
@login_required
def settings_update():
    data = request.get_json()
    if not data:
        return jsonify({'status': 'error', 'message': 'No data provided'}), 400

    first_name = data.get('first_name', '').strip()
    last_name = data.get('last_name', '').strip()
    email = data.get('email', '').strip().lower()

    if not first_name or not email:
        return jsonify({'status': 'error', 'message': 'First Name and Email are required'}), 400

    # formatting check for email could go here

    current_user.first_name = first_name
    current_user.last_name = last_name
    
    # Check if email is being changed and if it's taken
    if email != current_user.email:
        existing = User.query.filter_by(email=email).first()
        if existing:
            return jsonify({'status': 'error', 'message': 'Email already in use'}), 400
        current_user.email = email
    current_user.about_me = data.get('about_me', current_user.about_me)

    try:
        db.session.commit()
        return jsonify({'status': 'success', 'message': 'Profile updated successfully'})
    except Exception as e:
        db.session.rollback()
        return jsonify({'status': 'error', 'message': str(e)}), 500

# NOTE: We intentionally avoid JSON-based API endpoints for this semester project.


def call_ai_api(messages):
    """Call the configured AI API service with retry logic"""
    
    if not AI_API_KEY:
        return "⚠️ AI API key not configured. Please set AI_API_KEY environment variable or add it to app.py. For now, I'm your study companion. How can I help you today?"
    
    max_retries = 3
    retry_delay = 2 # initial delay in seconds
    
    for attempt in range(max_retries):
        try:
            if AI_API_TYPE == 'openai':
                # OpenAI API
                print(f"Calling OpenAI API with model: {AI_MODEL} (Attempt {attempt + 1})")
                
                response = requests.post(
                    'https://api.openai.com/v1/chat/completions',
                    headers={
                        'Authorization': f'Bearer {AI_API_KEY}',
                        'Content-Type': 'application/json'
                    },
                    json={
                        'model': AI_MODEL,
                        'messages': messages,
                        'temperature': 0.7,
                        'max_tokens': 1000
                    },
                    timeout=30
                )
                
                if response.status_code == 429:
                    print(f"Rate limited by OpenAI. Retrying in {retry_delay}s...")
                    time.sleep(retry_delay)
                    retry_delay *= 2
                    continue
                    
                if response.status_code != 200:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error', {}).get('message', response.text or 'Unknown error')
                    print(f"OpenAI API Error: {error_msg}")
                    return f"⚠️ API Error: {error_msg}"
                
                response.raise_for_status()
                result = response.json()['choices'][0]['message']['content']
                print("OpenAI API Success!")
                return result
                
            elif AI_API_TYPE == 'google':
                # Google Gemini API - Using REST API
                print(f"Calling Google Gemini API with model: {AI_MODEL} (Attempt {attempt + 1})")
                
                system_message = None
                gemini_contents = []
                
                for msg in messages:
                    if msg['role'] == 'system':
                        system_message = msg['content']
                    elif msg['role'] == 'user':
                        gemini_contents.append({
                            'role': 'user',
                            'parts': [{'text': msg['content']}]
                        })
                    elif msg['role'] == 'assistant':
                        gemini_contents.append({
                            'role': 'model',
                            'parts': [{'text': msg['content']}]
                        })
                
                if system_message:
                    if gemini_contents and gemini_contents[0]['role'] == 'user':
                        gemini_contents[0]['parts'][0]['text'] = system_message + "\n\n" + gemini_contents[0]['parts'][0]['text']
                    else:
                        gemini_contents.insert(0, {
                            'role': 'user',
                            'parts': [{'text': system_message}]
                        })
                
                payload = {'contents': gemini_contents}
                
                # Robust endpoint construction
                if "/" in AI_MODEL:
                    endpoint = f'https://generativelanguage.googleapis.com/v1beta/{AI_MODEL}:generateContent?key={AI_API_KEY}'
                else:
                    endpoint = f'https://generativelanguage.googleapis.com/v1beta/models/{AI_MODEL}:generateContent?key={AI_API_KEY}'
                
                response = requests.post(endpoint, headers={'Content-Type': 'application/json'}, json=payload, timeout=30)
                
                print(f"Google Gemini API Response Status: {response.status_code}")
                
                if response.status_code == 429:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error', {}).get('message', '')
                    
                    wait_time = retry_delay
                    # Try to parse "retry in X.Xs" from message
                    match = re.search(r'retry in ([\d\.]+)s', error_msg.lower())
                    if match:
                        wait_time = float(match.group(1)) + 1
                    
                    print(f"Rate limited by Google. Retrying in {wait_time}s...")
                    time.sleep(wait_time)
                    retry_delay *= 2
                    continue
                
                if response.status_code != 200:
                    error_data = response.json() if response.text else {}
                    error_msg = error_data.get('error', {}).get('message', response.text or 'Unknown error')
                    print(f"Google Gemini API Error: {error_msg}")
                    return f"⚠️ API Error: {error_msg}"
                
                result_data = response.json()
                if 'candidates' in result_data and len(result_data['candidates']) > 0:
                    result = result_data['candidates'][0]['content']['parts'][0]['text']
                    print("Google Gemini API Success!")
                    return result
                else:
                    return "⚠️ Error: Unexpected response format from Gemini API"
                
            else:
                return f"⚠️ Unknown AI API type: {AI_API_TYPE}. Supported types: openai, anthropic, lovable, google"
                
        except (requests.exceptions.RequestException, Exception) as e:
            if attempt < max_retries - 1:
                print(f"Connection/Unexpected Error: {str(e)}. Retrying...")
                time.sleep(retry_delay)
                retry_delay *= 2
                continue
            return f"⚠️ Error: {type(e).__name__}: {str(e)}"
            
    return "⚠️ Error: Maximum retries exceeded. Please try again later."



# ------------------------------
# SocketIO & Real-time Logic
# ------------------------------

@socketio.on('join')
def on_join(data):
    group_id = data.get('group_id')
    if group_id:
        join_room(str(group_id))

@socketio.on('send_message')
def handle_message(data):
    group_id = data.get('group_id')
    content = data.get('content', '')
    file_path = data.get('file_path')
    
    print(f"DEBUG: Message received for Group {group_id} from User {current_user.id}")
    
    if not group_id or not current_user.is_authenticated:
        print("DEBUG: Message rejected - missing group_id or auth")
        return

    # Check membership
    membership = GroupMember.query.filter_by(group_id=group_id, user_id=current_user.id).first()
    if not membership:
        print(f"DEBUG: Message rejected - User {current_user.id} not in Group {group_id}")
        return

    msg = GroupChatMessage(
        group_id=group_id,
        user_id=current_user.id,
        role='user',
        content=content,
        file_path=file_path
    )
    db.session.add(msg)
    db.session.commit()
    print(f"DEBUG: Message saved ID {msg.id}")

    # Convert timestamp to IST
    ist_time = to_ist_time(msg.created_at)

    emit('receive_message', {
        'id': msg.id,
        'user_id': current_user.id,
        'username': current_user.first_name or 'User',
        'content': msg.content,
        'file_path': msg.file_path,
        'created_at': ist_time,
        'role': 'user'
    }, room=str(group_id))

    
    # AI Logic (Simple mention check)
    if '@StudyVerse' in content.lower() or '@assistant' in content.lower():
        reply = ChatService.personal_reply(current_user, content)
        ai_msg = GroupChatMessage(group_id=group_id, user_id=None, role='assistant', content=reply)
        db.session.add(ai_msg)
        db.session.commit()
        
        # Convert AI message timestamp to IST
        ai_ist_time = to_ist_time(ai_msg.created_at)
        
        emit('receive_message', {
            'id': ai_msg.id,
            'user_id': None,
            'username': 'StudyVerse',
            'content': ai_msg.content,
            'created_at': ai_ist_time,
            'role': 'assistant'
        }, room=str(group_id))


@app.route('/group/upload', methods=['POST'])
@login_required
def group_upload():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    filename = secure_filename(file.filename)
    timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
    unique_filename = f"{timestamp}_{filename}"
    save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
    file.save(save_path)
    
    return jsonify({
        'url': url_for('static', filename=f'uploads/{unique_filename}'),
        'filename': filename
    })

@app.route('/profile/upload_cover', methods=['POST'])
@login_required
def profile_upload_cover():
    if 'file' not in request.files:
        return jsonify({'error': 'No file part'}), 400
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No selected file'}), 400
    
    if file:
        filename = secure_filename(file.filename)
        # Use timestamp to avoid caching issues
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        unique_filename = f"cover_{current_user.id}_{timestamp}_{filename}"
        save_path = os.path.join(app.config['UPLOAD_FOLDER'], unique_filename)
        file.save(save_path)
        
        # Update user profile
        current_user.cover_image = f"uploads/{unique_filename}"
        db.session.commit()
        
        return jsonify({
            'status': 'success',
            'url': url_for('static', filename=f'uploads/{unique_filename}')
        })
    
    return jsonify({'error': 'Upload failed'}), 500

# ------------------------------
# BYTE BATTLE LOGIC (1v1 AI Referee)
# ------------------------------

@app.route('/battle')
@login_required
def battle():
    return render_template('battle.html')

# In-memory battle state (with persistence)
BATTLES_FILE = 'battles_data.json'
battles = {}

def load_battles():
    global battles
    if os.path.exists(BATTLES_FILE):
        try:
            with open(BATTLES_FILE, 'r') as f:
                battles = json.load(f)
            # Re-convert timestamps if necessary or just use as is (JSON keys are always strings)
            # JSON keys for players dict will be strings, but code often expects int for dict lookup
            # We need to handle this type conversion carefully if IDs are ints.
            # In existing code: room['players'][current_user.id] implies ID is key.
            # When loaded from JSON, keys in 'players' dict "123" will be string.
            # We must fix this structure after loading.
            for room in battles.values():
                if 'players' in room:
                    room['players'] = {int(k) if k.isdigit() else k: v for k, v in room['players'].items()}
                if 'submissions' in room:
                    room['submissions'] = {int(k) if k.isdigit() else k: v for k, v in room['submissions'].items()}
            print(f"Loaded {len(battles)} battles from {BATTLES_FILE}")
        except Exception as e:
            print(f"Error loading battles: {e}")
            battles = {}

def save_battles():
    try:
        with open(BATTLES_FILE, 'w') as f:
            json.dump(battles, f, default=str) # default=str to handle datetime if any
    except Exception as e:
        print(f"Error saving battles: {e}")

# Load on startup
load_battles()

def generate_room_code(length=4):
    import random, string
    chars = string.ascii_uppercase + string.digits
    while True:
        code = ''.join(random.choice(chars) for _ in range(length))
        if code not in battles:
            return code

@socketio.on('battle_create')
def on_battle_create(data):
    if not current_user.is_authenticated:
        return
    
    room_code = generate_room_code()
    battles[room_code] = {
        'host': current_user.id,
        'players': {
            current_user.id: {
                'name': current_user.first_name or 'Player 1',
                'sid': request.sid,
                'joined_at': datetime.utcnow().isoformat() # improved for JSON serialization
            }
        },
        'state': 'waiting', # waiting, setup, battle, judging, result
        'config': {'difficulty': None, 'language': None},
        'problem': None,
        'submissions': {},
        'rematch_votes': {}, # player_id: "yes"/"no"
        'pending_join': None # Stores info about player requesting to join
    }
    save_battles()
    
    join_room(room_code)
    emit('battle_created', {'room_code': room_code, 'player_id': current_user.id})
    print(f"Battle created: {room_code} by {current_user.first_name}")

@socketio.on('battle_join_request')
def on_battle_join_request(data):
    if not current_user.is_authenticated:
        return
        
    room_code = data.get('room_code', '').strip().upper()
    print(f"DEBUG: Join request for {room_code} by {current_user.first_name} (ID: {current_user.id})")
    
    if room_code not in battles:
        emit('battle_error', {'message': 'Invalid room code.'})
        return
        
    room = battles[room_code]
    if len(room['players']) >= 2:
        emit('battle_error', {'message': 'Room is full.'})
        return

    # Check if already in (re-join)
    if current_user.id in room['players']:
        room['players'][current_user.id]['sid'] = request.sid
        save_battles() # Update SID
        join_room(room_code)
        emit('battle_rejoined', {'state': room['state'], 'room_code': room_code})
        return

    # Store pending request
    room['pending_join'] = {
        'id': current_user.id,
        'name': current_user.first_name or 'Opponent',
        'sid': request.sid
    }
    save_battles()
    
    # Notify Host
    host_id = room['host']
    if host_id in room['players']:
        host_sid = room['players'][host_id]['sid']
        socketio.emit('battle_join_request_notify', {
            'player_name': room['pending_join']['name']
        }, room=host_sid)

@socketio.on('battle_join_response')
def on_battle_join_response(data):
    room_code = data.get('room_code')
    accepted = data.get('accepted')
    
    if not room_code or room_code not in battles:
        return
    
    room = battles[room_code]
    # Only host can accept
    if current_user.id != room['host']:
        return
        
    pending = room.get('pending_join')
    if not pending:
        return
        
    if accepted:
        # Add player
        # Ensure pending ID is int if originating from JSON
        p_id = int(pending['id']) if isinstance(pending['id'], (str, int)) and str(pending['id']).isdigit() else pending['id']
        
        room['players'][p_id] = {
            'name': pending['name'],
            'sid': pending['sid'],
            'joined_at': datetime.utcnow().isoformat()
        }
        room['pending_join'] = None
        save_battles()
        
        # Manually join the socket room for the new player
        socketio.emit('join_accepted', {'room_code': room_code}, room=pending['sid'])
    else:
        socketio.emit('battle_error', {'message': 'Host rejected your request.'}, room=pending['sid'])
        room['pending_join'] = None
        save_battles()

@socketio.on('battle_confirm_join')
def on_battle_confirm_join(data):
    room_code = data.get('room_code')
    if room_code in battles and current_user.id in battles[room_code]['players']:
        join_room(room_code)
        
        # Room is full, move to SETUP immediately
        room = battles[room_code]
        room['state'] = 'setup'
        save_battles()
        
        # Notify both to open UI
        emit('battle_entered', {'room_code': room_code}, room=room_code)
        
        # AI Welcome Message
        host_name = room['players'][room['host']]['name']
        socketio.emit('battle_chat_message', {
            'sender': 'ByteBot',
            'message': (
                "Welcome to Byte Battle ⚔️\n"
                "Both players are connected.\n\n"
                f"Host ({host_name}), please select:\n"
                "• Difficulty: Easy / Medium / Hard\n"
                "• Language: Python / JS / Java / C"
            ),
            'type': 'system'
        }, room=room_code)

@socketio.on('battle_chat_send')
def on_battle_chat_send(data):
    room_code = data.get('room_code')
    message = data.get('message', '').strip()
    
    if not room_code or room_code not in battles:
        return
        
    room = battles[room_code]
    player = room['players'].get(current_user.id)
    if not player:
        return

    # Broadcast user message
    emit('battle_chat_message', {
        'sender': player['name'],
        'message': message,
        'type': 'user'
    }, room=room_code)
    
    # Handle Setup Logic via Chat
    if room['state'] == 'setup':
        if current_user.id == room['host']:
            # Parse settings
            msg_lower = message.lower()
            
            # Difficulty
            if 'easy' in msg_lower: room['config']['difficulty'] = 'Easy'
            elif 'medium' in msg_lower: room['config']['difficulty'] = 'Medium'
            elif 'hard' in msg_lower: room['config']['difficulty'] = 'Hard'
            
            # Language
            if 'python' in msg_lower: room['config']['language'] = 'Python'
            elif 'javascript' in msg_lower or 'js' in msg_lower: room['config']['language'] = 'JavaScript'
            elif 'java' in msg_lower: room['config']['language'] = 'Java'
            elif 'c++' in msg_lower or 'cpp' in msg_lower: room['config']['language'] = 'C++'
            elif 'c' in msg_lower: room['config']['language'] = 'C'
            
            # Check if done
            config = room['config']
            if config['difficulty'] and config['language']:
                room['state'] = 'generating'
                emit('battle_chat_message', {
                    'sender': 'ByteBot',
                    'message': f"Configuration locked: {config['difficulty']} | {config['language']}.\nGenerating problem...",
                    'type': 'system'
                }, room=room_code)
                
                # Start Battle
                socketio.start_background_task(start_battle_task, room_code)
            else:
                 # Feedback on what's missing
                 missing = []
                 if not config['difficulty']: missing.append("Difficulty")
                 if not config['language']: missing.append("Language")
                 if missing:
                      # Only reply if it looked like an attempt (optional, to avoid spam)
                      pass 

def start_battle_task(room_code):
    with app.app_context():
        room = battles[room_code]
        config = room['config']
        
        problem = generate_battle_problem(config['difficulty'], config['language'])
        room['problem'] = problem
        room['state'] = 'battle'
        room['start_time'] = datetime.utcnow().timestamp()
        
        # Announce problem
        socketio.emit('battle_chat_message', {
            'sender': 'ByteBot',
            'message': "Here is your challenge.\nTimer has started.",
            'type': 'system'
        }, room=room_code)
        
        socketio.emit('battle_started', {
            'problem': problem,
            'duration': 600, # 10 mins
            'language': config['language']
        }, room=room_code)

@socketio.on('battle_submit')
def on_battle_submit(data):
    room_code = data.get('room_code')
    code = data.get('code')
    
    if not room_code or room_code not in battles:
        return
        
    room = battles[room_code]
    if room['state'] != 'battle':
        return
        
    # Store submission
    submission_time = datetime.utcnow().timestamp()
    time_taken = submission_time - room.get('start_time', submission_time)
    
    room['submissions'][current_user.id] = {
        'code': code,
        'time_taken': time_taken,
        'player_name': room['players'][current_user.id]['name']
    }
    
    # Notify others
    emit('battle_notification', {'message': f"🔔 {room['players'][current_user.id]['name']} has submitted their solution."}, room=room_code)
    emit('battle_chat_message', {'sender': 'ByteBot', 'message': f"🔔 {room['players'][current_user.id]['name']} has submitted their solution.", 'type': 'system'}, room=room_code)
    
    # Check if all submitted
    if len(room['submissions']) == len(room['players']):
        room['state'] = 'judging'
        emit('battle_state_change', {'state': 'judging'}, room=room_code)
        socketio.start_background_task(judge_battle, room_code)

def generate_battle_problem(difficulty, language):
    prompt = (
        f"Generate a single {difficulty} difficulty coding interview problem suitable for {language}. "
        "Return ONLY valid JSON with this structure: "
        "{ \"title\": \"Problem Title\", \"description\": \"Clear problem statement...\", "
        "\"input_format\": \"Input description...\", \"output_format\": \"Output description...\", "
        "\"example_input\": \"...\", \"example_output\": \"...\" }"
    )
    
    try:
        response = call_ai_api([{'role': 'user', 'content': prompt}])
        # Cleanup JSON
        if "```json" in response:
            response = response.split("```json")[1].split("```")[0]
        elif "```" in response:
            response = response.split("```")[1].split("```")[0]
            
        return json.loads(response.strip())
    except Exception as e:
        print(f"Error generating problem: {e}")
        return {
            "title": "Palindrome Check",
            "description": "Write a program to check if a string is a palindrome.",
            "input_format": "A single string S.",
            "output_format": "Print 'YES' if palindrome, else 'NO'.",
            "example_input": "racecar",
            "example_output": "YES"
        }

def judge_battle(room_code):
    """Background task to judge the battle"""
    with app.app_context():
        room = battles.get(room_code)
        if not room:
            return

        submissions = list(room['submissions'].values())
        if not submissions:
            return

        problem_desc = json.dumps(room['problem'])
        subs_text = ""
        for i, sub in enumerate(submissions):
            subs_text += f"\nPlayer ({sub['player_name']}) Code [Time: {round(sub['time_taken'],1)}s]:\n{sub['code']}\n"

        prompt = (
            f"You are the referee of a coding battle. Problem: {problem_desc}\n"
            f"Submissions: {subs_text}\n"
            "Evaluate based on: 1. Correctness (Passes all edge cases?) 2. Logic quality 3. Time Taken.\n"
            "Return ONLY valid JSON: "
            "{ \"winner\": \"Player Name\" (or 'Draw'), \"reason\": \"Why they won...\", "
            "\"winner_id\": 123 (user id or null if draw), "
            "\"scores\": { \"Player 1 Name\": 90, \"Player 2 Name\": 85 } }"
        )
        
        # Helper to find user ID by name (AI might not return ID perfectly, so we map names)
        # Actually better to ask AI for index or just rely on name matching?
        # Let's map names to IDs first.
        name_to_id = { p['name']: pid for pid, p in room['players'].items() }
        
        try:
            response = call_ai_api([{'role': 'user', 'content': prompt}])
             # Cleanup JSON
            if "```json" in response:
                response = response.split("```json")[1].split("```")[0]
            elif "```" in response:
                response = response.split("```")[1].split("```")[0]
            
            result = json.loads(response.strip())
            
            # Award XP
            difficulty = room['config'].get('difficulty', 'Easy')
            xp_map = {'Easy': 100, 'Medium': 500, 'Hard': 1000}
            base_xp = xp_map.get(difficulty, 100)
            
            winner_name = result.get('winner')
            winner_id = name_to_id.get(winner_name)
            
            if winner_id:
                # Winner gets full XP
                GamificationService.add_xp(winner_id, 'battle_win', base_xp)
                result['xp_awarded'] = {winner_name: base_xp}
            elif winner_name == 'Draw':
                # Both get 50%
                half_xp = int(base_xp * 0.5)
                xp_dict = {}
                for pid in room['players']:
                    GamificationService.add_xp(pid, 'battle_draw', half_xp)
                    pname = room['players'][pid]['name']
                    xp_dict[pname] = half_xp
                result['xp_awarded'] = xp_dict
            
            socketio.emit('battle_result', result, room=room_code)
            
            # Trigger Rematch Question
            socketio.emit('battle_chat_message', {
                'sender': 'ByteBot',
                'message': "Do you want another round? (yes / no)",
                'type': 'system'
            }, room=room_code)
            
        except Exception as e:
            print(f"Judging error: {e}")
            socketio.emit('battle_error', {'message': 'AI Referee failed to judge. It\'s a draw!'}, room=room_code)

@socketio.on('battle_rematch_vote')
def on_battle_rematch_vote(data):
    room_code = data.get('room_code')
    vote = data.get('vote') # 'yes' or 'no'
    
    if not room_code or room_code not in battles:
        return
        
    room = battles[room_code]
    room['rematch_votes'][current_user.id] = vote
    
    player_name = room['players'][current_user.id]['name']
    
    # Notify about the vote
    emit('battle_chat_message', {'sender': 'ByteBot', 'message': f"{player_name} voted: {vote.upper()}", 'type': 'system'}, room=room_code)
    
    # If someone votes NO, end immediately (don't wait for both votes)
    if vote == 'no':
        emit('battle_chat_message', {
            'sender': 'ByteBot', 
            'message': f"{player_name} declined rematch. Battle concluded. Thanks for playing! 👋",
            'type': 'system'
        }, room=room_code)
        
        # Send event to close modal and return both players to entry screen
        emit('battle_rematch_declined', {}, room=room_code)
        return
    
    # If this person voted YES, notify the other player
    emit('battle_chat_message', {
        'sender': 'ByteBot', 
        'message': f"{player_name} wants a rematch! Waiting for opponent's response...",
        'type': 'system'
    }, room=room_code)
    
    # Check if everyone voted (both said yes)
    if len(room['rematch_votes']) == 2:
        votes = list(room['rematch_votes'].values())
        if all(v == 'yes' for v in votes):
            # Restart
            room['state'] = 'setup'
            room['submissions'] = {}
            room['problem'] = None
            room['config'] = {'difficulty': None, 'language': None}
            room['rematch_votes'] = {}
            
            emit('battle_restart', {}, room=room_code)
            emit('battle_chat_message', {
                'sender': 'ByteBot', 
                'message': "Rematch accepted! 🔥 Host, please choose settings again (Easy/Medium/Hard and Python/Java/C/JavaScript).",
                'type': 'system'
            }, room=room_code)



# Profile management can be extended later (kept simple for this semester project).

# ------------------------------
# FRIENDS & PROFILE LOGIC
# ------------------------------

@app.context_processor
def inject_user_context():
    if current_user.is_authenticated:
        # 1. Focus Buddies
        friends = []
        try:
            friendships = Friendship.query.filter(
                ((Friendship.user_id == current_user.id) | (Friendship.friend_id == current_user.id)) & 
                (Friendship.status == 'accepted')
            ).all()
            
            for f in friendships:
                fid = f.friend_id if f.user_id == current_user.id else f.user_id
                friend = User.query.get(fid)
                if friend:
                    # Check online status (within 5 mins)
                    is_online = (datetime.utcnow() - (friend.last_seen or datetime.min)) < timedelta(minutes=5)
                    friends.append({
                        'id': friend.id,
                        'name': f"{friend.first_name} {friend.last_name}",
                        'avatar': friend.get_avatar(64),
                        'is_online': is_online,
                        'is_public': friend.is_public_profile,
                        'rank': GamificationService.get_rank(friend.level) if friend.is_public_profile else None,
                        'stats': {'level': friend.level, 'xp': friend.total_xp} if friend.is_public_profile else None
                    })
        except Exception:
            pass # Fail gracefully if table doesn't exist yet
        
        # 2. Sidebar Stats (Rank, Level Progress)
        current_rank = GamificationService.get_rank(current_user.level)
        # XP per level is 500 (from GamificationService)
        xp_per_level = 500
        current_xp_in_level = current_user.total_xp % xp_per_level
        level_progress = int((current_xp_in_level / xp_per_level) * 100)
        xp_remaining = xp_per_level - current_xp_in_level
        
        return dict(
            focus_buddies=friends,
            rank_name=current_rank['name'],
            rank_icon=current_rank['icon'],
            rank_color=current_rank['color'],
            level_progress=level_progress,
            xp_remaining=xp_remaining
        )
    return dict(focus_buddies=[])

@app.before_request
def update_last_seen():
    if current_user.is_authenticated:
        try:
            current_user.last_seen = datetime.utcnow()
            db.session.commit()
        except:
            pass # Ignore if DB issues

@app.route('/settings/public-profile', methods=['POST'])
@login_required
def toggle_public_profile():
    data = request.get_json()
    current_user.is_public_profile = data.get('is_public', True)
    db.session.commit()
    return jsonify({'status': 'success'})

@app.route('/friends')
@login_required
def friends_page():
    # Helper to format user
    def format_user(u):
        return {
            'id': u.id,
            'name': f"{u.first_name} {u.last_name}",
            'email': u.email,
            'avatar': u.get_avatar(100),
            'level': u.level,
            'rank': GamificationService.get_rank(u.level),
            'is_public': u.is_public_profile
        }

    # 1. My Friends
    accepted = Friendship.query.filter(
        ((Friendship.user_id == current_user.id) | (Friendship.friend_id == current_user.id)) & 
        (Friendship.status == 'accepted')
    ).all()
    my_friends = []
    for f in accepted:
        fid = f.friend_id if f.user_id == current_user.id else f.user_id
        friend = User.query.get(fid)
        if friend:
            my_friends.append(format_user(friend))

    # 2. Friend Requests (Received)
    requests = Friendship.query.filter_by(friend_id=current_user.id, status='pending').all()
    friend_requests = []
    for r in requests:
        sender = User.query.get(r.user_id)
        if sender:
            friend_requests.append({
                'request_id': r.id,
                **format_user(sender)
            })

    return render_template('friends.html', my_friends=my_friends, friend_requests=friend_requests)

@app.route('/api/users/search')
@login_required
def search_users():
    query = request.args.get('q', '').strip()
    if not query or len(query) < 2:
        return jsonify([])
    
    # Search by name or email
    users = User.query.filter(
        (User.id != current_user.id) & 
        (
            (User.email.ilike(f"%{query}%")) | 
            (User.first_name.ilike(f"%{query}%")) | 
            (User.last_name.ilike(f"%{query}%"))
        )
    ).limit(10).all()
    
    results = []
    for u in users:
        # Check friendship status
        friendship = Friendship.query.filter(
            ((Friendship.user_id == current_user.id) & (Friendship.friend_id == u.id)) |
            ((Friendship.user_id == u.id) & (Friendship.friend_id == current_user.id))
        ).first()
        
        status = 'none'
        if friendship:
            status = friendship.status
            if status == 'pending' and friendship.friend_id == current_user.id:
                status = 'received' # Request received from this user
            elif status == 'pending' and friendship.user_id == current_user.id:
                status = 'sent' # Request sent to this user
        
        results.append({
            'id': u.id,
            'name': f"{u.first_name} {u.last_name}",
            'email': u.email,
            'avatar': u.get_avatar(64),
            'status': status
        })
        
    return jsonify(results)

@app.route('/friends/request/<int:user_id>', methods=['POST'])
@login_required
def send_friend_request(user_id):
    if user_id == current_user.id:
        return jsonify({'error': 'Cannot add self'}), 400
        
    target = User.query.get(user_id)
    if not target:
        return jsonify({'error': 'User not found'}), 404
        
    existing = Friendship.query.filter(
        ((Friendship.user_id == current_user.id) & (Friendship.friend_id == user_id)) |
        ((Friendship.user_id == user_id) & (Friendship.friend_id == current_user.id))
    ).first()
    
    if existing:
        return jsonify({'error': 'Friendship or request already exists'}), 400
        
    req = Friendship(user_id=current_user.id, friend_id=user_id, status='pending')
    db.session.add(req)
    db.session.commit()
    
    return jsonify({'status': 'success'})

@app.route('/friends/accept/<int:request_id>', methods=['POST'])
@login_required
def accept_friend_request(request_id):
    req = Friendship.query.get(request_id)
    if not req or req.friend_id != current_user.id:
        return jsonify({'error': 'Invalid request'}), 404
        
    req.status = 'accepted'
    db.session.commit()
    return jsonify({'status': 'success'})

@app.route('/friends/reject/<int:request_id>', methods=['POST'])
@login_required
def reject_friend_request(request_id):
    req = Friendship.query.get(request_id)
    if not req or req.friend_id != current_user.id:
        return jsonify({'error': 'Invalid request'}), 404
        
    db.session.delete(req)
    db.session.commit()
    return jsonify({'status': 'success'})

@app.route('/api/subjects', methods=['GET', 'POST'])
@login_required
def manage_subjects():
    if request.method == 'GET':
        subjects = Subject.query.filter_by(user_id=current_user.id).all()
        return jsonify([{
            'id': s.id,
            'name': s.name,
            'color': s.color
        } for s in subjects])
    
    data = request.get_json()
    name = data.get('name', '').strip()
    color = data.get('color', '#4ade80')
    
    if not name:
        return jsonify({'error': 'Name is required'}), 400
        
    subject = Subject(user_id=current_user.id, name=name, color=color)
    db.session.add(subject)
    db.session.commit()
    
    return jsonify({
        'id': subject.id,
        'name': subject.name,
        'color': subject.color
    })

@app.route('/api/subjects/<int:subject_id>', methods=['PUT', 'DELETE'])
@login_required
def subject_detail(subject_id):
    subject = Subject.query.filter_by(id=subject_id, user_id=current_user.id).first()
    if not subject:
        return jsonify({'error': 'Subject not found'}), 404
        
    if request.method == 'DELETE':
        # Optional: Set subject_id to null for related sessions/tasks?
        StudySession.query.filter_by(subject_id=subject.id).update({'subject_id': None})
        db.session.delete(subject)
        db.session.commit()
        return jsonify({'status': 'success'})
        
    data = request.get_json()
    subject.name = data.get('name', subject.name)
    subject.color = data.get('color', subject.color)
    db.session.commit()
    
    return jsonify({
        'id': subject.id,
        'name': subject.name,
        'color': subject.color
    })

@app.route('/api/analytics/data')
@login_required
def analytics_data():
    subjects = Subject.query.filter_by(user_id=current_user.id).all()
    
    # 1. Subject Distribution & Proficiency
    subject_stats = []
    total_hours = 0
    
    for sub in subjects:
        # Calculate hours
        hours = db.session.query(db.func.sum(StudySession.duration)).filter_by(
            user_id=current_user.id, 
            subject_id=sub.id
        ).scalar() or 0
        hours = round(hours / 60, 1) # Convert to hours
        total_hours += hours
        
        # Calculate "Proficiency" (Mock Logic)
        import math
        base_score = min(100, int(math.log(hours * 10 + 1) * 20))
        
        subject_stats.append({
            'subject': sub.name,
            'color': sub.color,
            'hours': hours,
            'proficiency': base_score
        })
        
    # Add "Uncategorized" if any
    uncat_mins = db.session.query(db.func.sum(StudySession.duration)).filter_by(
        user_id=current_user.id, 
        subject_id=None
    ).scalar() or 0
    if uncat_mins > 0:
        uncat_hours = round(uncat_mins / 60, 1)
        total_hours += uncat_hours
        subject_stats.append({
            'subject': 'Uncategorized',
            'color': '#71717a', # slate-500
            'hours': uncat_hours,
            'proficiency': 0
        })

    # 2. Daily Focus Trend (Dynamic Range)
    today = datetime.utcnow().date()
    daily_trend = []
    labels = []
    
    days_range = request.args.get('days', 7, type=int)
    
    for i in range(days_range - 1, -1, -1):
        day = today - timedelta(days=i)
        day_start = datetime.combine(day, datetime.min.time())
        day_end = datetime.combine(day, datetime.max.time())
        
        day_mins = db.session.query(db.func.sum(StudySession.duration)).filter(
            StudySession.user_id == current_user.id,
            StudySession.completed_at >= day_start,
            StudySession.completed_at <= day_end
        ).scalar() or 0
        
        daily_trend.append(round(day_mins / 60, 1))
        labels.append(day.strftime('%a'))

    return jsonify({
        'subjects': subject_stats,
        'total_hours': round(total_hours, 1),
        'trend': {
            'labels': labels,
            'data': daily_trend
        }
    })


# ==========================================
# Database Initialization & Migration (Production Safe)
# ==========================================
def init_db():
    from sqlalchemy import text, inspect
    
    with app.app_context():
        try:
            db.create_all()
            print("Database tables verified/created.")
            
            # Auto-migration for schema updates
            inspector = inspect(db.engine)
            
            # 1. Check for file_path in group_chat_message
            if 'group_chat_message' in inspector.get_table_names():
                columns = [c['name'] for c in inspector.get_columns('group_chat_message')]
                if 'file_path' not in columns:
                    print("Running migration: Adding file_path to group_chat_message table...")
                    with db.engine.connect() as conn:
                        conn.execute(text("ALTER TABLE group_chat_message ADD COLUMN file_path VARCHAR(255)"))
                        conn.commit()
            
            # 2. Check for columns in user table
            if 'user' in inspector.get_table_names():
                with db.engine.connect() as conn:
                    columns = [c['name'] for c in inspector.get_columns('user')]
                    
                    # New Features (Friends/Public Profile)
                    if 'is_public_profile' not in columns:
                        print("Running migration: Adding is_public_profile to user table...")
                        conn.execute(text("ALTER TABLE user ADD COLUMN is_public_profile BOOLEAN DEFAULT 1"))
                    if 'last_seen' not in columns:
                        print("Running migration: Adding last_seen to user table...")
                        conn.execute(text("ALTER TABLE user ADD COLUMN last_seen DATETIME"))
                    
                    # Existing checks
                    if 'cover_image' not in columns:
                        print("Running migration: Adding cover_image to user table...")
                        conn.execute(text("ALTER TABLE user ADD COLUMN cover_image VARCHAR(255)"))
                    if 'google_id' not in columns:
                        print("Running migration: Adding google_id to user table...")
                        conn.execute(text("ALTER TABLE user ADD COLUMN google_id VARCHAR(100)"))
                    if 'profile_image' not in columns:
                        print("Running migration: Adding profile_image to user table...")
                        conn.execute(text("ALTER TABLE user ADD COLUMN profile_image VARCHAR(255)"))
                    
                    # Gamification Migrations
                    if 'total_xp' not in columns:
                        print("Running migration: Adding total_xp to user table...")
                        conn.execute(text("ALTER TABLE user ADD COLUMN total_xp INTEGER DEFAULT 0"))
                    if 'level' not in columns:
                        print("Running migration: Adding level to user table...")
                        conn.execute(text("ALTER TABLE user ADD COLUMN level INTEGER DEFAULT 1"))
                    if 'current_streak' not in columns:
                        print("Running migration: Adding current_streak to user table...")
                        conn.execute(text("ALTER TABLE user ADD COLUMN current_streak INTEGER DEFAULT 0"))
                    if 'longest_streak' not in columns:
                        print("Running migration: Adding longest_streak to user table...")
                        conn.execute(text("ALTER TABLE user ADD COLUMN longest_streak INTEGER DEFAULT 0"))
                    if 'last_activity_date' not in columns:
                        print("Running migration: Adding last_activity_date to user table...")
                        conn.execute(text("ALTER TABLE user ADD COLUMN last_activity_date DATE"))
                    conn.commit() # Commit user table changes
            
            # 3. Check for study_session columns
            if 'study_session' in inspector.get_table_names():
                with db.engine.connect() as conn:
                    columns = [c['name'] for c in inspector.get_columns('study_session')]
                    if 'subject_id' not in columns:
                        print("Running migration: Adding subject_id to study_session table...")
                        conn.execute(text("ALTER TABLE study_session ADD COLUMN subject_id INTEGER REFERENCES subject(id)"))
                    conn.commit() # Commit study_session changes
            
            print("Migration checks completed.")
        except Exception as e:
            print(f"Migration check failed (safe to ignore if new DB): {e}")

# Optimize connection pooling for Render (High Scalability)
app.config['SQLALCHEMY_ENGINE_OPTIONS'] = {
    'pool_pre_ping': True,
    'pool_recycle': 300,
    'pool_size': 20,       # Handle more concurrent users
    'max_overflow': 10,    # Burst capacity
}
app.config['UPLOAD_FOLDER'] = os.path.join('static', 'uploads')

# ... (Previous config lines) ...

# ==========================================
# Startup Tasks (Background)
# ==========================================
def run_startup_tasks():
    """Runs maintenance tasks without blocking server boot."""
    with app.app_context():
        try:
            # 1. Update XP for Daksh (Requested Feature)
            time.sleep(2) # Wait for DB to be fully ready
            users = User.query.filter(User.first_name.like('Daksh%')).all()
            for user in users:
                print(f"Startup: Updating XP for {user.first_name}...")
                user.total_xp += 1000000
                new_level = GamificationService.calculate_level(user.total_xp)
                if new_level > user.level:
                    user.level = new_level
            if users:
                db.session.commit()
                print("Startup: XP Update Complete.")
        except Exception as e:
            print(f"Startup Task Error: {e}")

# Initialize DB immediately
init_db()

# Spawn background startup task
eventlet.spawn(run_startup_tasks)

if __name__ == '__main__':
    # Use socketio.run instead of app.run
    socketio.run(app, debug=True, host='0.0.0.0', port=5000)
