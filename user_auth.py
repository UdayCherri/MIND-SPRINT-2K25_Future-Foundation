"""
Simple User Database for BAYMAX Authentication
Stores users with hashed passwords
"""

import json
import hashlib
import os
from datetime import datetime
from typing import Optional, Dict, List

USER_DB_PATH = "./user_database.json"

def hash_password(password: str) -> str:
    """Hash password using SHA-256"""
    return hashlib.sha256(password.encode()).hexdigest()

def load_users() -> Dict:
    """Load users from JSON file"""
    if not os.path.exists(USER_DB_PATH):
        # Create default users
        default_users = {
            "demo@baymax.ai": {
                "user_id": "demo_user_123",
                "email": "demo@baymax.ai",
                "password_hash": hash_password("demo123"),
                "name": "Demo User",
                "created_at": datetime.now().isoformat(),
                "last_login": None
            },
            "test@baymax.ai": {
                "user_id": "test_user_456",
                "email": "test@baymax.ai",
                "password_hash": hash_password("test123"),
                "name": "Test User",
                "created_at": datetime.now().isoformat(),
                "last_login": None
            }
        }
        save_users(default_users)
        return default_users
    
    with open(USER_DB_PATH, 'r') as f:
        return json.load(f)

def save_users(users: Dict):
    """Save users to JSON file"""
    with open(USER_DB_PATH, 'w') as f:
        json.dump(users, f, indent=2)

def authenticate_user(email: str, password: str) -> Optional[Dict]:
    """
    Authenticate user with email and password
    Returns user data if successful, None otherwise
    """
    users = load_users()
    
    if email not in users:
        return None
    
    user = users[email]
    password_hash = hash_password(password)
    
    if user["password_hash"] == password_hash:
        # Update last login
        user["last_login"] = datetime.now().isoformat()
        users[email] = user
        save_users(users)
        
        # Return user data without password hash
        return {
            "user_id": user["user_id"],
            "email": user["email"],
            "name": user["name"],
            "created_at": user["created_at"],
            "last_login": user["last_login"]
        }
    
    return None

def register_user(email: str, password: str, name: str) -> Optional[Dict]:
    """
    Register a new user
    Returns user data if successful, None if user already exists
    """
    users = load_users()
    
    if email in users:
        return None  # User already exists
    
    user_id = f"user_{len(users) + 1}_{int(datetime.now().timestamp())}"
    
    users[email] = {
        "user_id": user_id,
        "email": email,
        "password_hash": hash_password(password),
        "name": name,
        "created_at": datetime.now().isoformat(),
        "last_login": None
    }
    
    save_users(users)
    
    return {
        "user_id": user_id,
        "email": email,
        "name": name,
        "created_at": users[email]["created_at"],
        "last_login": None
    }

def get_user_by_id(user_id: str) -> Optional[Dict]:
    """Get user data by user_id"""
    users = load_users()
    
    for email, user in users.items():
        if user["user_id"] == user_id:
            return {
                "user_id": user["user_id"],
                "email": user["email"],
                "name": user["name"],
                "created_at": user["created_at"],
                "last_login": user["last_login"]
            }
    
    return None

if __name__ == "__main__":
    # Initialize database with default users
    print("Initializing user database...")
    users = load_users()
    print(f"✓ Loaded {len(users)} users")
    
    # Test authentication
    print("\nTesting authentication:")
    result = authenticate_user("demo@baymax.ai", "demo123")
    if result:
        print(f"✓ Demo user authenticated: {result['user_id']}")
    else:
        print("✗ Authentication failed")
    
    result = authenticate_user("test@baymax.ai", "test123")
    if result:
        print(f"✓ Test user authenticated: {result['user_id']}")
    else:
        print("✗ Authentication failed")
    
    print(f"\n✓ User database ready at: {USER_DB_PATH}")
