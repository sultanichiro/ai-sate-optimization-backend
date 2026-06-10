"""
Configuration settings for the application.
Loads environment variables and provides centralized configuration.
"""

import os
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()


class Settings:
    """Application settings loaded from environment variables."""
    
    # Database
    DATABASE_URL: str = os.getenv(
        "DATABASE_URL", 
        "mysql+pymysql://root:password@localhost:3306/sate_keliling"
    )
    
    # JWT Authentication
    SECRET_KEY: str = os.getenv("SECRET_KEY", "your-super-secret-key")
    ALGORITHM: str = os.getenv("ALGORITHM", "HS256")
    ACCESS_TOKEN_EXPIRE_MINUTES: int = int(
        os.getenv("ACCESS_TOKEN_EXPIRE_MINUTES", "1440")
    )
    
    # Q-Learning Parameters
    ALPHA: float = float(os.getenv("ALPHA", "0.1"))  # Learning rate
    GAMMA: float = float(os.getenv("GAMMA", "0.9"))  # Discount factor
    EPSILON: float = float(os.getenv("EPSILON", "0.1"))  # Exploration rate
    MAX_EPISODES: int = int(os.getenv("MAX_EPISODES", "100"))
    WAKTU_OPERASIONAL: int = int(os.getenv("WAKTU_OPERASIONAL", "240"))  # 4 hours in minutes


settings = Settings()
