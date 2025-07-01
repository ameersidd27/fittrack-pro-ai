"""
Configuration module for FitTrack Pro
Handles environment variables, API keys, and app settings
"""

import os
from typing import Optional
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

class Config:
    """Application configuration class"""
    
    # OpenAI Configuration
    OPENAI_API_KEY: Optional[str] = os.getenv('OPENAI_API_KEY')
    AI_MODEL: str = os.getenv('AI_MODEL', 'gpt-4o-mini')
    MAX_AI_REQUESTS_PER_MINUTE: int = int(os.getenv('MAX_AI_REQUESTS_PER_MINUTE', '3'))
    
    # USDA FoodData Central API Configuration
    USDA_API_KEY: str = os.getenv('USDA_API_KEY', 'DEMO_KEY')
    USDA_BASE_URL: str = "https://api.nal.usda.gov/fdc/v1"
    
    # App Configuration
    APP_ENVIRONMENT: str = os.getenv('APP_ENVIRONMENT', 'development')
    DEBUG_MODE: bool = os.getenv('DEBUG_MODE', 'True').lower() == 'true'
    
    # Cache Configuration
    CACHE_DURATION_HOURS: int = int(os.getenv('CACHE_DURATION_HOURS', '24'))
    MAX_CACHE_SIZE_MB: int = int(os.getenv('MAX_CACHE_SIZE_MB', '100'))
    
    # File paths
    DATA_DIR: str = "data"
    FOOD_LOG_FILE: str = os.path.join(DATA_DIR, "food_logs.csv")
    USER_PROFILE_FILE: str = os.path.join(DATA_DIR, "user_profile.csv")
    WORKOUT_LOG_FILE: str = os.path.join(DATA_DIR, "workout_logs.csv")
    GOAL_TRACKING_FILE: str = os.path.join(DATA_DIR, "goal_tracking.csv")
    FOOD_CACHE_FILE: str = os.path.join(DATA_DIR, "food_cache.json")
    AI_CONVERSATION_FILE: str = os.path.join(DATA_DIR, "ai_conversations.json")
    AI_USAGE_FILE: str = os.path.join(DATA_DIR, "ai_usage.json")
    AI_INSIGHTS_FILE: str = os.path.join(DATA_DIR, "ai_insights.json")
    
    # BMR Calculation Constants
    BMR_FORMULAS = {
        'mifflin_st_jeor': 'Mifflin-St Jeor',
        'harris_benedict': 'Harris-Benedict',
        'katch_mcardle': 'Katch-McArdle'
    }
    
    # Activity Level Multipliers
    ACTIVITY_MULTIPLIERS = {
        'sedentary': 1.2,           # Little or no exercise
        'lightly_active': 1.375,    # Light exercise 1-3 days/week
        'moderately_active': 1.55,  # Moderate exercise 3-5 days/week
        'very_active': 1.725,       # Hard exercise 6-7 days/week
        'extremely_active': 1.9     # Very hard exercise, physical job
    }
    
    # Goal Types
    GOAL_TYPES = {
        'weight_loss': 'Weight Loss',
        'weight_gain': 'Weight Gain',
        'maintenance': 'Maintenance',
        'recomposition': 'Body Recomposition'
    }
    
    # Measurement Units
    UNITS = {
        'metric': 'Metric (kg, cm)',
        'imperial': 'Imperial (lbs, inches)'
    }
    
    @classmethod
    def validate_config(cls) -> dict:
        """Validate configuration and return status"""
        status = {
            'openai_configured': bool(cls.OPENAI_API_KEY),
            'usda_configured': cls.USDA_API_KEY != 'DEMO_KEY',
            'data_dir_exists': os.path.exists(cls.DATA_DIR),
            'environment': cls.APP_ENVIRONMENT,
            'debug_mode': cls.DEBUG_MODE
        }
        return status
    
    @classmethod
    def ensure_data_directory(cls):
        """Ensure data directory exists"""
        if not os.path.exists(cls.DATA_DIR):
            os.makedirs(cls.DATA_DIR)
            print(f"Created data directory: {cls.DATA_DIR}")

# Initialize data directory
Config.ensure_data_directory() 