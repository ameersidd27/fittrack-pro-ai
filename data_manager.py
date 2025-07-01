"""
Data management module for FitTrack Pro
Handles all data operations including CRUD operations for user data
"""

import pandas as pd
import json
import os
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Union
import streamlit as st
from config import Config
from models import UserProfile, Goal, FoodEntry, WorkoutEntry, NutritionSummary

class DataManager:
    """Central data management class"""
    
    def __init__(self):
        self.ensure_data_files()
    
    def ensure_data_files(self):
        """Ensure all required data files exist"""
        files_to_create = {
            Config.USER_PROFILE_FILE: {
                'columns': ['name', 'age', 'gender', 'height', 'weight', 'body_fat_percentage', 
                           'activity_level', 'bmr_formula', 'units', 'dietary_restrictions', 
                           'favorite_foods', 'medical_conditions', 'ai_coaching_style', 
                           'notification_preferences', 'created_date', 'last_updated']
            },
            Config.FOOD_LOG_FILE: {
                'columns': ['date', 'food_name', 'calories', 'protein', 'carbs', 'fat', 
                           'fiber', 'sodium', 'serving_size', 'meal_type', 'notes', 'timestamp']
            },
            Config.WORKOUT_LOG_FILE: {
                'columns': ['date', 'workout_type', 'exercise_name', 'duration_minutes', 
                           'intensity', 'calories_burned', 'sets', 'reps', 'weight', 
                           'distance', 'heart_rate_avg', 'heart_rate_max', 'notes', 'timestamp']
            },
            Config.GOAL_TRACKING_FILE: {
                'columns': ['goal_type', 'target_value', 'current_value', 'start_value', 
                           'start_date', 'target_date', 'timeline_weeks', 'weekly_change_rate', 
                           'status', 'notes', 'created_date', 'last_updated']
            }
        }
        
        for file_path, config in files_to_create.items():
            if not os.path.exists(file_path):
                df = pd.DataFrame(columns=config['columns'])
                df.to_csv(file_path, index=False)
                print(f"Created {file_path}")
    
    # User Profile Management
    def load_user_profile(self) -> Optional[UserProfile]:
        """Load user profile from CSV"""
        try:
            if os.path.exists(Config.USER_PROFILE_FILE):
                df = pd.read_csv(Config.USER_PROFILE_FILE)
                if not df.empty:
                    # Convert JSON strings back to lists/dicts
                    row = df.iloc[0]
                    profile_data = row.to_dict()
                    
                    # Handle JSON fields
                    for field in ['dietary_restrictions', 'favorite_foods', 'medical_conditions']:
                        if field in profile_data and pd.notna(profile_data[field]):
                            try:
                                profile_data[field] = json.loads(profile_data[field])
                            except:
                                profile_data[field] = []
                        else:
                            profile_data[field] = []
                    
                    if 'notification_preferences' in profile_data and pd.notna(profile_data['notification_preferences']):
                        try:
                            profile_data['notification_preferences'] = json.loads(profile_data['notification_preferences'])
                        except:
                            profile_data['notification_preferences'] = {}
                    else:
                        profile_data['notification_preferences'] = {}
                    
                    return UserProfile.from_dict(profile_data)
        except Exception as e:
            st.error(f"Error loading user profile: {e}")
        return None
    
    def save_user_profile(self, profile: UserProfile) -> bool:
        """Save user profile to CSV"""
        try:
            profile_data = profile.to_dict()
            
            # Convert lists/dicts to JSON strings for CSV storage
            for field in ['dietary_restrictions', 'favorite_foods', 'medical_conditions']:
                profile_data[field] = json.dumps(profile_data[field])
            
            profile_data['notification_preferences'] = json.dumps(profile_data['notification_preferences'])
            
            df = pd.DataFrame([profile_data])
            df.to_csv(Config.USER_PROFILE_FILE, index=False)
            return True
        except Exception as e:
            st.error(f"Error saving user profile: {e}")
            return False
    
    # Food Log Management
    def load_food_logs(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Load food logs with optional date filtering"""
        try:
            if os.path.exists(Config.FOOD_LOG_FILE):
                df = pd.read_csv(Config.FOOD_LOG_FILE)
                if not df.empty:
                    # Convert date column to datetime for filtering
                    df['date'] = pd.to_datetime(df['date'], errors='coerce')
                    
                    if start_date:
                        df = df[df['date'] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df['date'] <= pd.to_datetime(end_date)]
                    
                    return df
        except Exception as e:
            st.error(f"Error loading food logs: {e}")
        
        return pd.DataFrame()
    
    def save_food_entry(self, food_entry: FoodEntry) -> bool:
        """Save a new food entry"""
        try:
            df = self.load_food_logs()
            new_row = food_entry.to_dict()
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(Config.FOOD_LOG_FILE, index=False)
            return True
        except Exception as e:
            st.error(f"Error saving food entry: {e}")
            return False
    
    def get_today_food_summary(self) -> NutritionSummary:
        """Get nutrition summary for today"""
        today = date.today().strftime('%Y-%m-%d')
        df = self.load_food_logs(today, today)
        
        if df.empty:
            return NutritionSummary(date=today)
        
        summary = NutritionSummary(
            date=today,
            total_calories=df['calories'].sum(),
            total_protein=df['protein'].sum(),
            total_carbs=df['carbs'].sum(),
            total_fat=df['fat'].sum(),
            total_fiber=df['fiber'].sum(),
            total_sodium=df['sodium'].sum(),
            meal_count=len(df)
        )
        
        summary.calculate_macro_ratios()
        return summary
    
    # Workout Log Management
    def load_workout_logs(self, start_date: Optional[str] = None, end_date: Optional[str] = None) -> pd.DataFrame:
        """Load workout logs with optional date filtering"""
        try:
            if os.path.exists(Config.WORKOUT_LOG_FILE):
                df = pd.read_csv(Config.WORKOUT_LOG_FILE)
                if not df.empty:
                    df['date'] = pd.to_datetime(df['date'])
                    
                    if start_date:
                        df = df[df['date'] >= pd.to_datetime(start_date)]
                    if end_date:
                        df = df[df['date'] <= pd.to_datetime(end_date)]
                    
                    return df
        except Exception as e:
            st.error(f"Error loading workout logs: {e}")
        
        return pd.DataFrame()
    
    def save_workout_entry(self, workout_entry: WorkoutEntry) -> bool:
        """Save a new workout entry"""
        try:
            df = self.load_workout_logs()
            new_row = workout_entry.to_dict()
            df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
            df.to_csv(Config.WORKOUT_LOG_FILE, index=False)
            return True
        except Exception as e:
            st.error(f"Error saving workout entry: {e}")
            return False
    
    def get_today_workout_summary(self) -> Dict[str, Any]:
        """Get workout summary for today"""
        today = date.today().strftime('%Y-%m-%d')
        df = self.load_workout_logs(today, today)
        
        if df.empty:
            return {
                'total_calories_burned': 0,
                'total_duration': 0,
                'workout_count': 0,
                'workouts': []
            }
        
        return {
            'total_calories_burned': df['calories_burned'].sum(),
            'total_duration': df['duration_minutes'].sum(),
            'workout_count': len(df),
            'workouts': df.to_dict('records')
        }
    
    # Goal Management
    def load_goals(self) -> List[Goal]:
        """Load all goals"""
        try:
            if os.path.exists(Config.GOAL_TRACKING_FILE):
                df = pd.read_csv(Config.GOAL_TRACKING_FILE)
                goals = []
                for _, row in df.iterrows():
                    goal_data = row.to_dict()
                    goals.append(Goal.from_dict(goal_data))
                return goals
        except Exception as e:
            st.error(f"Error loading goals: {e}")
        return []
    
    def save_goal(self, goal: Goal) -> bool:
        """Save a new goal"""
        try:
            goals = self.load_goals()
            goals.append(goal)
            
            goal_data = [g.to_dict() for g in goals]
            df = pd.DataFrame(goal_data)
            df.to_csv(Config.GOAL_TRACKING_FILE, index=False)
            return True
        except Exception as e:
            st.error(f"Error saving goal: {e}")
            return False
    
    def update_goal_progress(self, goal_id: int, current_value: float) -> bool:
        """Update goal progress"""
        try:
            goals = self.load_goals()
            if 0 <= goal_id < len(goals):
                goals[goal_id].current_value = current_value
                goals[goal_id].last_updated = datetime.now().isoformat()
                
                goal_data = [g.to_dict() for g in goals]
                df = pd.DataFrame(goal_data)
                df.to_csv(Config.GOAL_TRACKING_FILE, index=False)
                return True
        except Exception as e:
            st.error(f"Error updating goal progress: {e}")
        return False
    
    # AI Data Management
    def load_ai_conversations(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load AI conversation history"""
        try:
            if os.path.exists(Config.AI_CONVERSATION_FILE):
                with open(Config.AI_CONVERSATION_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Error loading AI conversations: {e}")
        return {}
    
    def save_ai_conversation(self, user_id: str, message: str, response: str, context: Optional[Dict] = None):
        """Save AI conversation"""
        try:
            conversations = self.load_ai_conversations()
            
            if user_id not in conversations:
                conversations[user_id] = []
            
            conversations[user_id].append({
                'timestamp': datetime.now().isoformat(),
                'message': message,
                'response': response,
                'context': context or {}
            })
            
            # Keep only last 50 conversations
            conversations[user_id] = conversations[user_id][-50:]
            
            with open(Config.AI_CONVERSATION_FILE, 'w') as f:
                json.dump(conversations, f, indent=2)
        except Exception as e:
            st.error(f"Error saving AI conversation: {e}")
    
    def load_ai_usage(self) -> Dict[str, Any]:
        """Load AI usage statistics"""
        try:
            if os.path.exists(Config.AI_USAGE_FILE):
                with open(Config.AI_USAGE_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Error loading AI usage: {e}")
        return {'requests': [], 'total_tokens': 0, 'total_cost': 0.0}
    
    def save_ai_usage(self, tokens_used: int, cost: float):
        """Save AI usage data"""
        try:
            usage = self.load_ai_usage()
            
            usage['requests'].append({
                'timestamp': datetime.now().isoformat(),
                'tokens': tokens_used,
                'cost': cost
            })
            
            usage['total_tokens'] += tokens_used
            usage['total_cost'] += cost
            
            # Keep only last 100 requests
            usage['requests'] = usage['requests'][-100:]
            
            with open(Config.AI_USAGE_FILE, 'w') as f:
                json.dump(usage, f, indent=2)
        except Exception as e:
            st.error(f"Error saving AI usage: {e}")
    
    # Analytics and Reporting
    def get_weekly_nutrition_summary(self) -> Dict[str, Any]:
        """Get weekly nutrition summary"""
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        
        df = self.load_food_logs(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            return {
                'avg_daily_calories': 0,
                'total_days': 0,
                'daily_patterns': {},
                'macro_averages': {'protein': 0, 'carbs': 0, 'fat': 0}
            }
        
        # Group by date
        daily_summary = df.groupby('date').agg({
            'calories': 'sum',
            'protein': 'sum',
            'carbs': 'sum',
            'fat': 'sum'
        }).reset_index()
        
        return {
            'avg_daily_calories': daily_summary['calories'].mean(),
            'total_days': len(daily_summary),
            'daily_patterns': daily_summary.set_index('date')['calories'].to_dict(),
            'macro_averages': {
                'protein': daily_summary['protein'].mean(),
                'carbs': daily_summary['carbs'].mean(),
                'fat': daily_summary['fat'].mean()
            }
        }
    
    def get_weekly_workout_summary(self) -> Dict[str, Any]:
        """Get weekly workout summary"""
        end_date = date.today()
        start_date = end_date - timedelta(days=7)
        
        df = self.load_workout_logs(start_date.strftime('%Y-%m-%d'), end_date.strftime('%Y-%m-%d'))
        
        if df.empty:
            return {
                'total_calories_burned': 0,
                'total_duration': 0,
                'workout_days': 0,
                'avg_daily_calories_burned': 0
            }
        
        return {
            'total_calories_burned': df['calories_burned'].sum(),
            'total_duration': df['duration_minutes'].sum(),
            'workout_days': len(df['date'].unique()),
            'avg_daily_calories_burned': df['calories_burned'].sum() / 7
        }
    
    # Data Export and Backup
    def export_user_data(self) -> Dict[str, Any]:
        """Export all user data for backup"""
        try:
            return {
                'user_profile': self.load_user_profile().to_dict() if self.load_user_profile() else None,
                'food_logs': self.load_food_logs().to_dict('records'),
                'workout_logs': self.load_workout_logs().to_dict('records'),
                'goals': [g.to_dict() for g in self.load_goals()],
                'ai_conversations': self.load_ai_conversations(),
                'ai_usage': self.load_ai_usage(),
                'export_date': datetime.now().isoformat()
            }
        except Exception as e:
            st.error(f"Error exporting user data: {e}")
            return {}
    
    def import_user_data(self, data: Dict[str, Any]) -> bool:
        """Import user data from backup"""
        try:
            # Import user profile
            if data.get('user_profile'):
                profile = UserProfile.from_dict(data['user_profile'])
                self.save_user_profile(profile)
            
            # Import food logs
            if data.get('food_logs'):
                df = pd.DataFrame(data['food_logs'])
                df.to_csv(Config.FOOD_LOG_FILE, index=False)
            
            # Import workout logs
            if data.get('workout_logs'):
                df = pd.DataFrame(data['workout_logs'])
                df.to_csv(Config.WORKOUT_LOG_FILE, index=False)
            
            # Import goals
            if data.get('goals'):
                goal_data = [Goal.from_dict(g) for g in data['goals']]
                goal_dicts = [g.to_dict() for g in goal_data]
                df = pd.DataFrame(goal_dicts)
                df.to_csv(Config.GOAL_TRACKING_FILE, index=False)
            
            # Import AI data
            if data.get('ai_conversations'):
                with open(Config.AI_CONVERSATION_FILE, 'w') as f:
                    json.dump(data['ai_conversations'], f, indent=2)
            
            if data.get('ai_usage'):
                with open(Config.AI_USAGE_FILE, 'w') as f:
                    json.dump(data['ai_usage'], f, indent=2)
            
            return True
        except Exception as e:
            st.error(f"Error importing user data: {e}")
            return False

# Global data manager instance
data_manager = DataManager() 