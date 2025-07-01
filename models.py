"""
Data models for FitTrack Pro
Defines user profiles, goals, workouts, and nutrition data structures
"""

from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Union, Any
from dataclasses import dataclass, field
import pandas as pd
import json
import os
from config import Config

@dataclass
class UserProfile:
    """User profile data model"""
    name: str = ""
    age: int = 25
    gender: str = "Male"
    height: float = 170.0  # cm
    weight: float = 70.0   # kg
    body_fat_percentage: Optional[float] = None
    activity_level: str = "moderately_active"
    bmr_formula: str = "mifflin_st_jeor"
    units: str = "metric"
    dietary_restrictions: List[str] = field(default_factory=list)
    favorite_foods: List[str] = field(default_factory=list)
    medical_conditions: List[str] = field(default_factory=list)
    ai_coaching_style: str = "motivational"
    notification_preferences: Dict[str, bool] = field(default_factory=dict)
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def calculate_bmr(self) -> float:
        """Calculate BMR using selected formula"""
        if self.bmr_formula == "mifflin_st_jeor":
            return self._calculate_mifflin_st_jeor()
        elif self.bmr_formula == "harris_benedict":
            return self._calculate_harris_benedict()
        elif self.bmr_formula == "katch_mcardle":
            return self._calculate_katch_mcardle()
        else:
            return self._calculate_mifflin_st_jeor()
    
    def _calculate_mifflin_st_jeor(self) -> float:
        """Mifflin-St Jeor Equation"""
        if self.gender.lower() == 'male':
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age + 5
        else:
            bmr = 10 * self.weight + 6.25 * self.height - 5 * self.age - 161
        return round(bmr)
    
    def _calculate_harris_benedict(self) -> float:
        """Harris-Benedict Equation"""
        if self.gender.lower() == 'male':
            bmr = 88.362 + (13.397 * self.weight) + (4.799 * self.height) - (5.677 * self.age)
        else:
            bmr = 447.593 + (9.247 * self.weight) + (3.098 * self.height) - (4.330 * self.age)
        return round(bmr)
    
    def _calculate_katch_mcardle(self) -> float:
        """Katch-McArdle Equation (requires body fat percentage)"""
        if self.body_fat_percentage is None:
            # Fallback to Mifflin-St Jeor if body fat not available
            return self._calculate_mifflin_st_jeor()
        
        lean_mass = self.weight * (1 - self.body_fat_percentage / 100)
        bmr = 370 + (21.6 * lean_mass)
        return round(bmr)
    
    def calculate_tdee(self) -> float:
        """Calculate Total Daily Energy Expenditure"""
        bmr = self.calculate_bmr()
        multiplier = Config.ACTIVITY_MULTIPLIERS.get(self.activity_level, 1.2)
        return round(bmr * multiplier)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'name': self.name,
            'age': self.age,
            'gender': self.gender,
            'height': self.height,
            'weight': self.weight,
            'body_fat_percentage': self.body_fat_percentage,
            'activity_level': self.activity_level,
            'bmr_formula': self.bmr_formula,
            'units': self.units,
            'dietary_restrictions': self.dietary_restrictions,
            'favorite_foods': self.favorite_foods,
            'medical_conditions': self.medical_conditions,
            'ai_coaching_style': self.ai_coaching_style,
            'notification_preferences': self.notification_preferences,
            'created_date': self.created_date,
            'last_updated': self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'UserProfile':
        """Create UserProfile from dictionary"""
        return cls(**data)

@dataclass
class Goal:
    """Goal tracking data model"""
    goal_type: str  # weight_loss, weight_gain, maintenance, recomposition
    target_value: float
    current_value: float
    start_value: float
    start_date: str
    target_date: str
    timeline_weeks: int
    weekly_change_rate: float  # kg per week
    status: str = "active"  # active, completed, paused, abandoned
    notes: str = ""
    created_date: str = field(default_factory=lambda: datetime.now().isoformat())
    last_updated: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def calculate_progress_percentage(self) -> float:
        """Calculate progress percentage toward goal"""
        if self.goal_type == "weight_loss":
            total_change = self.start_value - self.target_value
            current_change = self.start_value - self.current_value
            if total_change == 0:
                return 100.0
            return min(100.0, max(0.0, (current_change / total_change) * 100))
        elif self.goal_type == "weight_gain":
            total_change = self.target_value - self.start_value
            current_change = self.current_value - self.start_value
            if total_change == 0:
                return 100.0
            return min(100.0, max(0.0, (current_change / total_change) * 100))
        else:
            return 0.0
    
    def calculate_daily_calorie_target(self, tdee: float) -> float:
        """Calculate daily calorie target based on goal"""
        if self.goal_type == "weight_loss":
            # 1 kg = 7700 calories, so weekly deficit = weekly_change_rate * 7700
            weekly_deficit = self.weekly_change_rate * 7700
            daily_deficit = weekly_deficit / 7
            return round(tdee - daily_deficit)
        elif self.goal_type == "weight_gain":
            weekly_surplus = self.weekly_change_rate * 7700
            daily_surplus = weekly_surplus / 7
            return round(tdee + daily_surplus)
        else:
            return round(tdee)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'goal_type': self.goal_type,
            'target_value': self.target_value,
            'current_value': self.current_value,
            'start_value': self.start_value,
            'start_date': self.start_date,
            'target_date': self.target_date,
            'timeline_weeks': self.timeline_weeks,
            'weekly_change_rate': self.weekly_change_rate,
            'status': self.status,
            'notes': self.notes,
            'created_date': self.created_date,
            'last_updated': self.last_updated
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'Goal':
        """Create Goal from dictionary"""
        return cls(**data)

@dataclass
class FoodEntry:
    """Food logging entry data model"""
    date: str
    food_name: str
    calories: float
    protein: float = 0.0
    carbs: float = 0.0
    fat: float = 0.0
    fiber: float = 0.0
    sodium: float = 0.0
    serving_size: str = ""
    meal_type: str = ""  # breakfast, lunch, dinner, snack
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'date': self.date,
            'food_name': self.food_name,
            'calories': self.calories,
            'protein': self.protein,
            'carbs': self.carbs,
            'fat': self.fat,
            'fiber': self.fiber,
            'sodium': self.sodium,
            'serving_size': self.serving_size,
            'meal_type': self.meal_type,
            'notes': self.notes,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'FoodEntry':
        """Create FoodEntry from dictionary"""
        return cls(**data)

@dataclass
class WorkoutEntry:
    """Workout logging entry data model"""
    date: str
    workout_type: str  # cardio, strength, sports, yoga, etc.
    exercise_name: str
    duration_minutes: int
    intensity: str = "moderate"  # light, moderate, vigorous
    calories_burned: Optional[float] = None
    sets: Optional[int] = None
    reps: Optional[int] = None
    weight: Optional[float] = None
    distance: Optional[float] = None
    heart_rate_avg: Optional[int] = None
    heart_rate_max: Optional[int] = None
    notes: str = ""
    timestamp: str = field(default_factory=lambda: datetime.now().isoformat())
    
    def calculate_calories_burned(self, user_weight: float) -> float:
        """Calculate calories burned based on METs and user weight"""
        # METs values for different activities
        mets_values = {
            'walking': {'light': 2.5, 'moderate': 3.5, 'vigorous': 4.5},
            'running': {'light': 6.0, 'moderate': 8.0, 'vigorous': 12.0},
            'cycling': {'light': 4.0, 'moderate': 6.0, 'vigorous': 10.0},
            'swimming': {'light': 4.0, 'moderate': 6.0, 'vigorous': 8.0},
            'weight_training': {'light': 3.0, 'moderate': 4.5, 'vigorous': 6.0},
            'yoga': {'light': 2.0, 'moderate': 3.0, 'vigorous': 4.0},
            'basketball': {'light': 4.0, 'moderate': 6.0, 'vigorous': 8.0},
            'soccer': {'light': 4.0, 'moderate': 7.0, 'vigorous': 10.0},
            'tennis': {'light': 4.0, 'moderate': 6.0, 'vigorous': 8.0},
            'hiking': {'light': 3.0, 'moderate': 5.0, 'vigorous': 7.0}
        }
        
        # Get METs value for the exercise
        exercise_mets = mets_values.get(self.exercise_name.lower(), 
                                      mets_values.get(self.workout_type.lower(), 
                                                     {'moderate': 4.0}))
        mets = exercise_mets.get(self.intensity, 4.0)
        
        # Calculate calories burned: (METs * weight in kg * duration in hours)
        hours = self.duration_minutes / 60
        calories = mets * user_weight * hours
        
        return round(calories)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'date': self.date,
            'workout_type': self.workout_type,
            'exercise_name': self.exercise_name,
            'duration_minutes': self.duration_minutes,
            'intensity': self.intensity,
            'calories_burned': self.calories_burned,
            'sets': self.sets,
            'reps': self.reps,
            'weight': self.weight,
            'distance': self.distance,
            'heart_rate_avg': self.heart_rate_avg,
            'heart_rate_max': self.heart_rate_max,
            'notes': self.notes,
            'timestamp': self.timestamp
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'WorkoutEntry':
        """Create WorkoutEntry from dictionary"""
        return cls(**data)

@dataclass
class NutritionSummary:
    """Daily nutrition summary data model"""
    date: str
    total_calories: float = 0.0
    total_protein: float = 0.0
    total_carbs: float = 0.0
    total_fat: float = 0.0
    total_fiber: float = 0.0
    total_sodium: float = 0.0
    meal_count: int = 0
    water_intake: float = 0.0
    target_calories: float = 0.0
    calories_remaining: float = 0.0
    macro_ratios: Dict[str, float] = field(default_factory=dict)
    
    def calculate_macro_ratios(self) -> Dict[str, float]:
        """Calculate macro ratios as percentages"""
        if self.total_calories > 0:
            protein_cals = self.total_protein * 4
            carbs_cals = self.total_carbs * 4
            fat_cals = self.total_fat * 9
            
            self.macro_ratios = {
                'protein': round((protein_cals / self.total_calories) * 100, 1),
                'carbs': round((carbs_cals / self.total_calories) * 100, 1),
                'fat': round((fat_cals / self.total_calories) * 100, 1)
            }
        return self.macro_ratios
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for storage"""
        return {
            'date': self.date,
            'total_calories': self.total_calories,
            'total_protein': self.total_protein,
            'total_carbs': self.total_carbs,
            'total_fat': self.total_fat,
            'total_fiber': self.total_fiber,
            'total_sodium': self.total_sodium,
            'meal_count': self.meal_count,
            'water_intake': self.water_intake,
            'target_calories': self.target_calories,
            'calories_remaining': self.calories_remaining,
            'macro_ratios': self.macro_ratios
        }
    
    @classmethod
    def from_dict(cls, data: Dict[str, Any]) -> 'NutritionSummary':
        """Create NutritionSummary from dictionary"""
        return cls(**data) 