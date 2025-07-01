"""
Workout System module for FitTrack Pro
Handles workout tracking, calorie calculations, and exercise database
"""

import json
import pandas as pd
from datetime import datetime, date
from typing import Dict, List, Optional, Any
import streamlit as st
from config import Config
from data_manager import data_manager
from models import WorkoutEntry

# METs (Metabolic Equivalent of Task) database for calorie calculations
# Values represent calories burned per kg of body weight per hour
METS_DATABASE = {
    # Cardio exercises
    'walking': {
        'light': 2.5,      # Strolling, very slow
        'moderate': 3.5,   # Walking, normal pace
        'fast': 4.5,       # Walking, brisk pace
        'very_fast': 6.0   # Walking, very brisk pace
    },
    'running': {
        'light': 6.0,      # Jogging, slow
        'moderate': 8.0,   # Running, moderate pace
        'fast': 12.0,      # Running, fast pace
        'very_fast': 16.0  # Running, very fast pace
    },
    'cycling': {
        'light': 4.0,      # Cycling, leisurely
        'moderate': 6.0,   # Cycling, moderate effort
        'fast': 10.0,      # Cycling, fast pace
        'very_fast': 14.0  # Cycling, racing
    },
    'swimming': {
        'light': 4.0,      # Swimming, leisurely
        'moderate': 6.0,   # Swimming, moderate effort
        'fast': 8.0,       # Swimming, fast pace
        'very_fast': 12.0  # Swimming, competitive
    },
    
    # Strength training
    'weight_training': {
        'light': 3.0,      # Light weights, high reps
        'moderate': 4.5,   # Moderate weights, moderate reps
        'heavy': 6.0,      # Heavy weights, low reps
        'very_heavy': 8.0  # Very heavy weights, low reps
    },
    'bodyweight_exercises': {
        'light': 2.5,      # Easy bodyweight exercises
        'moderate': 4.0,   # Moderate bodyweight exercises
        'intense': 6.0,    # Intense bodyweight exercises
        'very_intense': 8.0 # Very intense bodyweight exercises
    },
    
    # Sports and activities
    'basketball': {
        'light': 4.0,      # Shooting hoops
        'moderate': 6.0,   # Recreational game
        'intense': 8.0,    # Competitive game
        'very_intense': 12.0 # Professional level
    },
    'tennis': {
        'light': 4.0,      # Recreational doubles
        'moderate': 6.0,   # Recreational singles
        'intense': 8.0,    # Competitive singles
        'very_intense': 10.0 # Professional level
    },
    'soccer': {
        'light': 5.0,      # Recreational
        'moderate': 7.0,   # Moderate effort
        'intense': 10.0,   # Competitive
        'very_intense': 12.0 # Professional
    },
    
    # Yoga and flexibility
    'yoga': {
        'light': 2.0,      # Gentle yoga
        'moderate': 3.0,   # Moderate yoga
        'intense': 4.5,    # Power yoga
        'very_intense': 6.0 # Hot yoga
    },
    'pilates': {
        'light': 2.5,      # Beginner pilates
        'moderate': 3.5,   # Intermediate pilates
        'intense': 4.5,    # Advanced pilates
        'very_intense': 5.5 # Very advanced pilates
    },
    
    # Other activities
    'dancing': {
        'light': 3.0,      # Slow dancing
        'moderate': 5.0,   # Moderate dancing
        'intense': 7.0,    # Fast dancing
        'very_intense': 9.0 # Very fast dancing
    },
    'hiking': {
        'light': 4.0,      # Easy trail
        'moderate': 6.0,   # Moderate trail
        'intense': 8.0,    # Difficult trail
        'very_intense': 10.0 # Very difficult trail
    },
    'rowing': {
        'light': 4.0,      # Light rowing
        'moderate': 6.0,   # Moderate rowing
        'intense': 8.0,    # Intense rowing
        'very_intense': 12.0 # Very intense rowing
    }
}

# Exercise categories for organization
EXERCISE_CATEGORIES = {
    'cardio': ['walking', 'running', 'cycling', 'swimming', 'rowing'],
    'strength': ['weight_training', 'bodyweight_exercises'],
    'sports': ['basketball', 'tennis', 'soccer'],
    'flexibility': ['yoga', 'pilates'],
    'other': ['dancing', 'hiking']
}

class WorkoutSystem:
    """Comprehensive workout tracking and analysis system"""
    
    def __init__(self):
        self.mets_db = METS_DATABASE
        self.categories = EXERCISE_CATEGORIES
    
    def calculate_calories_burned(self, exercise: str, intensity: str, duration_minutes: float, 
                                user_weight_kg: float) -> float:
        """Calculate calories burned using METs formula"""
        try:
            # Get METs value for exercise and intensity
            if exercise in self.mets_db:
                if intensity in self.mets_db[exercise]:
                    mets_value = self.mets_db[exercise][intensity]
                else:
                    # Use moderate intensity as default
                    mets_value = self.mets_db[exercise]['moderate']
            else:
                # Use generic moderate exercise METs
                mets_value = 4.0
            
            # Calculate calories burned
            # Formula: Calories = METs × weight (kg) × duration (hours)
            duration_hours = duration_minutes / 60
            calories_burned = mets_value * user_weight_kg * duration_hours
            
            return round(calories_burned)
            
        except Exception as e:
            st.error(f"Error calculating calories burned: {e}")
            return 0.0
    
    def get_available_exercises(self, category: Optional[str] = None) -> List[str]:
        """Get list of available exercises, optionally filtered by category"""
        if category and category in self.categories:
            return self.categories[category]
        elif category == 'all':
            return list(self.mets_db.keys())
        else:
            return list(self.mets_db.keys())
    
    def get_exercise_categories(self) -> List[str]:
        """Get list of exercise categories"""
        return list(self.categories.keys())
    
    def get_intensity_levels(self, exercise: str) -> List[str]:
        """Get available intensity levels for an exercise"""
        if exercise in self.mets_db:
            return list(self.mets_db[exercise].keys())
        else:
            return ['light', 'moderate', 'intense', 'very_intense']
    
    def log_workout(self, workout_type: str, exercise_name: str, duration_minutes: int,
                   intensity: str, sets: Optional[int] = None, reps: Optional[int] = None,
                   weight: Optional[float] = None, distance: Optional[float] = None,
                   heart_rate_avg: Optional[int] = None, heart_rate_max: Optional[int] = None,
                   notes: str = "") -> bool:
        """Log a new workout entry"""
        try:
            # Get user weight for calorie calculation
            profile = data_manager.load_user_profile()
            user_weight = profile.weight if profile else 70.0  # Default weight
            
            # Calculate calories burned
            calories_burned = self.calculate_calories_burned(
                exercise_name, intensity, duration_minutes, user_weight
            )
            
            # Create workout entry
            workout_entry = WorkoutEntry(
                date=date.today().strftime('%Y-%m-%d'),
                workout_type=workout_type,
                exercise_name=exercise_name,
                duration_minutes=duration_minutes,
                intensity=intensity,
                calories_burned=calories_burned,
                sets=sets,
                reps=reps,
                weight=weight,
                distance=distance,
                heart_rate_avg=heart_rate_avg,
                heart_rate_max=heart_rate_max,
                notes=notes
            )
            
            # Save to database
            return data_manager.save_workout_entry(workout_entry)
            
        except Exception as e:
            st.error(f"Error logging workout: {e}")
            return False
    
    def get_workout_summary(self, start_date: Optional[str] = None, 
                          end_date: Optional[str] = None) -> Dict[str, Any]:
        """Get comprehensive workout summary for a date range"""
        try:
            df = data_manager.load_workout_logs(start_date, end_date)
            
            if df.empty:
                return {
                    'total_workouts': 0,
                    'total_calories_burned': 0,
                    'total_duration': 0,
                    'avg_duration': 0,
                    'workout_types': {},
                    'exercises': {},
                    'intensity_distribution': {},
                    'weekly_pattern': {}
                }
            
            # Calculate summary statistics
            total_workouts = len(df)
            total_calories_burned = df['calories_burned'].sum()
            total_duration = df['duration_minutes'].sum()
            avg_duration = df['duration_minutes'].mean()
            
            # Workout type distribution
            workout_types = df['workout_type'].value_counts().to_dict()
            
            # Exercise distribution
            exercises = df['exercise_name'].value_counts().to_dict()
            
            # Intensity distribution
            intensity_distribution = df['intensity'].value_counts().to_dict()
            
            # Weekly pattern (if date range is provided)
            weekly_pattern = {}
            if start_date and end_date:
                df['date'] = pd.to_datetime(df['date'])
                df['day_of_week'] = df['date'].dt.day_name()
                weekly_pattern = df.groupby('day_of_week').agg({
                    'calories_burned': 'sum',
                    'duration_minutes': 'sum',
                    'exercise_name': 'count'
                }).to_dict('index')
            
            return {
                'total_workouts': total_workouts,
                'total_calories_burned': total_calories_burned,
                'total_duration': total_duration,
                'avg_duration': avg_duration,
                'workout_types': workout_types,
                'exercises': exercises,
                'intensity_distribution': intensity_distribution,
                'weekly_pattern': weekly_pattern
            }
            
        except Exception as e:
            st.error(f"Error getting workout summary: {e}")
            return {}
    
    def get_workout_recommendations(self, user_profile) -> List[Dict[str, Any]]:
        """Get personalized workout recommendations based on user profile and goals"""
        recommendations = []
        
        try:
            goals = data_manager.load_goals()
            recent_workouts = data_manager.load_workout_logs()
            
            # Get user's fitness level and goals
            fitness_level = self._assess_fitness_level(user_profile, recent_workouts)
            primary_goal = self._get_primary_goal(goals)
            
            # Generate recommendations based on goals and fitness level
            if primary_goal == 'weight_loss':
                recommendations.extend(self._get_weight_loss_workouts(fitness_level))
            elif primary_goal == 'muscle_gain':
                recommendations.extend(self._get_muscle_gain_workouts(fitness_level))
            elif primary_goal == 'endurance':
                recommendations.extend(self._get_endurance_workouts(fitness_level))
            else:
                recommendations.extend(self._get_general_fitness_workouts(fitness_level))
            
            # Add variety recommendations
            recommendations.extend(self._get_variety_recommendations(recent_workouts))
            
        except Exception as e:
            st.error(f"Error generating workout recommendations: {e}")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def _assess_fitness_level(self, profile, recent_workouts) -> str:
        """Assess user's fitness level based on profile and workout history"""
        if recent_workouts.empty:
            return 'beginner'
        
        # Analyze workout frequency and intensity
        avg_duration = recent_workouts['duration_minutes'].mean()
        avg_intensity = recent_workouts['intensity'].mode().iloc[0] if len(recent_workouts) > 0 else 'moderate'
        
        if avg_duration > 60 and avg_intensity in ['intense', 'very_intense']:
            return 'advanced'
        elif avg_duration > 30 and avg_intensity in ['moderate', 'intense']:
            return 'intermediate'
        else:
            return 'beginner'
    
    def _get_primary_goal(self, goals) -> str:
        """Determine user's primary fitness goal"""
        if not goals:
            return 'general_fitness'
        
        # Analyze goals to determine primary focus
        goal_types = [g.goal_type for g in goals]
        
        if 'weight_loss' in goal_types:
            return 'weight_loss'
        elif 'muscle_gain' in goal_types:
            return 'muscle_gain'
        elif 'endurance' in goal_types:
            return 'endurance'
        else:
            return 'general_fitness'
    
    def _get_weight_loss_workouts(self, fitness_level: str) -> List[Dict[str, Any]]:
        """Get workout recommendations for weight loss"""
        recommendations = []
        
        if fitness_level == 'beginner':
            recommendations.extend([
                {
                    'exercise': 'walking',
                    'intensity': 'moderate',
                    'duration': 30,
                    'frequency': '5-6 days/week',
                    'reason': 'Great for beginners, burns calories, and builds endurance'
                },
                {
                    'exercise': 'bodyweight_exercises',
                    'intensity': 'light',
                    'duration': 20,
                    'frequency': '3-4 days/week',
                    'reason': 'Builds muscle and increases metabolism'
                }
            ])
        elif fitness_level == 'intermediate':
            recommendations.extend([
                {
                    'exercise': 'running',
                    'intensity': 'moderate',
                    'duration': 45,
                    'frequency': '4-5 days/week',
                    'reason': 'High calorie burn and cardiovascular benefits'
                },
                {
                    'exercise': 'weight_training',
                    'intensity': 'moderate',
                    'duration': 40,
                    'frequency': '3 days/week',
                    'reason': 'Builds muscle to increase resting metabolism'
                }
            ])
        else:  # advanced
            recommendations.extend([
                {
                    'exercise': 'running',
                    'intensity': 'intense',
                    'duration': 60,
                    'frequency': '4-5 days/week',
                    'reason': 'Maximum calorie burn and endurance building'
                },
                {
                    'exercise': 'weight_training',
                    'intensity': 'heavy',
                    'duration': 50,
                    'frequency': '4 days/week',
                    'reason': 'Builds lean muscle mass and increases metabolism'
                }
            ])
        
        return recommendations
    
    def _get_muscle_gain_workouts(self, fitness_level: str) -> List[Dict[str, Any]]:
        """Get workout recommendations for muscle gain"""
        recommendations = []
        
        if fitness_level == 'beginner':
            recommendations.extend([
                {
                    'exercise': 'bodyweight_exercises',
                    'intensity': 'moderate',
                    'duration': 30,
                    'frequency': '3-4 days/week',
                    'reason': 'Builds foundational strength and muscle'
                },
                {
                    'exercise': 'weight_training',
                    'intensity': 'light',
                    'duration': 25,
                    'frequency': '3 days/week',
                    'reason': 'Introduces resistance training safely'
                }
            ])
        elif fitness_level == 'intermediate':
            recommendations.extend([
                {
                    'exercise': 'weight_training',
                    'intensity': 'moderate',
                    'duration': 45,
                    'frequency': '4 days/week',
                    'reason': 'Progressive overload for muscle growth'
                },
                {
                    'exercise': 'bodyweight_exercises',
                    'intensity': 'intense',
                    'duration': 30,
                    'frequency': '2-3 days/week',
                    'reason': 'Functional strength and muscle endurance'
                }
            ])
        else:  # advanced
            recommendations.extend([
                {
                    'exercise': 'weight_training',
                    'intensity': 'heavy',
                    'duration': 60,
                    'frequency': '5 days/week',
                    'reason': 'Maximum muscle growth and strength'
                },
                {
                    'exercise': 'bodyweight_exercises',
                    'intensity': 'very_intense',
                    'duration': 40,
                    'frequency': '2 days/week',
                    'reason': 'Advanced functional strength'
                }
            ])
        
        return recommendations
    
    def _get_endurance_workouts(self, fitness_level: str) -> List[Dict[str, Any]]:
        """Get workout recommendations for endurance"""
        recommendations = []
        
        if fitness_level == 'beginner':
            recommendations.extend([
                {
                    'exercise': 'walking',
                    'intensity': 'moderate',
                    'duration': 45,
                    'frequency': '5-6 days/week',
                    'reason': 'Builds cardiovascular endurance gradually'
                },
                {
                    'exercise': 'cycling',
                    'intensity': 'light',
                    'duration': 30,
                    'frequency': '3-4 days/week',
                    'reason': 'Low-impact cardio for endurance building'
                }
            ])
        elif fitness_level == 'intermediate':
            recommendations.extend([
                {
                    'exercise': 'running',
                    'intensity': 'moderate',
                    'duration': 60,
                    'frequency': '4-5 days/week',
                    'reason': 'Builds running endurance and stamina'
                },
                {
                    'exercise': 'cycling',
                    'intensity': 'moderate',
                    'duration': 45,
                    'frequency': '3-4 days/week',
                    'reason': 'Cross-training for endurance variety'
                }
            ])
        else:  # advanced
            recommendations.extend([
                {
                    'exercise': 'running',
                    'intensity': 'intense',
                    'duration': 90,
                    'frequency': '5-6 days/week',
                    'reason': 'Advanced endurance training'
                },
                {
                    'exercise': 'cycling',
                    'intensity': 'fast',
                    'duration': 60,
                    'frequency': '3-4 days/week',
                    'reason': 'High-intensity endurance intervals'
                }
            ])
        
        return recommendations
    
    def _get_general_fitness_workouts(self, fitness_level: str) -> List[Dict[str, Any]]:
        """Get general fitness workout recommendations"""
        recommendations = []
        
        if fitness_level == 'beginner':
            recommendations.extend([
                {
                    'exercise': 'walking',
                    'intensity': 'moderate',
                    'duration': 30,
                    'frequency': '5 days/week',
                    'reason': 'Great starting point for overall fitness'
                },
                {
                    'exercise': 'yoga',
                    'intensity': 'light',
                    'duration': 20,
                    'frequency': '3 days/week',
                    'reason': 'Improves flexibility and mindfulness'
                }
            ])
        elif fitness_level == 'intermediate':
            recommendations.extend([
                {
                    'exercise': 'running',
                    'intensity': 'moderate',
                    'duration': 40,
                    'frequency': '4 days/week',
                    'reason': 'Cardiovascular fitness and endurance'
                },
                {
                    'exercise': 'weight_training',
                    'intensity': 'moderate',
                    'duration': 35,
                    'frequency': '3 days/week',
                    'reason': 'Strength and muscle tone'
                }
            ])
        else:  # advanced
            recommendations.extend([
                {
                    'exercise': 'running',
                    'intensity': 'intense',
                    'duration': 50,
                    'frequency': '4-5 days/week',
                    'reason': 'Advanced cardiovascular fitness'
                },
                {
                    'exercise': 'weight_training',
                    'intensity': 'heavy',
                    'duration': 45,
                    'frequency': '4 days/week',
                    'reason': 'Advanced strength and muscle development'
                }
            ])
        
        return recommendations
    
    def _get_variety_recommendations(self, recent_workouts) -> List[Dict[str, Any]]:
        """Get recommendations to add variety to workouts"""
        recommendations = []
        
        if recent_workouts.empty:
            return recommendations
        
        # Analyze recent workouts to suggest variety
        recent_exercises = recent_workouts['exercise_name'].value_counts()
        
        # Suggest different exercise types
        if 'walking' in recent_exercises and recent_exercises['walking'] > 3:
            recommendations.append({
                'exercise': 'cycling',
                'intensity': 'moderate',
                'duration': 30,
                'frequency': '2-3 days/week',
                'reason': 'Add variety to your cardio routine'
            })
        
        if 'weight_training' in recent_exercises and recent_exercises['weight_training'] > 2:
            recommendations.append({
                'exercise': 'yoga',
                'intensity': 'moderate',
                'duration': 25,
                'frequency': '2 days/week',
                'reason': 'Improve flexibility and recovery'
            })
        
        return recommendations

# Global workout system instance
workout_system = WorkoutSystem() 