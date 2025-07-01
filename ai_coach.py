"""
AI Coaching module for FitTrack Pro
Provides personalized AI-powered nutrition and fitness coaching
"""

import openai
import json
import time
from datetime import datetime, date, timedelta
from typing import Dict, List, Optional, Any, Tuple
import streamlit as st
from config import Config
from data_manager import data_manager

class AICoach:
    """AI-powered nutrition and fitness coach"""
    
    def __init__(self):
        self.client = None
        self.setup_openai()
        self.conversation_history = []
        self.max_history_length = 20
    
    def setup_openai(self):
        """Setup OpenAI client"""
        if Config.OPENAI_API_KEY:
            try:
                self.client = openai.OpenAI(api_key=Config.OPENAI_API_KEY)
            except Exception as e:
                st.error(f"Error setting up OpenAI: {e}")
    
    def is_available(self) -> bool:
        """Check if AI coaching is available"""
        return self.client is not None and Config.OPENAI_API_KEY is not None
    
    def check_rate_limit(self) -> bool:
        """Check if we're within rate limits"""
        usage = data_manager.load_ai_usage()
        recent_requests = [
            req for req in usage['requests'] 
            if datetime.fromisoformat(req['timestamp']) > datetime.now() - timedelta(minutes=1)
        ]
        return len(recent_requests) < Config.MAX_AI_REQUESTS_PER_MINUTE
    
    def call_openai_api(self, messages: List[Dict[str, str]], max_tokens: int = 500) -> Optional[str]:
        """Call OpenAI API with error handling and rate limiting"""
        if not self.is_available():
            return "AI coaching is currently unavailable. Please check your OpenAI API key configuration."
        
        if not self.check_rate_limit():
            return "Rate limit reached. Please wait a moment before asking another question."
        
        try:
            if self.client is None:
                return "AI coaching is currently unavailable. Please check your OpenAI API key configuration."
            response = self.client.chat.completions.create(
                model=Config.AI_MODEL,
                messages=messages,
                max_tokens=max_tokens,
                temperature=0.7
            )
            
            ai_response = response.choices[0].message.content
            
            # Track usage
            tokens_used = response.usage.total_tokens
            cost = tokens_used * 0.00000015  # GPT-4o-mini pricing
            data_manager.save_ai_usage(tokens_used, cost)
            
            return ai_response
            
        except Exception as e:
            return f"Sorry, I'm having trouble connecting right now. Please try again later. (Error: {str(e)})"
    
    def get_user_context(self) -> Dict[str, Any]:
        """Get comprehensive user context for AI analysis"""
        profile = data_manager.load_user_profile()
        today_food = data_manager.get_today_food_summary()
        today_workout = data_manager.get_today_workout_summary()
        weekly_nutrition = data_manager.get_weekly_nutrition_summary()
        weekly_workout = data_manager.get_weekly_workout_summary()
        goals = data_manager.load_goals()
        
        context = {
            'profile': profile.to_dict() if profile else None,
            'today_food': today_food.to_dict(),
            'today_workout': today_workout,
            'weekly_nutrition': weekly_nutrition,
            'weekly_workout': weekly_workout,
            'goals': [g.to_dict() for g in goals],
            'current_date': date.today().isoformat(),
            'current_time': datetime.now().strftime('%H:%M')
        }
        
        return context
    
    def generate_daily_analysis(self) -> str:
        """Generate comprehensive daily nutrition and fitness analysis"""
        context = self.get_user_context()
        
        system_prompt = """You are a knowledgeable, supportive nutrition and fitness coach for FitTrack Pro. 
        You provide personalized, actionable advice based on user data. 
        Be encouraging, specific, and practical in your recommendations.
        Focus on the user's specific goals and current progress."""
        
        user_prompt = f"""
        Please analyze today's nutrition and fitness data and provide personalized insights:
        
        User Profile: {context['profile']}
        Today's Food: {context['today_food']}
        Today's Workout: {context['today_workout']}
        Weekly Nutrition: {context['weekly_nutrition']}
        Weekly Workout: {context['weekly_workout']}
        Current Goals: {context['goals']}
        
        Provide:
        1. Brief analysis of today's nutrition (calories, macros, meal timing)
        2. Workout analysis and calorie burn assessment
        3. Progress toward goals
        4. 2-3 specific, actionable recommendations for tomorrow
        5. Motivational message based on user's coaching style preference
        
        Keep the response concise but helpful (3-4 paragraphs max).
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_openai_api(messages, max_tokens=600)
    
    def generate_weekly_report(self) -> str:
        """Generate comprehensive weekly analysis and recommendations"""
        context = self.get_user_context()
        
        system_prompt = """You are a knowledgeable, supportive nutrition and fitness coach for FitTrack Pro. 
        You provide detailed weekly analysis and strategic recommendations for long-term success."""
        
        user_prompt = f"""
        Please provide a comprehensive weekly analysis and strategic recommendations:
        
        User Profile: {context['profile']}
        Weekly Nutrition Summary: {context['weekly_nutrition']}
        Weekly Workout Summary: {context['weekly_workout']}
        Current Goals: {context['goals']}
        
        Provide:
        1. Weekly nutrition pattern analysis (consistency, macro balance, timing)
        2. Workout consistency and intensity assessment
        3. Progress toward goals with specific metrics
        4. Identification of patterns (good and areas for improvement)
        5. Strategic recommendations for the upcoming week
        6. Goal adjustment suggestions if needed
        
        Make this a comprehensive but readable analysis (4-5 paragraphs).
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_openai_api(messages, max_tokens=800)
    
    def answer_question(self, question: str) -> str:
        """Answer user's nutrition or fitness question"""
        context = self.get_user_context()
        
        system_prompt = """You are a knowledgeable, supportive nutrition and fitness coach for FitTrack Pro. 
        Answer questions based on scientific evidence and the user's specific situation.
        Be educational, practical, and encouraging."""
        
        user_prompt = f"""
        User Question: {question}
        
        User Context:
        Profile: {context['profile']}
        Current Goals: {context['goals']}
        Recent Nutrition: {context['today_food']}
        Recent Workout: {context['today_workout']}
        
        Provide a helpful, educational response that considers the user's specific situation and goals.
        Keep it concise but informative (2-3 paragraphs).
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_openai_api(messages, max_tokens=500)
    
    def get_meal_suggestions(self, meal_type: str, remaining_calories: float) -> str:
        """Get AI-powered meal suggestions"""
        context = self.get_user_context()
        
        system_prompt = """You are a knowledgeable nutrition coach providing meal suggestions.
        Consider the user's goals, preferences, and remaining calories for the day."""
        
        user_prompt = f"""
        Please suggest meal options for {meal_type} with approximately {remaining_calories} calories remaining.
        
        User Context:
        Profile: {context['profile']}
        Goals: {context['goals']}
        Today's Food So Far: {context['today_food']}
        Dietary Restrictions: {context['profile']['dietary_restrictions'] if context['profile'] else []}
        Favorite Foods: {context['profile']['favorite_foods'] if context['profile'] else []}
        
        Provide:
        1. 3-4 specific meal suggestions with approximate calories
        2. Consider macro balance for the day
        3. Include favorite foods when possible
        4. Respect dietary restrictions
        5. Brief explanation of why these suggestions are good choices
        
        Keep suggestions practical and easy to implement.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_openai_api(messages, max_tokens=400)
    
    def get_workout_recommendations(self) -> str:
        """Get AI-powered workout recommendations"""
        context = self.get_user_context()
        
        system_prompt = """You are a knowledgeable fitness coach providing workout recommendations.
        Consider the user's fitness level, goals, and recent workout history."""
        
        user_prompt = f"""
        Please provide workout recommendations based on the user's current situation.
        
        User Context:
        Profile: {context['profile']}
        Goals: {context['goals']}
        Recent Workouts: {context['weekly_workout']}
        Today's Workout: {context['today_workout']}
        
        Provide:
        1. Recommended workout type for today/tomorrow
        2. Specific exercises or activities
        3. Duration and intensity recommendations
        4. Rest day considerations
        5. Progress tracking suggestions
        
        Consider the user's fitness level and goals when making recommendations.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_openai_api(messages, max_tokens=400)
    
    def analyze_patterns(self) -> str:
        """Analyze user's eating and workout patterns"""
        context = self.get_user_context()
        
        system_prompt = """You are a data-savvy nutrition and fitness coach analyzing user patterns.
        Identify trends, habits, and opportunities for improvement."""
        
        user_prompt = f"""
        Please analyze the user's nutrition and workout patterns:
        
        Weekly Nutrition: {context['weekly_nutrition']}
        Weekly Workout: {context['weekly_workout']}
        Goals: {context['goals']}
        
        Identify:
        1. Consistent patterns (good habits to maintain)
        2. Inconsistent patterns (areas for improvement)
        3. Day-of-week trends
        4. Correlation between nutrition and workout performance
        5. Progress toward goals
        6. Specific recommendations based on patterns
        
        Provide actionable insights based on the data.
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_openai_api(messages, max_tokens=600)
    
    def get_motivational_message(self) -> str:
        """Get personalized motivational message"""
        context = self.get_user_context()
        
        system_prompt = """You are a supportive, motivational coach.
        Provide encouragement based on the user's progress and coaching style preference."""
        
        user_prompt = f"""
        Please provide a motivational message for the user:
        
        User Context:
        Profile: {context['profile']}
        Goals: {context['goals']}
        Recent Progress: {context['today_food']} and {context['today_workout']}
        Coaching Style Preference: {context['profile']['ai_coaching_style'] if context['profile'] else 'motivational'}
        
        Create a brief, encouraging message that:
        1. Acknowledges recent progress
        2. Reinforces positive habits
        3. Provides gentle motivation for continued effort
        4. Matches the user's preferred coaching style
        
        Keep it concise and uplifting (1-2 paragraphs).
        """
        
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
        
        return self.call_openai_api(messages, max_tokens=300)
    
    def chat_with_coach(self, user_message: str) -> str:
        """Interactive chat with the AI coach"""
        context = self.get_user_context()
        
        # Add user message to conversation history
        self.conversation_history.append({"role": "user", "content": user_message})
        
        # Keep conversation history manageable
        if len(self.conversation_history) > self.max_history_length:
            self.conversation_history = self.conversation_history[-self.max_history_length:]
        
        system_prompt = f"""You are a knowledgeable, supportive nutrition and fitness coach for FitTrack Pro.
        You have access to the user's profile, goals, and recent data.
        Be conversational, helpful, and provide specific advice based on their situation.
        Coaching Style: {context['profile']['ai_coaching_style'] if context['profile'] else 'motivational'}
        """
        
        # Build messages with context and conversation history
        messages = [{"role": "system", "content": system_prompt}]
        
        # Add context as a system message
        context_message = f"""
        User Context:
        Profile: {context['profile']}
        Goals: {context['goals']}
        Today's Progress: {context['today_food']} and {context['today_workout']}
        """
        messages.append({"role": "system", "content": context_message})
        
        # Add conversation history
        messages.extend(self.conversation_history)
        
        response = self.call_openai_api(messages, max_tokens=400)
        if response is None:
            response = "Sorry, I'm having trouble connecting right now. Please try again later."
        # Add AI response to conversation history
        self.conversation_history.append({"role": "assistant", "content": response})
        return response
    
    def save_conversation(self, user_id: str, message: str, response: str, context: Optional[Dict] = None):
        """Save conversation to history"""
        data_manager.save_ai_conversation(user_id, message, response, context)
    
    def get_usage_stats(self) -> Dict[str, Any]:
        """Get AI usage statistics"""
        return data_manager.load_ai_usage()

# Global AI coach instance
ai_coach = AICoach() 