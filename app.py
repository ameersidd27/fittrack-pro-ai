import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
import os
import requests
import json
import time
import numpy as np
from collections import defaultdict
from typing import Dict, List, Optional, Tuple
import hashlib

# Conditional OpenAI import
try:
    import openai
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

# CSV file paths
FOOD_LOG_FILE = "food_logs.csv"
USER_PROFILE_FILE = "user_profile.csv"
FOOD_CACHE_FILE = "food_cache.json"

# USDA FoodData Central API configuration
USDA_API_KEY = "DEMO_KEY"  # Replace with your actual API key for production
USDA_BASE_URL = "https://api.nal.usda.gov/fdc/v1"

# AI Coaching System Configuration
AI_INSIGHTS_FILE = "ai_insights.json"
AI_CONVERSATION_FILE = "ai_conversations.json"
AI_USAGE_FILE = "ai_usage.json"

# OpenAI Configuration
if OPENAI_AVAILABLE:
    try:
        openai_client = openai.OpenAI(api_key=os.getenv('OPENAI_API_KEY'))
        AI_ENABLED = True if os.getenv('OPENAI_API_KEY') else False
    except:
        AI_ENABLED = False
else:
    AI_ENABLED = False

MODEL = "gpt-4o-mini"
MAX_REQUESTS_PER_MINUTE = 3

def load_food_logs():
    """Load food logs from CSV file"""
    if os.path.exists(FOOD_LOG_FILE):
        return pd.read_csv(FOOD_LOG_FILE)
    else:
        # Create empty DataFrame with proper columns
        return pd.DataFrame({
            'date': [],
            'food_name': [],
            'calories': [],
            'timestamp': []
        })

def load_user_profile():
    """Load user profile from CSV file"""
    if os.path.exists(USER_PROFILE_FILE):
        return pd.read_csv(USER_PROFILE_FILE)
    else:
        # Create empty DataFrame with proper columns
        return pd.DataFrame({
            'age': [],
            'weight': [],
            'height': [],
            'gender': [],
            'activity_level': [],
            'bmr': [],
            'daily_calories': [],
            'last_updated': []
        })

def save_user_profile(age, weight, height, gender, activity_level):
    """Save user profile to CSV"""
    bmr = calculate_bmr(age, weight, height, gender)
    daily_calories = calculate_daily_calories(bmr, activity_level)
    
    # Create new profile
    new_profile = {
        'age': [age],
        'weight': [weight],
        'height': [height],
        'gender': [gender],
        'activity_level': [activity_level],
        'bmr': [bmr],
        'daily_calories': [daily_calories],
        'last_updated': [datetime.now().strftime('%Y-%m-%d %H:%M:%S')]
    }
    
    # Save to CSV
    df = pd.DataFrame(new_profile)
    df.to_csv(USER_PROFILE_FILE, index=False)
    return df

def get_user_calorie_target():
    """Get user's daily calorie target from profile"""
    df = load_user_profile()
    if not df.empty:
        return df.iloc[0]['daily_calories']
    else:
        # Fallback to default calculation
        return calculate_daily_calories(1500, 'moderately_active')

def save_food_log(food_name, calories):
    """Save a new food log entry to CSV"""
    # Load existing data
    df = load_food_logs()
    
    # Create new row
    new_row = {
        'date': date.today().strftime('%Y-%m-%d'),
        'food_name': food_name,
        'calories': calories,
        'timestamp': datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    }
    
    # Add to DataFrame
    df = pd.concat([df, pd.DataFrame([new_row])], ignore_index=True)
    
    # Save to CSV
    df.to_csv(FOOD_LOG_FILE, index=False)
    return df

def get_today_calories():
    """Get total calories for today"""
    df = load_food_logs()
    today = date.today().strftime('%Y-%m-%d')
    today_foods = df[df['date'] == today]
    return today_foods['calories'].sum()

def calculate_bmr(age, weight, height, gender):
    """Calculate Basal Metabolic Rate using Mifflin-St Jeor Equation"""
    if gender.lower() == 'male':
        bmr = 10 * weight + 6.25 * height - 5 * age + 5
    else:  # female
        bmr = 10 * weight + 6.25 * height - 5 * age - 161
    return round(bmr)

def calculate_daily_calories(bmr, activity_level):
    """Calculate daily calorie needs based on activity level"""
    activity_multipliers = {
        'sedentary': 1.2,      # Little or no exercise
        'lightly_active': 1.375,  # Light exercise 1-3 days/week
        'moderately_active': 1.55,  # Moderate exercise 3-5 days/week
        'very_active': 1.725,   # Hard exercise 6-7 days/week
        'extremely_active': 1.9  # Very hard exercise, physical job
    }
    
    daily_calories = bmr * activity_multipliers.get(activity_level, 1.2)
    return round(daily_calories)

# Fallback food database (used when API fails)
FALLBACK_FOOD_DATABASE = {
    'oatmeal': {'calories': 150, 'protein': 6, 'carbs': 27, 'fat': 3, 'category': 'breakfast'},
    'eggs': {'calories': 70, 'protein': 6, 'carbs': 1, 'fat': 5, 'category': 'breakfast'},
    'chicken_breast': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'category': 'lunch'},
    'salmon': {'calories': 208, 'protein': 25, 'carbs': 0, 'fat': 12, 'category': 'lunch'},
    'brown_rice': {'calories': 110, 'protein': 2.5, 'carbs': 23, 'fat': 0.9, 'category': 'lunch'},
    'apple': {'calories': 95, 'protein': 0.5, 'carbs': 25, 'fat': 0.3, 'category': 'snack'},
    'almonds': {'calories': 164, 'protein': 6, 'carbs': 6, 'fat': 14, 'category': 'snack'},
    'banana': {'calories': 105, 'protein': 1, 'carbs': 27, 'fat': 0.4, 'category': 'breakfast'}
}

def load_food_cache():
    """Load cached food data from JSON file"""
    if os.path.exists(FOOD_CACHE_FILE):
        try:
            with open(FOOD_CACHE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_food_cache(cache_data):
    """Save food cache to JSON file"""
    try:
        with open(FOOD_CACHE_FILE, 'w') as f:
            json.dump(cache_data, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save cache: {e}")

def search_usda_foods(query, max_results=10):
    """Search for foods using USDA FoodData Central API"""
    try:
        # Check cache first
        cache = load_food_cache()
        cache_key = f"search_{query.lower()}"
        
        if cache_key in cache:
            # Check if cache is less than 24 hours old
            if time.time() - cache[cache_key]['timestamp'] < 86400:  # 24 hours
                return cache[cache_key]['data']
        
        # API search
        search_url = f"{USDA_BASE_URL}/foods/search"
        params = {
            'api_key': USDA_API_KEY,
            'query': query,
            'pageSize': max_results,
            'dataType': ['Foundation', 'SR Legacy', 'Survey (FNDDS)']
        }
        
        response = requests.get(search_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        foods = []
        
        for food in data.get('foods', []):
            food_info = {
                'fdc_id': food.get('fdcId'),
                'name': food.get('description', 'Unknown'),
                'brand': food.get('brandOwner', ''),
                'category': food.get('foodCategory', ''),
                'data_type': food.get('dataType', ''),
                'nutrients': {}
            }
            
            # Extract nutrients
            for nutrient in food.get('foodNutrients', []):
                nutrient_id = nutrient.get('nutrientId')
                value = nutrient.get('value', 0)
                
                # Map common nutrients
                if nutrient_id == 1008:  # Calories
                    food_info['nutrients']['calories'] = value
                elif nutrient_id == 1003:  # Protein
                    food_info['nutrients']['protein'] = value
                elif nutrient_id == 1005:  # Carbohydrates
                    food_info['nutrients']['carbs'] = value
                elif nutrient_id == 1004:  # Total Fat
                    food_info['nutrients']['fat'] = value
            
            foods.append(food_info)
        
        # Cache the results
        cache[cache_key] = {
            'data': foods,
            'timestamp': time.time()
        }
        save_food_cache(cache)
        
        return foods
        
    except Exception as e:
        st.error(f"API search failed: {e}")
        return []

def get_usda_food_details(fdc_id):
    """Get detailed nutrition information for a specific food"""
    try:
        # Check cache first
        cache = load_food_cache()
        cache_key = f"details_{fdc_id}"
        
        if cache_key in cache:
            # Check if cache is less than 24 hours old
            if time.time() - cache[cache_key]['timestamp'] < 86400:  # 24 hours
                return cache[cache_key]['data']
        
        # API call for detailed info
        detail_url = f"{USDA_BASE_URL}/food/{fdc_id}"
        params = {'api_key': USDA_API_KEY}
        
        response = requests.get(detail_url, params=params, timeout=10)
        response.raise_for_status()
        
        data = response.json()
        
        # Extract detailed nutrition info
        nutrients = {}
        for nutrient in data.get('foodNutrients', []):
            nutrient_id = nutrient.get('nutrientId')
            value = nutrient.get('value', 0)
            unit = nutrient.get('unitName', 'g')
            
            # Map common nutrients
            if nutrient_id == 1008:  # Calories
                nutrients['calories'] = value
            elif nutrient_id == 1003:  # Protein
                nutrients['protein'] = value
            elif nutrient_id == 1005:  # Carbohydrates
                nutrients['carbs'] = value
            elif nutrient_id == 1004:  # Total Fat
                nutrients['fat'] = value
            elif nutrient_id == 1079:  # Fiber
                nutrients['fiber'] = value
            elif nutrient_id == 1093:  # Sodium
                nutrients['sodium'] = value
            elif nutrient_id == 1087:  # Calcium
                nutrients['calcium'] = value
            elif nutrient_id == 1089:  # Iron
                nutrients['iron'] = value
        
        food_details = {
            'fdc_id': fdc_id,
            'name': data.get('description', 'Unknown'),
            'brand': data.get('brandOwner', ''),
            'category': data.get('foodCategory', ''),
            'nutrients': nutrients
        }
        
        # Cache the results
        cache[cache_key] = {
            'data': food_details,
            'timestamp': time.time()
        }
        save_food_cache(cache)
        
        return food_details
        
    except Exception as e:
        st.error(f"Failed to get food details: {e}")
        return None

def search_foods_with_fallback(query, max_results=10):
    """Search for foods with fallback to local database"""
    # Try USDA API first
    usda_results = search_usda_foods(query, max_results)
    
    if usda_results:
        return usda_results
    
    # Fallback to local database
    fallback_results = []
    query_lower = query.lower()
    
    for food_name, nutrition in FALLBACK_FOOD_DATABASE.items():
        if query_lower in food_name.lower():
            fallback_results.append({
                'fdc_id': f"fallback_{food_name}",
                'name': food_name.replace('_', ' ').title(),
                'brand': 'Local Database',
                'category': nutrition['category'],
                'data_type': 'Fallback',
                'nutrients': {
                    'calories': nutrition['calories'],
                    'protein': nutrition['protein'],
                    'carbs': nutrition['carbs'],
                    'fat': nutrition['fat']
                }
            })
    
    return fallback_results[:max_results]

def get_meal_category():
    """Determine meal category based on current time"""
    current_hour = datetime.now().hour
    
    if 5 <= current_hour < 11:
        return 'breakfast'
    elif 11 <= current_hour < 16:
        return 'lunch'
    elif 16 <= current_hour < 21:
        return 'dinner'
    else:
        return 'snack'

def get_remaining_calories():
    """Calculate remaining daily calories based on user profile and food logged today"""
    # Get user's actual calorie target from profile
    daily_target = get_user_calorie_target()
    
    # Get calories eaten today
    calories_eaten = get_today_calories()
    
    remaining = daily_target - calories_eaten
    return max(0, remaining), daily_target

def suggest_foods(remaining_calories, meal_category=None):
    """Suggest foods that fit the remaining calories"""
    if meal_category is None:
        meal_category = get_meal_category()
    
    suggestions = []
    
    # Use fallback database for suggestions (since we need categorized foods)
    for food_name, nutrition in FALLBACK_FOOD_DATABASE.items():
        if nutrition['category'] == meal_category and nutrition['calories'] <= remaining_calories:
            suggestions.append({
                'name': food_name.replace('_', ' ').title(),
                'calories': nutrition['calories'],
                'protein': nutrition['protein'],
                'carbs': nutrition['carbs'],
                'fat': nutrition['fat'],
                'category': nutrition['category']
            })
    
    # Sort by protein content (prioritize protein-rich foods)
    suggestions.sort(key=lambda x: x['protein'], reverse=True)
    
    # Return top 5 suggestions
    return suggestions[:5]

# AI Coaching System Functions
def load_ai_conversations():
    """Load AI conversation history"""
    if os.path.exists(AI_CONVERSATION_FILE):
        try:
            with open(AI_CONVERSATION_FILE, 'r') as f:
                return json.load(f)
        except:
            return {}
    return {}

def save_ai_conversation(user_id: str, message: str, response: str, context: Optional[Dict] = None):
    """Save AI conversation to history"""
    conversations = load_ai_conversations()
    
    if user_id not in conversations:
        conversations[user_id] = []
    
    conversations[user_id].append({
        'timestamp': datetime.now().isoformat(),
        'message': message,
        'response': response,
        'context': context or {}
    })
    
    # Keep only last 50 conversations to manage file size
    conversations[user_id] = conversations[user_id][-50:]
    
    try:
        with open(AI_CONVERSATION_FILE, 'w') as f:
            json.dump(conversations, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save conversation: {e}")

def load_ai_usage():
    """Load AI usage tracking data"""
    if os.path.exists(AI_USAGE_FILE):
        try:
            with open(AI_USAGE_FILE, 'r') as f:
                return json.load(f)
        except:
            return {'requests': [], 'total_tokens': 0, 'total_cost': 0.0}
    return {'requests': [], 'total_tokens': 0, 'total_cost': 0.0}

def save_ai_usage(tokens_used: int, cost: float):
    """Save AI usage data"""
    usage = load_ai_usage()
    
    usage['requests'].append({
        'timestamp': datetime.now().isoformat(),
        'tokens': tokens_used,
        'cost': cost
    })
    
    usage['total_tokens'] += tokens_used
    usage['total_cost'] += cost
    
    # Keep only last 100 requests
    usage['requests'] = usage['requests'][-100:]
    
    try:
        with open(AI_USAGE_FILE, 'w') as f:
            json.dump(usage, f, indent=2)
    except Exception as e:
        st.error(f"Failed to save usage data: {e}")

def analyze_daily_nutrition():
    """Analyze today's nutrition patterns"""
    df = load_food_logs()
    today = date.today().strftime('%Y-%m-%d')
    today_foods = df[df['date'] == today]
    
    if today_foods.empty:
        return {
            'total_calories': 0,
            'meal_count': 0,
            'meal_timing': [],
            'nutrition_gaps': [],
            'excesses': [],
            'balance_score': 0
        }
    
    # Basic analysis
    total_calories = today_foods['calories'].sum()
    meal_count = len(today_foods)
    
    # Meal timing analysis
    meal_timing = []
    for _, row in today_foods.iterrows():
        try:
            meal_time = datetime.strptime(row['timestamp'], '%Y-%m-%d %H:%M:%S')
            meal_timing.append({
                'time': meal_time.strftime('%H:%M'),
                'hour': meal_time.hour,
                'food': row['food_name'],
                'calories': row['calories']
            })
        except:
            pass
    
    # Sort by time
    meal_timing.sort(key=lambda x: x['hour'])
    
    # Get user's calorie target
    daily_target = get_user_calorie_target()
    
    # Calculate gaps and excesses
    nutrition_gaps = []
    excesses = []
    
    if total_calories < daily_target * 0.8:
        nutrition_gaps.append(f"Under-eating by {daily_target - total_calories:.0f} calories")
    elif total_calories > daily_target * 1.2:
        excesses.append(f"Over-eating by {total_calories - daily_target:.0f} calories")
    
    # Balance score (0-100)
    if daily_target > 0:
        balance_score = max(0, 100 - abs(total_calories - daily_target) / daily_target * 100)
    else:
        balance_score = 0
    
    return {
        'total_calories': total_calories,
        'meal_count': meal_count,
        'meal_timing': meal_timing,
        'nutrition_gaps': nutrition_gaps,
        'excesses': excesses,
        'balance_score': balance_score,
        'daily_target': daily_target
    }

def analyze_weekly_patterns():
    """Analyze weekly eating patterns"""
    df = load_food_logs()
    
    if df.empty:
        return {}
    
    # Get last 7 days
    end_date = date.today()
    start_date = end_date - timedelta(days=7)
    
    # Filter for last 7 days
    df['date_obj'] = pd.to_datetime(df['date'])
    weekly_data = df[(df['date_obj'] >= pd.Timestamp(start_date)) & 
                     (df['date_obj'] <= pd.Timestamp(end_date))]
    
    if weekly_data.empty:
        return {}
    
    # Group by day of week
    weekly_data['day_of_week'] = weekly_data['date_obj'].dt.day_name()
    daily_calories = weekly_data.groupby('day_of_week')['calories'].sum()
    
    # Calculate patterns
    avg_daily_calories = daily_calories.mean()
    daily_target = get_user_calorie_target()
    
    patterns = {}
    for day, calories in daily_calories.items():
        deviation = calories - daily_target
        patterns[day] = {
            'calories': calories,
            'deviation': deviation,
            'percentage': (calories / daily_target * 100) if daily_target > 0 else 0
        }
    
    # Identify consistent patterns
    consistent_patterns = []
    for day, data in patterns.items():
        if data['deviation'] < -300:  # Under-eating by 300+ calories
            consistent_patterns.append(f"Consistently under-eating on {day}s")
        elif data['deviation'] > 300:  # Over-eating by 300+ calories
            consistent_patterns.append(f"Consistently over-eating on {day}s")
    
    return {
        'daily_patterns': patterns,
        'avg_daily_calories': avg_daily_calories,
        'consistent_patterns': consistent_patterns,
        'daily_target': daily_target,
        'total_days': len(daily_calories)
    }

def generate_ai_prompt(context_type: str, user_data: Dict) -> str:
    """Generate AI prompts based on context type"""
    
    base_prompt = """You are a knowledgeable, supportive nutrition coach for FitTrack Pro. 
    You provide personalized, actionable advice based on user data. 
    Be encouraging, specific, and practical in your recommendations.
    Keep responses concise but helpful (2-3 sentences for brief responses, up to 5-6 for detailed analysis)."""
    
    if context_type == "daily_analysis":
        daily_analysis = analyze_daily_nutrition()
        
        prompt = f"""{base_prompt}
        
        Today's Nutrition Analysis:
        - Total Calories: {daily_analysis['total_calories']}
        - Daily Target: {daily_analysis['daily_target']}
        - Meals Eaten: {daily_analysis['meal_count']}
        - Balance Score: {daily_analysis['balance_score']:.0f}/100
        - Nutrition Gaps: {', '.join(daily_analysis['nutrition_gaps']) if daily_analysis['nutrition_gaps'] else 'None'}
        - Excesses: {', '.join(daily_analysis['excesses']) if daily_analysis['excesses'] else 'None'}
        
        Meal Timing: {[f"{meal['time']} - {meal['food']} ({meal['calories']} cal)" for meal in daily_analysis['meal_timing']]}
        
        Provide a brief, encouraging analysis of today's nutrition and 1-2 specific suggestions for improvement."""
        
    elif context_type == "weekly_patterns":
        weekly_analysis = analyze_weekly_patterns()
        
        if not weekly_analysis:
            return "No weekly data available for analysis."
        
        prompt = f"""{base_prompt}
        
        Weekly Pattern Analysis:
        - Average Daily Calories: {weekly_analysis['avg_daily_calories']:.0f}
        - Daily Target: {weekly_analysis['daily_target']}
        - Consistent Patterns: {', '.join(weekly_analysis['consistent_patterns']) if weekly_analysis['consistent_patterns'] else 'None'}
        
        Daily Breakdown:
        {chr(10).join([f"- {day}: {data['calories']:.0f} cal ({data['deviation']:+.0f} from target)" for day, data in weekly_analysis['daily_patterns'].items()])}
        
        Provide insights about weekly eating patterns and 2-3 specific recommendations for improvement."""
        
    elif context_type == "general_question":
        prompt = f"""{base_prompt}
        
        User Question: {user_data.get('question', '')}
        
        Provide a helpful, educational response to the user's nutrition question."""
        
    else:
        prompt = base_prompt
    
    return prompt or "No prompt generated."

def call_openai_api(prompt: str, max_tokens: int = 300) -> Optional[str]:
    """Call OpenAI API with error handling and rate limiting"""
    if not AI_ENABLED or not OPENAI_AVAILABLE:
        return "AI coaching is currently unavailable. Please check your OpenAI API key configuration."
    
    try:
        # Simple rate limiting check
        usage = load_ai_usage()
        recent_requests = [req for req in usage['requests'] 
                          if datetime.fromisoformat(req['timestamp']) > datetime.now() - timedelta(minutes=1)]
        
        if len(recent_requests) >= MAX_REQUESTS_PER_MINUTE:
            return "Rate limit reached. Please wait a moment before asking another question."
        
        # Call OpenAI API
        response = openai_client.chat.completions.create(
            model=MODEL,
            messages=[
                {"role": "system", "content": "You are a supportive nutrition coach for FitTrack Pro."},
                {"role": "user", "content": prompt}
            ],
            max_tokens=max_tokens,
            temperature=0.7
        )
        
        ai_response = response.choices[0].message.content
        
        # Track usage
        tokens_used = response.usage.total_tokens
        cost = tokens_used * 0.00000015  # GPT-4o-mini pricing
        save_ai_usage(tokens_used, cost)
        
        return ai_response
        
    except Exception as e:
        return f"Sorry, I'm having trouble connecting right now. Please try again later. (Error: {str(e)})"

def get_ai_insight(context_type: str, user_data: Optional[Dict] = None) -> str:
    """Get AI insight for specific context"""
    if user_data is None:
        user_data = {}
    
    prompt = generate_ai_prompt(context_type, user_data)
    return call_openai_api(prompt)

# App title
st.title("FitTrack Pro")
st.subheader("Fitness & Nutrition Tracker 💯")

# Sidebar for navigation
page = st.sidebar.selectbox("Choose a page", ["Dashboard", "Log Food", "View Food Logs", "BMR Calculator", "User Profile", "AI Coach", "Log Workout", "Progress"])

if page == "Dashboard":
    st.header("📊 Today's Overview")
    
    # Get today's calories
    today_calories = get_today_calories()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Calories Eaten", f"{today_calories:,}", "0")
    with col2:
        st.metric("Calories Burned", "300", "50")
    with col3:
        net_calories = today_calories - 300
        st.metric("Net Calories", f"{net_calories:,}", f"{today_calories - 300}")
    
    # Smart Food Suggestions Section
    st.markdown("---")
    st.subheader("🍽️ Smart Food Suggestions")
    
    # Check if user has a profile
    profile_df = load_user_profile()
    if profile_df.empty:
        st.warning("⚠️ No user profile found. Please set up your profile in the 'User Profile' page to get personalized food suggestions.")
        st.info("For now, using default calorie target of 2,325 calories (moderate activity level).")
    
    # Get remaining calories and suggestions
    remaining_calories, daily_target = get_remaining_calories()
    meal_category = get_meal_category()
    suggestions = suggest_foods(remaining_calories, meal_category)
    
    # Display remaining calories info
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Daily Target", f"{daily_target:,} cal")
    with col2:
        st.metric("Remaining", f"{remaining_calories:,} cal")
    
    # Show meal context
    meal_emojis = {
        'breakfast': '🌅',
        'lunch': '☀️', 
        'dinner': '🌙',
        'snack': '🍿'
    }
    
    meal_names = {
        'breakfast': 'Breakfast',
        'lunch': 'Lunch',
        'dinner': 'Dinner', 
        'snack': 'Snack'
    }
    
    st.info(f"{meal_emojis.get(meal_category, '🍽️')} It's {meal_names[meal_category]} time! Here are some suggestions that fit your remaining calories:")
    
    if suggestions:
        # Display suggestions in a nice format
        for i, food in enumerate(suggestions, 1):
            with st.container():
                col1, col2, col3, col4, col5 = st.columns([3, 1, 1, 1, 1])
                
                with col1:
                    st.write(f"**{i}. {food['name']}**")
                
                with col2:
                    st.write(f"**{food['calories']}** cal")
                
                with col3:
                    st.write(f"P: {food['protein']}g")
                
                with col4:
                    st.write(f"C: {food['carbs']}g")
                
                with col5:
                    st.write(f"F: {food['fat']}g")
                
                # Add a subtle separator
                st.markdown("<hr style='margin: 0.5rem 0; border-color: #333333;'>", unsafe_allow_html=True)
        
        # Quick add buttons
        st.subheader("Quick Add")
        cols = st.columns(len(suggestions))
        for i, (col, food) in enumerate(zip(cols, suggestions)):
            with col:
                if st.button(f"Add {food['name']}", key=f"quick_add_{i}"):
                    save_food_log(food['name'], food['calories'])
                    st.success(f"Added {food['name']}!")
                    st.rerun()
    else:
        st.warning("No foods found that fit your remaining calories. Consider logging a smaller portion or choosing a different meal category.")
        
        # Show alternative suggestions from other categories
        st.subheader("Alternative Suggestions")
        all_suggestions = suggest_foods(remaining_calories, None)  # No category filter
        if all_suggestions:
            for i, food in enumerate(all_suggestions[:3], 1):
                st.write(f"**{i}. {food['name']}** - {food['calories']} cal (P: {food['protein']}g, C: {food['carbs']}g, F: {food['fat']}g)")
    
    # AI Insights Section
    if AI_ENABLED:
        st.markdown("---")
        st.subheader("🤖 AI Coach Insights")
        
        # Quick AI analysis button
        if st.button("Get Today's AI Analysis", key="dashboard_ai"):
            with st.spinner("AI Coach is analyzing..."):
                ai_insight = get_ai_insight("daily_analysis")
                st.success("AI Analysis Complete!")
                st.info(ai_insight)
                
                # Save conversation
                save_ai_conversation("user", "Dashboard daily analysis", ai_insight)
        else:
            st.info("Click the button above to get personalized AI insights about your nutrition today!")

elif page == "Log Food":
    st.header("🍎 Log Your Food")
    
    # Food search section
    st.subheader("🔍 Search for Foods")
    search_query = st.text_input("Search for a food item", placeholder="e.g., apple, chicken breast, oatmeal")
    
    if search_query and len(search_query) >= 2:
        with st.spinner("Searching for foods..."):
            search_results = search_foods_with_fallback(search_query, max_results=8)
        
        if search_results:
            st.subheader("Search Results")
            
            # Display search results
            for i, food in enumerate(search_results):
                with st.expander(f"{food['name']} - {food.get('brand', '')}"):
                    col1, col2, col3, col4 = st.columns(4)
                    
                    with col1:
                        st.write("**Calories:**")
                        st.write(f"{food['nutrients'].get('calories', 'N/A')}")
                    
                    with col2:
                        st.write("**Protein:**")
                        st.write(f"{food['nutrients'].get('protein', 'N/A')}g")
                    
                    with col3:
                        st.write("**Carbs:**")
                        st.write(f"{food['nutrients'].get('carbs', 'N/A')}g")
                    
                    with col4:
                        st.write("**Fat:**")
                        st.write(f"{food['nutrients'].get('fat', 'N/A')}g")
                    
                    # Add button for this food
                    if st.button(f"Add {food['name']}", key=f"add_search_{i}"):
                        calories = food['nutrients'].get('calories', 0)
                        save_food_log(food['name'], calories)
                        st.success(f"Added {food['name']} ({calories} calories)")
                        st.balloons()
                        st.rerun()
        else:
            st.warning("No foods found. Try a different search term.")
    
    # Manual food entry section
    st.markdown("---")
    st.subheader("✏️ Manual Food Entry")
    
    food_name = st.text_input("Food item name")
    calories = st.number_input("Calories", min_value=0, value=100)
    
    if st.button("Add Food"):
        if food_name.strip():  # Check if food name is not empty
            save_food_log(food_name, calories)
            st.success(f"Added {food_name} ({calories} calories)")
            st.balloons()
        else:
            st.error("Please enter a food name")

elif page == "View Food Logs":
    st.header("📋 Food Log History")
    
    df = load_food_logs()
    
    if not df.empty:
        # Show today's foods
        today = date.today().strftime('%Y-%m-%d')
        today_foods = df[df['date'] == today]
        
        if not today_foods.empty:
            st.subheader("🍽️ Today's Foods")
            st.dataframe(today_foods[['food_name', 'calories', 'timestamp']], hide_index=True)
            
            total_today = today_foods['calories'].sum()
            st.info(f"Total calories today: {total_today:,}")
        
        # Show all foods with date filter
        st.subheader("📅 All Food Logs")
        
        # Date filter
        unique_dates = sorted(df['date'].unique(), reverse=True)
        selected_date = st.selectbox("Select date to view:", unique_dates)
        
        if selected_date:
            filtered_df = df[df['date'] == selected_date]
            st.dataframe(filtered_df[['food_name', 'calories', 'timestamp']], hide_index=True)
            
            total_selected = filtered_df['calories'].sum()
            st.info(f"Total calories on {selected_date}: {total_selected:,}")
        
        # Summary statistics
        st.subheader("📊 Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Entries", len(df))
            st.metric("Total Calories", f"{df['calories'].sum():,}")
        
        with col2:
            st.metric("Average per Day", f"{df.groupby('date')['calories'].sum().mean():.0f}")
            st.metric("Days Tracked", len(df['date'].unique()))
    
    else:
        st.info("No food logs yet. Start logging your meals!")

elif page == "BMR Calculator":
    st.header("🧮 BMR & Daily Calorie Calculator")
    st.write("Calculate your Basal Metabolic Rate (BMR) and daily calorie needs based on your body composition and activity level.")
    
    # Input form
    with st.form("bmr_calculator"):
        col1, col2 = st.columns(2)
        
        with col1:
            age = st.number_input("Age (years)", min_value=15, max_value=100, value=25)
            weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
            gender = st.selectbox("Gender", ["Male", "Female"])
        
        with col2:
            height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
            activity_level = st.selectbox(
                "Activity Level",
                ["sedentary", "lightly_active", "moderately_active", "very_active", "extremely_active"],
                format_func=lambda x: {
                    "sedentary": "Sedentary (Little or no exercise)",
                    "lightly_active": "Lightly Active (Light exercise 1-3 days/week)",
                    "moderately_active": "Moderately Active (Moderate exercise 3-5 days/week)",
                    "very_active": "Very Active (Hard exercise 6-7 days/week)",
                    "extremely_active": "Extremely Active (Very hard exercise, physical job)"
                }[x]
            )
        
        submitted = st.form_submit_button("Calculate BMR & Daily Calories")
    
    if submitted:
        # Calculate BMR and daily calories
        bmr = calculate_bmr(age, weight, height, gender)
        daily_calories = calculate_daily_calories(bmr, activity_level)
        
        # Display results
        st.subheader("📊 Your Results")
        
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Basal Metabolic Rate (BMR)", f"{bmr:,} calories/day")
            st.info("BMR is the number of calories your body needs at rest to maintain basic life functions.")
        
        with col2:
            st.metric("Daily Calorie Needs", f"{daily_calories:,} calories/day")
            st.info("Total calories needed based on your activity level.")
        
        # Calorie breakdown
        st.subheader("🎯 Calorie Breakdown")
        
        # Create a simple chart showing the breakdown
        maintenance_calories = daily_calories
        weight_loss_calories = daily_calories - 500  # 500 calorie deficit for weight loss
        weight_gain_calories = daily_calories + 500  # 500 calorie surplus for weight gain
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Weight Loss", f"{weight_loss_calories:,} cal/day", "-500")
            st.caption("0.5 kg/week loss")
        
        with col2:
            st.metric("Maintenance", f"{maintenance_calories:,} cal/day", "0")
            st.caption("Maintain current weight")
        
        with col3:
            st.metric("Weight Gain", f"{weight_gain_calories:,} cal/day", "+500")
            st.caption("0.5 kg/week gain")
        
        # Tips and recommendations
        st.subheader("💡 Tips & Recommendations")
        
        st.write("""
        **For Weight Loss:**
        - Aim for a 500-750 calorie daily deficit
        - Focus on high-protein foods to preserve muscle mass
        - Include regular exercise for better results
        
        **For Weight Maintenance:**
        - Track your food intake to stay within your calorie target
        - Maintain a balanced diet with adequate protein, carbs, and fats
        - Regular exercise helps maintain muscle mass and metabolism
        
        **For Weight Gain:**
        - Aim for a 300-500 calorie daily surplus
        - Focus on strength training to build muscle
        - Include adequate protein in your diet
        """)
        
        # Add save profile button
        st.markdown("---")
        st.subheader("💾 Save Your Profile")
        st.write("Save your information to get personalized food suggestions on the Dashboard.")
        
        if st.button("Save Profile"):
            save_user_profile(age, weight, height, gender, activity_level)
            st.success("Profile saved! Your Dashboard will now show personalized calorie targets.")

elif page == "User Profile":
    st.header("👤 User Profile")
    
    # Load existing profile
    profile_df = load_user_profile()
    
    if not profile_df.empty:
        st.subheader("Current Profile")
        profile = profile_df.iloc[0]
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Age", f"{profile['age']} years")
            st.metric("Weight", f"{profile['weight']} kg")
            st.metric("Height", f"{profile['height']} cm")
        
        with col2:
            st.metric("Gender", profile['gender'])
            st.metric("Activity Level", profile['activity_level'].replace('_', ' ').title())
            st.metric("Daily Target", f"{profile['daily_calories']:,} calories")
        
        st.info(f"Profile last updated: {profile['last_updated']}")
        
        if st.button("Update Profile"):
            st.session_state.show_update_form = True
    
    # Show update form
    if not profile_df.empty and not st.session_state.get('show_update_form', False):
        st.subheader("Update Profile")
        st.write("Click 'Update Profile' above to modify your information.")
    else:
        st.subheader("Create/Update Profile")
        
        with st.form("user_profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                age = st.number_input("Age (years)", min_value=15, max_value=100, value=25)
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
                gender = st.selectbox("Gender", ["Male", "Female"])
            
            with col2:
                height = st.number_input("Height (cm)", min_value=100, max_value=250, value=170)
                activity_level = st.selectbox(
                    "Activity Level",
                    ["sedentary", "lightly_active", "moderately_active", "very_active", "extremely_active"],
                    format_func=lambda x: {
                        "sedentary": "Sedentary (Little or no exercise)",
                        "lightly_active": "Lightly Active (Light exercise 1-3 days/week)",
                        "moderately_active": "Moderately Active (Moderate exercise 3-5 days/week)",
                        "very_active": "Very Active (Hard exercise 6-7 days/week)",
                        "extremely_active": "Extremely Active (Very hard exercise, physical job)"
                    }[x]
                )
            
            submitted = st.form_submit_button("Save Profile")
        
        if submitted:
            save_user_profile(age, weight, height, gender, activity_level)
            st.success("Profile saved successfully!")
            st.session_state.show_update_form = False
            st.rerun()

elif page == "AI Coach":
    st.header("🤖 AI Nutrition Coach")
    
    if not AI_ENABLED:
        st.warning("⚠️ AI coaching is not available. Please set your OpenAI API key as an environment variable.")
        st.info("To enable AI coaching, set the OPENAI_API_KEY environment variable with your OpenAI API key.")
        st.code("export OPENAI_API_KEY='your_openai_api_key_here'")
    else:
        # AI Coach tabs
        tab1, tab2, tab3, tab4 = st.tabs(["Daily Briefing", "Chat with Coach", "Weekly Report", "Ask Questions"])
        
        with tab1:
            st.subheader("📊 Today's AI Analysis")
            
            # Get daily analysis
            daily_analysis = analyze_daily_nutrition()
            
            # Display basic metrics
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Calories Today", f"{daily_analysis['total_calories']:,}")
            with col2:
                st.metric("Meals Eaten", daily_analysis['meal_count'])
            with col3:
                st.metric("Balance Score", f"{daily_analysis['balance_score']:.0f}/100")
            
            # AI Insight
            if st.button("Get AI Analysis", key="daily_ai"):
                with st.spinner("Analyzing your nutrition..."):
                    ai_insight = get_ai_insight("daily_analysis")
                    st.success("AI Analysis Complete!")
                    st.write(ai_insight)
                    
                    # Save conversation
                    save_ai_conversation("user", "Daily nutrition analysis", ai_insight, daily_analysis)
        
        with tab2:
            st.subheader("💬 Chat with Your Coach")
            
            # Chat interface
            if "chat_messages" not in st.session_state:
                st.session_state.chat_messages = []
            
            # Display chat history
            for message in st.session_state.chat_messages:
                if message["role"] == "user":
                    st.write(f"**You:** {message['content']}")
                else:
                    st.write(f"**Coach:** {message['content']}")
            
            # Chat input
            user_message = st.text_input("Ask your nutrition coach anything:", key="chat_input")
            
            if st.button("Send", key="send_chat"):
                if user_message:
                    # Add user message to chat
                    st.session_state.chat_messages.append({"role": "user", "content": user_message})
                    
                    # Get AI response
                    with st.spinner("Coach is thinking..."):
                        ai_response = get_ai_insight("general_question", {"question": user_message})
                    
                    # Add AI response to chat
                    st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
                    
                    # Save conversation
                    save_ai_conversation("user", user_message, ai_response)
                    
                    st.rerun()
        
        with tab3:
            st.subheader("📈 Weekly Pattern Analysis")
            
            # Get weekly analysis
            weekly_analysis = analyze_weekly_patterns()
            
            if weekly_analysis:
                # Display weekly metrics
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Avg Daily Calories", f"{weekly_analysis['avg_daily_calories']:.0f}")
                with col2:
                    st.metric("Days Analyzed", weekly_analysis['total_days'])
                
                # Show patterns
                if weekly_analysis['consistent_patterns']:
                    st.subheader("🔍 Detected Patterns")
                    for pattern in weekly_analysis['consistent_patterns']:
                        st.info(pattern)
                
                # AI Weekly Insight
                if st.button("Get Weekly AI Analysis", key="weekly_ai"):
                    with st.spinner("Analyzing weekly patterns..."):
                        ai_insight = get_ai_insight("weekly_patterns")
                        st.success("Weekly Analysis Complete!")
                        st.write(ai_insight)
                        
                        # Save conversation
                        save_ai_conversation("user", "Weekly pattern analysis", ai_insight, weekly_analysis)
            else:
                st.info("Not enough data for weekly analysis. Log more foods to get insights!")
        
        with tab4:
            st.subheader("❓ Ask Questions")
            
            # Quick question buttons
            col1, col2 = st.columns(2)
            
            with col1:
                if st.button("Am I eating enough protein?"):
                    question = "Am I eating enough protein?"
                    with st.spinner("Analyzing..."):
                        response = get_ai_insight("general_question", {"question": question})
                    st.write(f"**Q:** {question}")
                    st.write(f"**A:** {response}")
                
                if st.button("When should I eat carbs?"):
                    question = "When should I eat carbs?"
                    with st.spinner("Analyzing..."):
                        response = get_ai_insight("general_question", {"question": question})
                    st.write(f"**Q:** {question}")
                    st.write(f"**A:** {response}")
            
            with col2:
                if st.button("Why am I not losing weight?"):
                    question = "Why am I not losing weight?"
                    with st.spinner("Analyzing..."):
                        response = get_ai_insight("general_question", {"question": question})
                    st.write(f"**Q:** {question}")
                    st.write(f"**A:** {response}")
                
                if st.button("How can I improve my nutrition?"):
                    question = "How can I improve my nutrition?"
                    with st.spinner("Analyzing..."):
                        response = get_ai_insight("general_question", {"question": question})
                    st.write(f"**Q:** {question}")
                    st.write(f"**A:** {response}")
            
            # Custom question
            st.subheader("Ask Your Own Question")
            custom_question = st.text_area("Type your nutrition question:")
            
            if st.button("Ask Coach", key="custom_question"):
                if custom_question:
                    with st.spinner("Coach is thinking..."):
                        response = get_ai_insight("general_question", {"question": custom_question})
                    st.write(f"**Q:** {custom_question}")
                    st.write(f"**A:** {response}")
                    
                    # Save conversation
                    save_ai_conversation("user", custom_question, response)
        
        # AI Usage Stats
        st.markdown("---")
        st.subheader("📊 AI Usage Statistics")
        
        usage = load_ai_usage()
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Total Requests", len(usage['requests']))
        with col2:
            st.metric("Total Tokens", f"{usage['total_tokens']:,}")
        with col3:
            st.metric("Total Cost", f"${usage['total_cost']:.4f}")

elif page == "Log Workout":
    st.header("💪 Log Your Workout")
    
    exercise = st.text_input("Exercise name")
    sets = st.number_input("Sets", min_value=1, value=3)
    reps = st.number_input("Reps", min_value=1, value=10)
    
    if st.button("Add Exercise"):
        st.success(f"Added {exercise}: {sets} sets x {reps} reps")

else:  # Progress page
    st.header("📈 Your Progress")
    st.info("Progress tracking coming soon!")
    