"""
FitTrack Pro - Enhanced AI-Powered Fitness Platform
A comprehensive fitness and nutrition tracking app with AI coaching
"""

import streamlit as st
import pandas as pd
from datetime import date, datetime, timedelta
import os
import time
import json
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import numpy as np

# Import our custom modules
from config import Config
from data_manager import data_manager
from models import UserProfile, Goal, FoodEntry, WorkoutEntry, NutritionSummary
from food_api import food_api_manager
from ai_coach import ai_coach

# Page configuration
st.set_page_config(
    page_title="FitTrack Pro - AI-Powered Fitness Platform",
    page_icon="💪",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for dark theme and modern styling
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
        color: #fafafa;
    }
    .stApp {
        background-color: #0e1117;
    }
    .stSidebar {
        background-color: #262730;
    }
    .stMetric {
        background-color: #262730;
        border-radius: 10px;
        padding: 10px;
    }
    .stButton > button {
        background-color: #00ff88;
        color: #000000;
        border-radius: 10px;
        border: none;
        padding: 10px 20px;
        font-weight: bold;
    }
    .stButton > button:hover {
        background-color: #00cc6a;
        color: #000000;
    }
    .success-message {
        background-color: #00ff88;
        color: #000000;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .warning-message {
        background-color: #ffaa00;
        color: #000000;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .info-message {
        background-color: #0088ff;
        color: #ffffff;
        padding: 10px;
        border-radius: 10px;
        margin: 10px 0;
    }
</style>
""", unsafe_allow_html=True)

# Initialize session state
if 'chat_messages' not in st.session_state:
    st.session_state.chat_messages = []

def main():
    """Main application function"""
    
    # App title
    st.title("FitTrack Pro")
    st.subheader("AI-Powered Fitness & Nutrition Platform 💪")
    
    # Check configuration status
    config_status = Config.validate_config()
    
    # Sidebar for navigation
    st.sidebar.title("Navigation")
    
    # Show configuration status
    with st.sidebar.expander("🔧 System Status", expanded=False):
        st.write("**OpenAI API:**", "✅ Configured" if config_status['openai_configured'] else "❌ Not Configured")
        st.write("**USDA API:**", "✅ Configured" if config_status['usda_configured'] else "⚠️ Using Demo Key")
        st.write("**Data Directory:**", "✅ Ready" if config_status['data_dir_exists'] else "❌ Missing")
        st.write("**Environment:**", config_status['environment'].title())
    
    # Navigation menu
    page = st.sidebar.selectbox(
        "Choose a page",
        [
            "📊 Dashboard",
            "🍎 Log Food",
            "💪 Log Workout", 
            "📋 Food History",
            "🏃‍♂️ Workout History",
            "👤 User Profile",
            "🎯 Goal Setting",
            "🤖 AI Coach",
            "📈 Progress Analytics",
            "⚙️ Settings"
        ]
    )
    
    # Route to appropriate page
    if page == "📊 Dashboard":
        show_dashboard()
    elif page == "🍎 Log Food":
        show_log_food()
    elif page == "💪 Log Workout":
        show_log_workout()
    elif page == "📋 Food History":
        show_food_history()
    elif page == "🏃‍♂️ Workout History":
        show_workout_history()
    elif page == "👤 User Profile":
        show_user_profile()
    elif page == "🎯 Goal Setting":
        show_goal_setting()
    elif page == "🤖 AI Coach":
        show_ai_coach()
    elif page == "📈 Progress Analytics":
        show_progress_analytics()
    elif page == "⚙️ Settings":
        show_settings()

def show_dashboard():
    """Display the main dashboard"""
    st.header("📊 Today's Overview")
    
    # Get user profile and data
    profile = data_manager.load_user_profile()
    today_food = data_manager.get_today_food_summary()
    today_workout = data_manager.get_today_workout_summary()
    
    # Calculate metrics
    calories_eaten = today_food.total_calories
    calories_burned = today_workout['total_calories_burned']
    net_calories = calories_eaten - calories_burned
    
    # Get user's calorie target
    if profile:
        daily_target = profile.calculate_tdee()
        remaining_calories = daily_target - calories_eaten
    else:
        daily_target = 2000  # Default
        remaining_calories = daily_target - calories_eaten
    
    # Main metrics
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Calories Eaten",
            f"{calories_eaten:,}",
            f"{calories_eaten - (today_food.total_calories - 100):,}" if today_food.total_calories > 100 else "0"
        )
    
    with col2:
        st.metric(
            "Calories Burned",
            f"{calories_burned:,}",
            f"{calories_burned - (today_workout['total_calories_burned'] - 50):,}" if today_workout['total_calories_burned'] > 50 else "0"
        )
    
    with col3:
        st.metric(
            "Net Calories",
            f"{net_calories:,}",
            f"{net_calories - (net_calories - 100):,}" if net_calories > 100 else "0"
        )
    
    with col4:
        st.metric(
            "Remaining",
            f"{remaining_calories:,}",
            f"{remaining_calories - (remaining_calories - 200):,}" if remaining_calories > 200 else "0"
        )
    
    # Progress bars
    st.subheader("📈 Daily Progress")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Calorie progress
        if daily_target > 0:
            calorie_progress = min(calories_eaten / daily_target, 1.0)
            st.progress(calorie_progress)
            st.write(f"Calories: {calories_eaten:,} / {daily_target:,} ({calorie_progress:.1%})")
    
    with col2:
        # Macro breakdown
        if calories_eaten > 0:
            protein_pct = (today_food.total_protein * 4) / calories_eaten
            carbs_pct = (today_food.total_carbs * 4) / calories_eaten
            fat_pct = (today_food.total_fat * 9) / calories_eaten
            
            st.write("**Macro Breakdown:**")
            st.write(f"Protein: {protein_pct:.1%}")
            st.write(f"Carbs: {carbs_pct:.1%}")
            st.write(f"Fat: {fat_pct:.1%}")
    
    # Smart Food Suggestions
    st.markdown("---")
    st.subheader("🍽️ Smart Food Suggestions")
    
    if profile:
        meal_category = food_api_manager.get_meal_category_by_time()
        suggestions = food_api_manager.get_food_suggestions(remaining_calories, meal_category)
        
        if suggestions:
            st.write(f"**{meal_category.title()} suggestions for {remaining_calories:.0f} remaining calories:**")
            
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
                    
                    st.markdown("<hr style='margin: 0.5rem 0; border-color: #333333;'>", unsafe_allow_html=True)
            
            # Quick add buttons
            st.subheader("Quick Add")
            cols = st.columns(len(suggestions))
            for i, (col, food) in enumerate(zip(cols, suggestions)):
                with col:
                    if st.button(f"Add {food['name']}", key=f"quick_add_{i}"):
                        food_entry = FoodEntry(
                            date=date.today().strftime('%Y-%m-%d'),
                            food_name=food['name'],
                            calories=food['calories'],
                            protein=food['protein'],
                            carbs=food['carbs'],
                            fat=food['fat'],
                            meal_type=meal_category
                        )
                        data_manager.save_food_entry(food_entry)
                        st.success(f"Added {food['name']}!")
                        st.rerun()
    else:
        st.warning("⚠️ Please set up your user profile to get personalized food suggestions.")
    
    # AI Insights
    if ai_coach.is_available():
        st.markdown("---")
        st.subheader("🤖 AI Coach Insights")
        
        if st.button("Get Today's AI Analysis", key="dashboard_ai"):
            with st.spinner("AI Coach is analyzing..."):
                ai_insight = ai_coach.generate_daily_analysis()
                st.success("AI Analysis Complete!")
                st.info(ai_insight)
                
                # Save conversation
                ai_coach.save_conversation("user", "Dashboard daily analysis", ai_insight)
        else:
            st.info("Click the button above to get personalized AI insights about your nutrition today!")
    else:
        st.warning("⚠️ AI coaching is not available. Please configure your OpenAI API key in settings.")

def show_log_food():
    """Display the food logging page"""
    st.header("🍎 Log Your Food")
    
    # Food search section
    st.subheader("🔍 Search for Foods")
    search_query = st.text_input("Search for a food item", placeholder="e.g., apple, chicken breast, oatmeal")
    
    if search_query and len(search_query) >= 2:
        with st.spinner("Searching for foods..."):
            search_results = food_api_manager.search_foods(search_query, max_results=8)
        
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
                        protein = food['nutrients'].get('protein', 0)
                        carbs = food['nutrients'].get('carbs', 0)
                        fat = food['nutrients'].get('fat', 0)
                        
                        food_entry = FoodEntry(
                            date=date.today().strftime('%Y-%m-%d'),
                            food_name=food['name'],
                            calories=calories,
                            protein=protein,
                            carbs=carbs,
                            fat=fat
                        )
                        
                        data_manager.save_food_entry(food_entry)
                        st.success(f"Added {food['name']} ({calories} calories)")
                        st.balloons()
                        st.rerun()
        else:
            st.warning("No foods found. Try a different search term.")
    
    # Manual food entry section
    st.markdown("---")
    st.subheader("✏️ Manual Food Entry")
    
    with st.form("manual_food_entry"):
        col1, col2 = st.columns(2)
        
        with col1:
            food_name = st.text_input("Food item name")
            calories = st.number_input("Calories", min_value=0, value=100)
            protein = st.number_input("Protein (g)", min_value=0.0, value=0.0, step=0.1)
        
        with col2:
            carbs = st.number_input("Carbohydrates (g)", min_value=0.0, value=0.0, step=0.1)
            fat = st.number_input("Fat (g)", min_value=0.0, value=0.0, step=0.1)
            meal_type = st.selectbox("Meal Type", ["", "breakfast", "lunch", "dinner", "snack"])
        
        notes = st.text_area("Notes (optional)")
        
        submitted = st.form_submit_button("Add Food")
        
        if submitted:
            if food_name.strip():
                food_entry = FoodEntry(
                    date=date.today().strftime('%Y-%m-%d'),
                    food_name=food_name,
                    calories=calories,
                    protein=protein,
                    carbs=carbs,
                    fat=fat,
                    meal_type=meal_type,
                    notes=notes
                )
                
                data_manager.save_food_entry(food_entry)
                st.success(f"Added {food_name} ({calories} calories)")
                st.balloons()
            else:
                st.error("Please enter a food name")

def show_log_workout():
    """Display the workout logging page"""
    st.header("💪 Log Your Workout")
    
    # Workout entry form
    with st.form("workout_entry"):
        col1, col2 = st.columns(2)
        
        with col1:
            workout_type = st.selectbox(
                "Workout Type",
                ["cardio", "strength", "sports", "yoga", "other"]
            )
            exercise_name = st.text_input("Exercise/Activity Name")
            duration_minutes = st.number_input("Duration (minutes)", min_value=1, value=30)
        
        with col2:
            intensity = st.selectbox("Intensity", ["light", "moderate", "vigorous"])
            sets = st.number_input("Sets (for strength)", min_value=0, value=0)
            reps = st.number_input("Reps per set", min_value=0, value=0)
        
        weight = st.number_input("Weight (kg)", min_value=0.0, value=0.0, step=0.5)
        distance = st.number_input("Distance (km)", min_value=0.0, value=0.0, step=0.1)
        notes = st.text_area("Notes (optional)")
        
        submitted = st.form_submit_button("Log Workout")
        
        if submitted:
            if exercise_name.strip():
                # Calculate calories burned
                profile = data_manager.load_user_profile()
                user_weight = profile.weight if profile else 70.0
                
                workout_entry = WorkoutEntry(
                    date=date.today().strftime('%Y-%m-%d'),
                    workout_type=workout_type,
                    exercise_name=exercise_name,
                    duration_minutes=duration_minutes,
                    intensity=intensity,
                    sets=sets if sets > 0 else None,
                    reps=reps if reps > 0 else None,
                    weight=weight if weight > 0 else None,
                    distance=distance if distance > 0 else None,
                    notes=notes
                )
                
                # Calculate calories burned
                calories_burned = workout_entry.calculate_calories_burned(user_weight)
                workout_entry.calories_burned = calories_burned
                
                data_manager.save_workout_entry(workout_entry)
                st.success(f"Logged {exercise_name} - {calories_burned:.0f} calories burned!")
                st.balloons()
            else:
                st.error("Please enter an exercise name")

def show_food_history():
    """Display food log history"""
    st.header("📋 Food Log History")
    
    df = data_manager.load_food_logs()
    
    if not df.empty:
        # Show today's foods
        today = date.today().strftime('%Y-%m-%d')
        today_foods = df[df['date'] == today]
        
        if not today_foods.empty:
            st.subheader("🍽️ Today's Foods")
            st.dataframe(today_foods[['food_name', 'calories', 'protein', 'carbs', 'fat', 'timestamp']], hide_index=True)
            
            total_today = today_foods['calories'].sum()
            st.info(f"Total calories today: {total_today:,}")
        
        # Show all foods with date filter
        st.subheader("📅 All Food Logs")
        
        # Date filter
        unique_dates = sorted(df['date'].unique(), reverse=True)
        selected_date = st.selectbox("Select date to view:", unique_dates)
        
        if selected_date:
            filtered_df = df[df['date'] == selected_date]
            st.dataframe(filtered_df[['food_name', 'calories', 'protein', 'carbs', 'fat', 'timestamp']], hide_index=True)
            
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

def show_workout_history():
    """Display workout log history"""
    st.header("🏃‍♂️ Workout History")
    
    df = data_manager.load_workout_logs()
    
    if not df.empty:
        # Show today's workouts
        today = date.today().strftime('%Y-%m-%d')
        today_workouts = df[df['date'] == today]
        
        if not today_workouts.empty:
            st.subheader("💪 Today's Workouts")
            st.dataframe(today_workouts[['exercise_name', 'workout_type', 'duration_minutes', 'calories_burned', 'timestamp']], hide_index=True)
            
            total_calories_burned = today_workouts['calories_burned'].sum()
            st.info(f"Total calories burned today: {total_calories_burned:.0f}")
        
        # Show all workouts with date filter
        st.subheader("📅 All Workout Logs")
        
        # Date filter
        unique_dates = sorted(df['date'].unique(), reverse=True)
        selected_date = st.selectbox("Select date to view:", unique_dates, key="workout_date")
        
        if selected_date:
            filtered_df = df[df['date'] == selected_date]
            st.dataframe(filtered_df[['exercise_name', 'workout_type', 'duration_minutes', 'calories_burned', 'timestamp']], hide_index=True)
            
            total_calories_burned = filtered_df['calories_burned'].sum()
            st.info(f"Total calories burned on {selected_date}: {total_calories_burned:.0f}")
        
        # Summary statistics
        st.subheader("📊 Summary")
        col1, col2 = st.columns(2)
        
        with col1:
            st.metric("Total Workouts", len(df))
            st.metric("Total Calories Burned", f"{df['calories_burned'].sum():.0f}")
        
        with col2:
            st.metric("Average Duration", f"{df['duration_minutes'].mean():.0f} min")
            st.metric("Workout Days", len(df['date'].unique()))
    
    else:
        st.info("No workout logs yet. Start logging your workouts!")

def show_user_profile():
    """Display user profile management"""
    st.header("👤 User Profile")
    
    # Load existing profile
    profile = data_manager.load_user_profile()
    
    if profile:
        st.subheader("Current Profile")
        
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Name", profile.name or "Not set")
            st.metric("Age", f"{profile.age} years")
            st.metric("Weight", f"{profile.weight} kg")
            st.metric("Height", f"{profile.height} cm")
        
        with col2:
            st.metric("Gender", profile.gender)
            st.metric("Activity Level", profile.activity_level.replace('_', ' ').title())
            st.metric("BMR", f"{profile.calculate_bmr():.0f} cal/day")
            st.metric("TDEE", f"{profile.calculate_tdee():.0f} cal/day")
        
        st.info(f"Profile last updated: {profile.last_updated}")
        
        if st.button("Update Profile"):
            st.session_state.show_update_form = True
    
    # Show update form
    if not profile or st.session_state.get('show_update_form', False):
        st.subheader("Create/Update Profile")
        
        with st.form("user_profile_form"):
            col1, col2 = st.columns(2)
            
            with col1:
                name = st.text_input("Name", value=profile.name if profile else "")
                age = st.number_input("Age (years)", min_value=15, max_value=100, value=profile.age if profile else 25)
                weight = st.number_input("Weight (kg)", min_value=30.0, max_value=300.0, value=profile.weight if profile else 70.0, step=0.1)
                gender = st.selectbox("Gender", ["Male", "Female"], index=0 if not profile or profile.gender == "Male" else 1)
            
            with col2:
                height = st.number_input("Height (cm)", min_value=100, max_value=250, value=profile.height if profile else 170)
                activity_level = st.selectbox(
                    "Activity Level",
                    list(Config.ACTIVITY_MULTIPLIERS.keys()),
                    index=list(Config.ACTIVITY_MULTIPLIERS.keys()).index(profile.activity_level) if profile else 2,
                    format_func=lambda x: x.replace('_', ' ').title()
                )
                bmr_formula = st.selectbox(
                    "BMR Formula",
                    list(Config.BMR_FORMULAS.keys()),
                    index=list(Config.BMR_FORMULAS.keys()).index(profile.bmr_formula) if profile else 0,
                    format_func=lambda x: Config.BMR_FORMULAS[x]
                )
                units = st.selectbox(
                    "Measurement Units",
                    list(Config.UNITS.keys()),
                    index=list(Config.UNITS.keys()).index(profile.units) if profile else 0,
                    format_func=lambda x: Config.UNITS[x]
                )
            
            # Dietary preferences
            st.subheader("Dietary Preferences")
            col1, col2 = st.columns(2)
            
            with col1:
                dietary_restrictions = st.multiselect(
                    "Dietary Restrictions",
                    ["None", "Vegetarian", "Vegan", "Gluten-Free", "Dairy-Free", "Keto", "Paleo"],
                    default=profile.dietary_restrictions if profile else []
                )
            
            with col2:
                favorite_foods = st.text_area(
                    "Favorite Foods (one per line)",
                    value="\n".join(profile.favorite_foods) if profile else "",
                    help="Enter your favorite foods, one per line"
                )
            
            # AI coaching preferences
            st.subheader("AI Coaching Preferences")
            ai_coaching_style = st.selectbox(
                "Coaching Style",
                ["motivational", "scientific", "casual", "strict"],
                index=["motivational", "scientific", "casual", "strict"].index(profile.ai_coaching_style) if profile else 0,
                format_func=lambda x: x.title()
            )
            
            submitted = st.form_submit_button("Save Profile")
        
        if submitted:
            # Create new profile
            new_profile = UserProfile(
                name=name,
                age=age,
                weight=weight,
                height=height,
                gender=gender,
                activity_level=activity_level,
                bmr_formula=bmr_formula,
                units=units,
                dietary_restrictions=dietary_restrictions,
                favorite_foods=favorite_foods.split('\n') if favorite_foods else [],
                ai_coaching_style=ai_coaching_style
            )
            
            data_manager.save_user_profile(new_profile)
            st.success("Profile saved successfully!")
            st.session_state.show_update_form = False
            st.rerun()

def show_goal_setting():
    """Display goal setting and tracking"""
    st.header("🎯 Goal Setting & Tracking")
    
    # Load existing goals
    goals = data_manager.load_goals()
    
    if goals:
        st.subheader("Current Goals")
        
        for i, goal in enumerate(goals):
            with st.expander(f"{goal.goal_type.replace('_', ' ').title()} - {goal.target_value} kg"):
                col1, col2 = st.columns(2)
                
                with col1:
                    st.write(f"**Start Weight:** {goal.start_value} kg")
                    st.write(f"**Current Weight:** {goal.current_value} kg")
                    st.write(f"**Target Weight:** {goal.target_value} kg")
                    st.write(f"**Timeline:** {goal.timeline_weeks} weeks")
                
                with col2:
                    progress = goal.calculate_progress_percentage()
                    st.progress(progress / 100)
                    st.write(f"**Progress:** {progress:.1f}%")
                    st.write(f"**Status:** {goal.status.title()}")
                    st.write(f"**Weekly Rate:** {goal.weekly_change_rate} kg/week")
                
                # Update current weight
                new_weight = st.number_input(
                    "Update Current Weight (kg)",
                    min_value=30.0,
                    max_value=300.0,
                    value=goal.current_value,
                    step=0.1,
                    key=f"update_weight_{i}"
                )
                
                if st.button("Update Weight", key=f"update_weight_btn_{i}"):
                    data_manager.update_goal_progress(i, new_weight)
                    st.success("Weight updated!")
                    st.rerun()
    
    # Create new goal
    st.subheader("Create New Goal")
    
    with st.form("new_goal_form"):
        col1, col2 = st.columns(2)
        
        with col1:
            goal_type = st.selectbox(
                "Goal Type",
                list(Config.GOAL_TYPES.keys()),
                format_func=lambda x: Config.GOAL_TYPES[x]
            )
            current_weight = st.number_input("Current Weight (kg)", min_value=30.0, max_value=300.0, value=70.0, step=0.1)
            target_weight = st.number_input("Target Weight (kg)", min_value=30.0, max_value=300.0, value=65.0, step=0.1)
        
        with col2:
            timeline_weeks = st.number_input("Timeline (weeks)", min_value=1, max_value=52, value=12)
            weekly_change_rate = st.number_input("Weekly Change Rate (kg/week)", min_value=0.1, max_value=2.0, value=0.5, step=0.1)
            notes = st.text_area("Notes (optional)")
        
        submitted = st.form_submit_button("Create Goal")
        
        if submitted:
            new_goal = Goal(
                goal_type=goal_type,
                target_value=target_weight,
                current_value=current_weight,
                start_value=current_weight,
                start_date=date.today().strftime('%Y-%m-%d'),
                target_date=(date.today() + timedelta(weeks=timeline_weeks)).strftime('%Y-%m-%d'),
                timeline_weeks=timeline_weeks,
                weekly_change_rate=weekly_change_rate,
                notes=notes
            )
            
            data_manager.save_goal(new_goal)
            st.success("Goal created successfully!")
            st.rerun()

def show_ai_coach():
    """Display AI coaching interface"""
    st.header("🤖 AI Nutrition Coach")
    
    if not ai_coach.is_available():
        st.warning("⚠️ AI coaching is not available. Please set your OpenAI API key in settings.")
        st.info("To enable AI coaching, set the OPENAI_API_KEY environment variable with your OpenAI API key.")
        st.code("export OPENAI_API_KEY='your-api-key-here'")
        return
    
    # AI Coach tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Daily Briefing", "Chat with Coach", "Weekly Report", "Ask Questions"])
    
    with tab1:
        st.subheader("📊 Today's AI Analysis")
        
        # Get daily analysis
        if st.button("Get AI Analysis", key="daily_ai"):
            with st.spinner("Analyzing your nutrition..."):
                ai_insight = ai_coach.generate_daily_analysis()
                st.success("AI Analysis Complete!")
                st.write(ai_insight)
                
                # Save conversation
                ai_coach.save_conversation("user", "Daily nutrition analysis", ai_insight)
        else:
            st.info("Click the button above to get personalized AI insights about your nutrition today!")
    
    with tab2:
        st.subheader("💬 Chat with Your Coach")
        
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
                    ai_response = ai_coach.chat_with_coach(user_message)
                
                # Add AI response to chat
                st.session_state.chat_messages.append({"role": "assistant", "content": ai_response})
                
                # Save conversation
                ai_coach.save_conversation("user", user_message, ai_response)
                
                st.rerun()
    
    with tab3:
        st.subheader("📈 Weekly Pattern Analysis")
        
        # Get weekly analysis
        if st.button("Get Weekly AI Analysis", key="weekly_ai"):
            with st.spinner("Analyzing weekly patterns..."):
                ai_insight = ai_coach.generate_weekly_report()
                st.success("Weekly Analysis Complete!")
                st.write(ai_insight)
                
                # Save conversation
                ai_coach.save_conversation("user", "Weekly pattern analysis", ai_insight)
        else:
            st.info("Click the button above to get comprehensive weekly analysis and recommendations!")
    
    with tab4:
        st.subheader("❓ Ask Questions")
        
        # Quick question buttons
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("Am I eating enough protein?"):
                question = "Am I eating enough protein?"
                with st.spinner("Analyzing..."):
                    response = ai_coach.answer_question(question)
                st.write(f"**Q:** {question}")
                st.write(f"**A:** {response}")
            
            if st.button("When should I eat carbs?"):
                question = "When should I eat carbs?"
                with st.spinner("Analyzing..."):
                    response = ai_coach.answer_question(question)
                st.write(f"**Q:** {question}")
                st.write(f"**A:** {response}")
        
        with col2:
            if st.button("Why am I not losing weight?"):
                question = "Why am I not losing weight?"
                with st.spinner("Analyzing..."):
                    response = ai_coach.answer_question(question)
                st.write(f"**Q:** {question}")
                st.write(f"**A:** {response}")
            
            if st.button("How can I improve my nutrition?"):
                question = "How can I improve my nutrition?"
                with st.spinner("Analyzing..."):
                    response = ai_coach.answer_question(question)
                st.write(f"**Q:** {question}")
                st.write(f"**A:** {response}")
        
        # Custom question
        st.subheader("Ask Your Own Question")
        custom_question = st.text_area("Type your nutrition question:")
        
        if st.button("Ask Coach", key="custom_question"):
            if custom_question:
                with st.spinner("Coach is thinking..."):
                    response = ai_coach.answer_question(custom_question)
                st.write(f"**Q:** {custom_question}")
                st.write(f"**A:** {response}")
                
                # Save conversation
                ai_coach.save_conversation("user", custom_question, response)
    
    # AI Usage Stats
    st.markdown("---")
    st.subheader("📊 AI Usage Statistics")
    
    usage = ai_coach.get_usage_stats()
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total Requests", len(usage['requests']))
    with col2:
        st.metric("Total Tokens", f"{usage['total_tokens']:,}")
    with col3:
        st.metric("Total Cost", f"${usage['total_cost']:.4f}")

def show_progress_analytics():
    """Display progress and analytics"""
    st.header("📊 Progress & Analytics")
    
    # Load data
    profile = data_manager.load_user_profile()
    food_df = data_manager.load_food_logs()
    workout_df = data_manager.load_workout_logs()
    goals = data_manager.load_goals()
    
    if food_df.empty and workout_df.empty:
        st.info("No data available for analytics. Start logging your food and workouts!")
        return
    
    # Analytics tabs
    tab1, tab2, tab3, tab4 = st.tabs(["Nutrition Trends", "Workout Trends", "Goal Progress", "AI Pattern Analysis"])
    
    with tab1:
        st.subheader("🍎 Nutrition Trends")
        
        if not food_df.empty:
            # Daily calorie trends
            daily_calories = food_df.groupby('date')['calories'].sum().reset_index()
            daily_calories['date'] = pd.to_datetime(daily_calories['date'])
            
            fig = px.line(daily_calories, x='date', y='calories', title='Daily Calorie Intake')
            st.plotly_chart(fig, use_container_width=True)
            
            # Macro breakdown over time
            if 'protein' in food_df.columns and 'carbs' in food_df.columns and 'fat' in food_df.columns:
                daily_macros = food_df.groupby('date').agg({
                    'protein': 'sum',
                    'carbs': 'sum',
                    'fat': 'sum'
                }).reset_index()
                daily_macros['date'] = pd.to_datetime(daily_macros['date'])
                
                fig = px.line(daily_macros, x='date', y=['protein', 'carbs', 'fat'], 
                             title='Daily Macro Intake', labels={'value': 'grams', 'variable': 'Macro'})
                st.plotly_chart(fig, use_container_width=True)
    
    with tab2:
        st.subheader("💪 Workout Trends")
        
        if not workout_df.empty:
            # Daily workout duration
            daily_duration = workout_df.groupby('date')['duration_minutes'].sum().reset_index()
            daily_duration['date'] = pd.to_datetime(daily_duration['date'])
            
            fig = px.bar(daily_duration, x='date', y='duration_minutes', title='Daily Workout Duration')
            st.plotly_chart(fig, use_container_width=True)
            
            # Calories burned over time
            if 'calories_burned' in workout_df.columns:
                daily_calories_burned = workout_df.groupby('date')['calories_burned'].sum().reset_index()
                daily_calories_burned['date'] = pd.to_datetime(daily_calories_burned['date'])
                
                fig = px.line(daily_calories_burned, x='date', y='calories_burned', title='Daily Calories Burned')
                st.plotly_chart(fig, use_container_width=True)
    
    with tab3:
        st.subheader("🎯 Goal Progress")
        
        if goals:
            for goal in goals:
                st.write(f"**{goal.goal_type.replace('_', ' ').title()} Goal:**")
                progress = goal.calculate_progress_percentage()
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric("Start Weight", f"{goal.start_value} kg")
                    st.metric("Current Weight", f"{goal.current_value} kg")
                    st.metric("Target Weight", f"{goal.target_value} kg")
                
                with col2:
                    st.metric("Progress", f"{progress:.1f}%")
                    st.metric("Weeks Remaining", max(0, goal.timeline_weeks - ((date.today() - datetime.strptime(goal.start_date, '%Y-%m-%d').date()).days // 7)))
                    st.metric("Weekly Rate", f"{goal.weekly_change_rate} kg/week")
                
                st.progress(progress / 100)
                st.markdown("---")
        else:
            st.info("No goals set. Create a goal in the Goal Setting page to track progress.")
    
    with tab4:
        st.subheader("🤖 AI Pattern Analysis")
        
        if ai_coach.is_available():
            if st.button("Analyze Patterns", key="analyze_patterns"):
                with st.spinner("AI is analyzing your patterns..."):
                    pattern_analysis = ai_coach.analyze_patterns()
                    st.write(pattern_analysis)
                    
                    # Save conversation
                    ai_coach.save_conversation("user", "Pattern analysis request", pattern_analysis)
        else:
            st.warning("AI coaching is not available. Please configure your OpenAI API key.")

def show_settings():
    """Display settings and configuration"""
    st.header("⚙️ Settings")
    
    # Configuration status
    st.subheader("🔧 System Configuration")
    
    config_status = Config.validate_config()
    
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("**OpenAI API:**", "✅ Configured" if config_status['openai_configured'] else "❌ Not Configured")
        st.write("**USDA API:**", "✅ Configured" if config_status['usda_configured'] else "⚠️ Using Demo Key")
        st.write("**Environment:**", config_status['environment'].title())
    
    with col2:
        st.write("**Data Directory:**", "✅ Ready" if config_status['data_dir_exists'] else "❌ Missing")
        st.write("**Debug Mode:**", "✅ Enabled" if config_status['debug_mode'] else "❌ Disabled")
        st.write("**AI Model:**", Config.AI_MODEL)
    
    # API Key Configuration
    st.subheader("🔑 API Key Configuration")
    
    st.info("""
    To configure API keys, create a `.env` file in the project root with the following variables:
    
    ```
    OPENAI_API_KEY=your_openai_api_key_here
    USDA_API_KEY=your_usda_api_key_here
    ```
    
    You can get your API keys from:
    - OpenAI: https://platform.openai.com/api-keys
    - USDA FoodData Central: https://fdc.nal.usda.gov/api-key-signup.html
    """)
    
    # Data Management
    st.subheader("💾 Data Management")
    
    col1, col2 = st.columns(2)
    
    with col1:
        if st.button("Export All Data"):
            export_data = data_manager.export_user_data()
            st.download_button(
                label="Download Data",
                data=json.dumps(export_data, indent=2),
                file_name=f"fittrack_pro_data_{date.today().strftime('%Y%m%d')}.json",
                mime="application/json"
            )
    
    with col2:
        if st.button("Clear All Data"):
            if st.checkbox("I understand this will delete all my data permanently"):
                # Clear all data files
                for file_path in [Config.FOOD_LOG_FILE, Config.WORKOUT_LOG_FILE, Config.USER_PROFILE_FILE, 
                                Config.GOAL_TRACKING_FILE, Config.AI_CONVERSATION_FILE, Config.AI_USAGE_FILE]:
                    if os.path.exists(file_path):
                        os.remove(file_path)
                st.success("All data cleared!")
                st.rerun()
    
    # Cache Management
    st.subheader("🗄️ Cache Management")
    
    if st.button("Clear Food Cache"):
        if os.path.exists(Config.FOOD_CACHE_FILE):
            os.remove(Config.FOOD_CACHE_FILE)
            st.success("Food cache cleared!")
    
    # Usage Statistics
    st.subheader("📊 Usage Statistics")
    
    usage = ai_coach.get_usage_stats()
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Total AI Requests", len(usage['requests']))
    with col2:
        st.metric("Total Tokens Used", f"{usage['total_tokens']:,}")
    with col3:
        st.metric("Total Cost", f"${usage['total_cost']:.4f}")
    
    # Recent requests
    if usage['requests']:
        st.subheader("Recent AI Requests")
        recent_requests = usage['requests'][-10:]  # Last 10 requests
        
        for req in recent_requests:
            st.write(f"**{req['timestamp']}** - {req['tokens']} tokens (${req['cost']:.4f})")

if __name__ == "__main__":
    main() 