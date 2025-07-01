"""
Food API integration module for FitTrack Pro
Handles USDA FoodData Central API, fallback databases, and food search functionality
"""

import requests
import json
import time
import hashlib
import os
from datetime import datetime
from typing import Dict, List, Optional, Any
import streamlit as st
from config import Config

# Fallback food database (used when API fails)
FALLBACK_FOOD_DATABASE = {
    'oatmeal': {'calories': 150, 'protein': 6, 'carbs': 27, 'fat': 3, 'category': 'breakfast'},
    'eggs': {'calories': 70, 'protein': 6, 'carbs': 1, 'fat': 5, 'category': 'breakfast'},
    'chicken_breast': {'calories': 165, 'protein': 31, 'carbs': 0, 'fat': 3.6, 'category': 'lunch'},
    'salmon': {'calories': 208, 'protein': 25, 'carbs': 0, 'fat': 12, 'category': 'lunch'},
    'brown_rice': {'calories': 110, 'protein': 2.5, 'carbs': 23, 'fat': 0.9, 'category': 'lunch'},
    'apple': {'calories': 95, 'protein': 0.5, 'carbs': 25, 'fat': 0.3, 'category': 'snack'},
    'almonds': {'calories': 164, 'protein': 6, 'carbs': 6, 'fat': 14, 'category': 'snack'},
    'banana': {'calories': 105, 'protein': 1, 'carbs': 27, 'fat': 0.4, 'category': 'breakfast'},
    'greek_yogurt': {'calories': 130, 'protein': 22, 'carbs': 9, 'fat': 0.5, 'category': 'breakfast'},
    'sweet_potato': {'calories': 103, 'protein': 2, 'carbs': 24, 'fat': 0.2, 'category': 'lunch'},
    'broccoli': {'calories': 55, 'protein': 3.7, 'carbs': 11, 'fat': 0.6, 'category': 'lunch'},
    'quinoa': {'calories': 120, 'protein': 4.4, 'carbs': 22, 'fat': 1.9, 'category': 'lunch'},
    'tuna': {'calories': 144, 'protein': 30, 'carbs': 0, 'fat': 1, 'category': 'lunch'},
    'spinach': {'calories': 23, 'protein': 2.9, 'carbs': 3.6, 'fat': 0.4, 'category': 'lunch'},
    'avocado': {'calories': 160, 'protein': 2, 'carbs': 9, 'fat': 15, 'category': 'snack'},
    'blueberries': {'calories': 85, 'protein': 1.1, 'carbs': 21, 'fat': 0.5, 'category': 'snack'},
    'peanut_butter': {'calories': 188, 'protein': 8, 'carbs': 6, 'fat': 16, 'category': 'snack'},
    'whole_wheat_bread': {'calories': 69, 'protein': 3.6, 'carbs': 12, 'fat': 1.1, 'category': 'breakfast'},
    'milk': {'calories': 103, 'protein': 8, 'carbs': 12, 'fat': 2.4, 'category': 'breakfast'},
    'beef_steak': {'calories': 250, 'protein': 26, 'carbs': 0, 'fat': 15, 'category': 'dinner'}
}

class FoodAPIManager:
    """Manages food API interactions and caching"""
    
    def __init__(self):
        self.cache = self.load_cache()
        self.last_api_call = 0
        self.rate_limit_delay = 1.0  # 1 second between API calls
    
    def load_cache(self) -> Dict[str, Any]:
        """Load cached food data from JSON file"""
        try:
            if os.path.exists(Config.FOOD_CACHE_FILE):
                with open(Config.FOOD_CACHE_FILE, 'r') as f:
                    return json.load(f)
        except Exception as e:
            st.error(f"Error loading food cache: {e}")
        return {}
    
    def save_cache(self):
        """Save food cache to JSON file"""
        try:
            with open(Config.FOOD_CACHE_FILE, 'w') as f:
                json.dump(self.cache, f, indent=2)
        except Exception as e:
            st.error(f"Error saving food cache: {e}")
    
    def get_cache_key(self, query: str, search_type: str = "search") -> str:
        """Generate cache key for query"""
        return f"{search_type}_{hashlib.md5(query.lower().encode()).hexdigest()}"
    
    def is_cache_valid(self, cache_entry: Dict[str, Any]) -> bool:
        """Check if cache entry is still valid"""
        cache_age = time.time() - cache_entry.get('timestamp', 0)
        return cache_age < (Config.CACHE_DURATION_HOURS * 3600)
    
    def search_usda_foods(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for foods using USDA FoodData Central API"""
        cache_key = self.get_cache_key(query, "search")
        
        # Check cache first
        if cache_key in self.cache and self.is_cache_valid(self.cache[cache_key]):
            return self.cache[cache_key]['data']
        
        # Rate limiting
        time_since_last_call = time.time() - self.last_api_call
        if time_since_last_call < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_call)
        
        try:
            # API search
            search_url = f"{Config.USDA_BASE_URL}/foods/search"
            params = {
                'api_key': Config.USDA_API_KEY,
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
                    elif nutrient_id == 1079:  # Fiber
                        food_info['nutrients']['fiber'] = value
                    elif nutrient_id == 1093:  # Sodium
                        food_info['nutrients']['sodium'] = value
                
                foods.append(food_info)
            
            # Cache the results
            self.cache[cache_key] = {
                'data': foods,
                'timestamp': time.time()
            }
            self.save_cache()
            
            self.last_api_call = time.time()
            return foods
            
        except Exception as e:
            st.error(f"USDA API search failed: {e}")
            return []
    
    def get_usda_food_details(self, fdc_id: str) -> Optional[Dict[str, Any]]:
        """Get detailed nutrition information for a specific food"""
        cache_key = self.get_cache_key(fdc_id, "details")
        
        # Check cache first
        if cache_key in self.cache and self.is_cache_valid(self.cache[cache_key]):
            return self.cache[cache_key]['data']
        
        # Rate limiting
        time_since_last_call = time.time() - self.last_api_call
        if time_since_last_call < self.rate_limit_delay:
            time.sleep(self.rate_limit_delay - time_since_last_call)
        
        try:
            # API call for detailed info
            detail_url = f"{Config.USDA_BASE_URL}/food/{fdc_id}"
            params = {'api_key': Config.USDA_API_KEY}
            
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
            self.cache[cache_key] = {
                'data': food_details,
                'timestamp': time.time()
            }
            self.save_cache()
            
            self.last_api_call = time.time()
            return food_details
            
        except Exception as e:
            st.error(f"Failed to get food details: {e}")
            return None
    
    def search_fallback_database(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search fallback food database"""
        results = []
        query_lower = query.lower()
        
        for food_name, nutrition in FALLBACK_FOOD_DATABASE.items():
            if query_lower in food_name.lower():
                results.append({
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
        
        return results[:max_results]
    
    def search_foods(self, query: str, max_results: int = 10) -> List[Dict[str, Any]]:
        """Search for foods with fallback to local database"""
        if not query or len(query) < 2:
            return []
        
        # Try USDA API first
        usda_results = self.search_usda_foods(query, max_results)
        
        if usda_results:
            return usda_results
        
        # Fallback to local database
        fallback_results = self.search_fallback_database(query, max_results)
        return fallback_results
    
    def get_food_suggestions(self, remaining_calories: float, meal_category: Optional[str] = None) -> List[Dict[str, Any]]:
        """Get food suggestions based on remaining calories and meal category"""
        suggestions = []
        
        # Use fallback database for suggestions (since we need categorized foods)
        for food_name, nutrition in FALLBACK_FOOD_DATABASE.items():
            if meal_category is None or nutrition['category'] == meal_category:
                if nutrition['calories'] <= remaining_calories:
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
    
    def get_meal_category_by_time(self) -> str:
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
    
    def calculate_portion_nutrition(self, base_nutrition: Dict[str, float], portion_size: float, unit: str = 'g') -> Dict[str, float]:
        """Calculate nutrition for a specific portion size"""
        # Convert common units to grams for calculation
        unit_conversions = {
            'g': 1,
            'oz': 28.35,
            'cup': 240,
            'tbsp': 15,
            'tsp': 5,
            'ml': 1,
            'l': 1000
        }
        
        conversion_factor = unit_conversions.get(unit.lower(), 1)
        grams = portion_size * conversion_factor
        
        # Calculate nutrition based on 100g serving
        multiplier = grams / 100
        
        return {
            'calories': base_nutrition.get('calories', 0) * multiplier,
            'protein': base_nutrition.get('protein', 0) * multiplier,
            'carbs': base_nutrition.get('carbs', 0) * multiplier,
            'fat': base_nutrition.get('fat', 0) * multiplier,
            'fiber': base_nutrition.get('fiber', 0) * multiplier,
            'sodium': base_nutrition.get('sodium', 0) * multiplier
        }
    
    def get_recent_foods(self, limit: int = 10) -> List[Dict[str, Any]]:
        """Get recently logged foods"""
        try:
            from data_manager import data_manager
            df = data_manager.load_food_logs()
            
            if df.empty:
                return []
            
            # Get unique foods with their most recent entry
            recent_foods = df.groupby('food_name').agg({
                'timestamp': 'max',
                'calories': 'mean',
                'protein': 'mean',
                'carbs': 'mean',
                'fat': 'mean'
            }).reset_index()
            
            # Sort by most recent
            recent_foods = recent_foods.sort_values('timestamp', ascending=False)
            
            return recent_foods.head(limit).to_dict('records')
            
        except Exception as e:
            st.error(f"Error getting recent foods: {e}")
            return []
    
    def get_favorite_foods(self) -> List[str]:
        """Get user's favorite foods from profile"""
        try:
            from data_manager import data_manager
            profile = data_manager.load_user_profile()
            if profile:
                return profile.favorite_foods
        except Exception as e:
            st.error(f"Error getting favorite foods: {e}")
        return []

# Global food API manager instance
food_api_manager = FoodAPIManager() 