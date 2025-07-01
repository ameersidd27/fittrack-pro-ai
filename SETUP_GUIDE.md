# FitTrack Pro - Complete Setup Guide

## 🚀 Overview

This guide will help you set up the enhanced FitTrack Pro platform, a comprehensive AI-powered fitness and nutrition tracking application. The platform includes:

- **AI-Powered Coaching**: Personalized nutrition and fitness advice using OpenAI GPT-4o Mini
- **USDA Food Database**: Real-time food search and nutrition data
- **Advanced Analytics**: Progress tracking, pattern analysis, and goal management
- **Comprehensive User Profiles**: BMR calculation, goal setting, and personalized recommendations
- **Workout Tracking**: METs-based calorie calculation and exercise logging
- **Modern UI**: Dark theme with responsive design

## 📋 Prerequisites

- Python 3.8 or higher
- Git (for cloning the repository)
- OpenAI API key (for AI coaching features)
- USDA FoodData Central API key (optional, demo key available)

## 🛠️ Installation

### Step 1: Clone and Setup Project

```bash
# Clone the repository (if using git)
git clone <repository-url>
cd fittrack-pro

# Or navigate to your project directory
cd /path/to/your/fittrack-pro
```

### Step 2: Create Virtual Environment

```bash
# Create virtual environment
python -m venv venv

# Activate virtual environment
# Windows:
venv\Scripts\activate
# macOS/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
# Install all required packages
pip install -r requirements.txt
```

**Required Packages:**
- streamlit>=1.46.0
- pandas>=2.3.0
- numpy>=2.3.0
- requests>=2.32.0
- python-dotenv>=1.0.0
- openai>=1.0.0
- plotly>=5.17.0
- altair>=5.5.0
- Pillow>=11.2.0
- python-dateutil>=2.9.0

## 🔑 API Key Configuration

### Step 1: Get OpenAI API Key

1. Visit [OpenAI Platform](https://platform.openai.com/api-keys)
2. Sign up or log in to your account
3. Navigate to "API Keys" section
4. Click "Create new secret key"
5. Copy the generated key (starts with `sk-`)

### Step 2: Get USDA API Key (Optional)

1. Visit [USDA FoodData Central](https://fdc.nal.usda.gov/api-key-signup.html)
2. Fill out the registration form
3. Check your email for the API key
4. The demo key will work for testing, but has rate limits

### Step 3: Create Environment File

Create a `.env` file in your project root:

```bash
# Create .env file
touch .env  # macOS/Linux
# OR
echo. > .env  # Windows
```

Add the following content to `.env`:

```env
# OpenAI API Configuration
OPENAI_API_KEY=your_openai_api_key_here

# USDA FoodData Central API Configuration
USDA_API_KEY=your_usda_api_key_here

# App Configuration
APP_ENVIRONMENT=development
DEBUG_MODE=True

# AI Configuration
AI_MODEL=gpt-4o-mini
MAX_AI_REQUESTS_PER_MINUTE=3

# Cache Configuration
CACHE_DURATION_HOURS=24
MAX_CACHE_SIZE_MB=100
```

**Replace the placeholder values:**
- `your_openai_api_key_here` with your actual OpenAI API key
- `your_usda_api_key_here` with your USDA API key (or leave as DEMO_KEY for testing)

## 🗂️ Database Initialization

### Step 1: Create Data Directory

The application will automatically create the data directory and required files on first run, but you can also create them manually:

```bash
# Create data directory
mkdir data

# The following files will be created automatically:
# - data/food_logs.csv
# - data/user_profile.csv
# - data/workout_logs.csv
# - data/goal_tracking.csv
# - data/food_cache.json
# - data/ai_conversations.json
# - data/ai_usage.json
# - data/ai_insights.json
```

### Step 2: Verify File Structure

Your project should have this structure:

```
fittrack-pro/
├── app_enhanced.py          # Main application
├── config.py               # Configuration module
├── models.py               # Data models
├── data_manager.py         # Data management
├── food_api.py             # Food API integration
├── ai_coach.py             # AI coaching system
├── requirements.txt        # Dependencies
├── .env                    # Environment variables
├── data/                   # Data directory
└── venv/                   # Virtual environment
```

## 🚀 Running the Application

### Step 1: Start the Application

```bash
# Make sure your virtual environment is activated
streamlit run app_enhanced.py
```

### Step 2: Access the Application

Open your web browser and navigate to:
```
http://localhost:8501
```

## ✅ Testing & Verification

### Step 1: System Status Check

1. Open the application
2. Go to the sidebar and expand "🔧 System Status"
3. Verify all components show "✅ Configured":
   - OpenAI API: ✅ Configured
   - USDA API: ✅ Configured (or ⚠️ Using Demo Key)
   - Data Directory: ✅ Ready

### Step 2: User Profile Setup

1. Navigate to "👤 User Profile"
2. Fill out your personal information:
   - Name, Age, Weight, Height, Gender
   - Activity Level
   - Dietary preferences
   - AI coaching style
3. Click "Save Profile"
4. Verify BMR and TDEE calculations appear

### Step 3: Food Logging Test

1. Navigate to "🍎 Log Food"
2. Search for a food item (e.g., "apple")
3. Verify USDA API results appear
4. Add a food item to your log
5. Check that it appears in "📋 Food History"

### Step 4: Workout Logging Test

1. Navigate to "💪 Log Workout"
2. Fill out workout details
3. Verify calorie calculation works
4. Check that it appears in "🏃‍♂️ Workout History"

### Step 5: AI Coaching Test

1. Navigate to "🤖 AI Coach"
2. Click "Get AI Analysis" in the Daily Briefing tab
3. Verify AI response appears
4. Test the chat interface with a question
5. Check usage statistics update

### Step 6: Goal Setting Test

1. Navigate to "🎯 Goal Setting"
2. Create a weight loss or gain goal
3. Verify progress tracking works
4. Update current weight and see progress percentage

## 🔧 Configuration Options

### Environment Variables

You can customize the application behavior by modifying the `.env` file:

```env
# AI Configuration
AI_MODEL=gpt-4o-mini                    # OpenAI model to use
MAX_AI_REQUESTS_PER_MINUTE=3           # Rate limiting

# Cache Configuration
CACHE_DURATION_HOURS=24                # How long to cache food data
MAX_CACHE_SIZE_MB=100                  # Maximum cache size

# App Configuration
APP_ENVIRONMENT=development            # development/production
DEBUG_MODE=True                        # Enable debug features
```

### BMR Calculation Options

The app supports multiple BMR calculation formulas:
- **Mifflin-St Jeor** (default, most accurate)
- **Harris-Benedict** (older formula)
- **Katch-McArdle** (requires body fat percentage)

### Activity Level Multipliers

- **Sedentary**: 1.2 (little or no exercise)
- **Lightly Active**: 1.375 (light exercise 1-3 days/week)
- **Moderately Active**: 1.55 (moderate exercise 3-5 days/week)
- **Very Active**: 1.725 (hard exercise 6-7 days/week)
- **Extremely Active**: 1.9 (very hard exercise, physical job)

## 🐛 Troubleshooting

### Common Issues

#### 1. OpenAI API Errors

**Problem:** "AI coaching is not available"
**Solution:**
- Verify your OpenAI API key is correct in `.env`
- Check that you have billing set up on OpenAI
- Ensure the API key has the necessary permissions

#### 2. USDA API Errors

**Problem:** Food search not working
**Solution:**
- The demo key has rate limits, consider getting a full API key
- Check your internet connection
- Verify the API key format in `.env`

#### 3. Import Errors

**Problem:** Module not found errors
**Solution:**
- Ensure virtual environment is activated
- Run `pip install -r requirements.txt`
- Check that all files are in the correct locations

#### 4. Data Not Saving

**Problem:** Changes not persisting
**Solution:**
- Verify the `data/` directory exists and is writable
- Check file permissions
- Ensure no other instances are running

#### 5. Performance Issues

**Problem:** Slow loading or response times
**Solution:**
- Clear the food cache: Settings → Clear Food Cache
- Reduce cache duration in `.env`
- Check your internet connection for API calls

### Debug Mode

Enable debug mode to see detailed error messages:

```env
DEBUG_MODE=True
```

### Logs and Monitoring

The application creates several log files:
- `data/ai_usage.json` - AI API usage tracking
- `data/ai_conversations.json` - Chat history
- `data/food_cache.json` - Cached food data

## 📊 Usage Statistics

### AI Usage Tracking

The application tracks AI usage to help manage costs:
- Total requests made
- Tokens used
- Cost incurred
- Rate limiting status

### Data Export

You can export all your data for backup:
1. Go to Settings
2. Click "Export All Data"
3. Download the JSON file

## 🔒 Security & Privacy

### Data Storage

- All data is stored locally in CSV and JSON files
- No data is sent to external servers except for API calls
- API keys are stored in environment variables

### API Key Security

- Never commit your `.env` file to version control
- Use environment variables for production deployment
- Regularly rotate your API keys

### Data Privacy

- User data is stored locally only
- AI conversations are saved for context but not shared
- You can clear all data at any time

## 🚀 Production Deployment

### Environment Setup

For production deployment:

```env
APP_ENVIRONMENT=production
DEBUG_MODE=False
OPENAI_API_KEY=your_production_key
USDA_API_KEY=your_production_key
```

### Performance Optimization

- Increase cache duration for better performance
- Monitor API usage and costs
- Consider using a production-grade database

### Security Considerations

- Use secure environment variable management
- Implement proper authentication if needed
- Regular security updates and monitoring

## 📈 Advanced Features

### Custom Food Database

You can extend the fallback food database in `food_api.py`:

```python
FALLBACK_FOOD_DATABASE = {
    'your_food': {
        'calories': 100,
        'protein': 10,
        'carbs': 15,
        'fat': 5,
        'category': 'snack'
    }
}
```

### Custom AI Prompts

Modify AI coaching behavior in `ai_coach.py`:

```python
def generate_daily_analysis(self) -> str:
    # Customize the system prompt
    system_prompt = """Your custom coaching style..."""
```

### Data Analysis

The application provides comprehensive analytics:
- Daily and weekly nutrition trends
- Workout performance tracking
- Goal progress visualization
- AI-powered pattern analysis

## 🆘 Support

### Getting Help

1. Check the troubleshooting section above
2. Verify your configuration matches the setup guide
3. Test with the demo keys first
4. Check the application logs for error messages

### Common Questions

**Q: How much does the AI coaching cost?**
A: Costs depend on usage. GPT-4o-mini costs approximately $0.00015 per 1K tokens.

**Q: Can I use the app without AI features?**
A: Yes, all core features work without OpenAI API key.

**Q: Is my data secure?**
A: Yes, all data is stored locally and not shared with third parties.

**Q: Can I export my data?**
A: Yes, use the export feature in Settings to download all your data.

## 🎉 Congratulations!

You've successfully set up FitTrack Pro! The application is now ready to help you:

- Track your nutrition with real-time food data
- Log workouts with accurate calorie calculations
- Get personalized AI coaching and insights
- Set and track fitness goals
- Analyze your progress with advanced analytics

Start by setting up your user profile and logging your first meal. The AI coach will provide personalized recommendations based on your goals and preferences.

Happy tracking! 💪 