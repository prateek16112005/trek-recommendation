from flask import Flask, request, jsonify, render_template
import pandas as pd
import requests
import urllib.parse
import joblib
import numpy as np

app = Flask(__name__)

# ===============================
# 📦 Load model and data
# ===============================
df = pd.read_csv('C:\\Users\\HP\\OneDrive\\Desktop\\trektrail\\final\\processed_trek_data.csv')
model = joblib.load('C:\\Users\\HP\\OneDrive\\Desktop\\trektrail\\final\\model.pkl')
ohe = joblib.load('C:\\Users\\HP\\OneDrive\\Desktop\\trektrail\\final\\ohe.pkl')
mlb = joblib.load('C:\\Users\\HP\\OneDrive\\Desktop\\trektrail\\final\\mlb.pkl')
columns = joblib.load('C:\\Users\\HP\\OneDrive\\Desktop\\trektrail\\final\\columns.pkl')

# Dropdown and range values
df['Location'] = df[['City', 'State', 'Country']].fillna('').agg(', '.join, axis=1)
states = sorted(df['State'].dropna().unique())
difficulties = sorted(df['Difficulty'].dropna().unique())
all_locations = sorted(df['Location'].dropna().unique())
filtered_data = df[['State', 'Difficulty', 'Location']].dropna().drop_duplicates().to_dict(orient='records')

length_min, length_max = round(df['Length (in km)'].min(), 1), round(df['Length (in km)'].max(), 1)
wind_min, wind_max = round(df['current_windspeed'].min(), 1), round(df['current_windspeed'].max(), 1)

# ✅ Predefined tags list (matches frontend JS)
tag_options = ["hiking", "forest", "views", "waterfall", "wildlife", "snow", "sunset", "lake", "nature", "photography"]

# Map state → season
season_mapping = {
    'Himachal Pradesh': 'April - June, September - November',
    'Uttarakhand': 'March - June, September - November',
    'Maharashtra': 'October - February',
    'Karnataka': 'October - February',
    'Kerala': 'September - March',
    'Jammu and Kashmir': 'May - October',
    'West Bengal': 'October - March',
    'Tamil Nadu': 'November - February',
    'Goa': 'November - February'
}

# ===============================
# 🌐 Geocoding + weather
# ===============================
def geocode_location(location):
    try:
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={urllib.parse.quote(location)}"
        response = requests.get(url, headers={"User-Agent": "TrekWeatherApp/1.0"}, timeout=10)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        return None, None
    except Exception as e:
        print("❌ Geocoding error:", e)
        return None, None

def fetch_current_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}¤t_weather=true"
        response = requests.get(url, timeout=10)
        return response.json().get("current_weather")
    except Exception as e:
        print("❌ Weather API error:", e)
        return None

# ===============================
# 🧠 Predict trek logic
# ===============================
def preprocess_user_input(user_input):
    input_data = pd.DataFrame({
        'Difficulty': [user_input['Difficulty']],
        'Best_Season': [user_input['Best_Season']],
        'State': [user_input['State']],
        'Length (in km)': [user_input['Length']],
        'current_windspeed': [user_input['windspeed']],
        'current_temperature': [user_input['temperature']],
        'number_of_reviews': [df['number_of_reviews'].mean()],
        'Est_time': [df['Est_time'].mean()],
        'current_weather_code': [df['current_weather_code'].mean()]
    })

    try:
        encoded_cat = pd.DataFrame(
            ohe.transform(input_data[['Difficulty', 'Best_Season', 'State']]),
            columns=ohe.get_feature_names_out(['Difficulty', 'Best_Season', 'State'])
        )
    except Exception as e:
        return None, f"Encoding error: {str(e)}"

    tags_encoded = pd.DataFrame(
        mlb.transform([user_input['tags_list']]),
        columns=mlb.classes_
    )

    X_input = pd.concat([
        input_data[['Length (in km)', 'current_windspeed', 'current_temperature',
                    'number_of_reviews', 'Est_time', 'current_weather_code']],
        encoded_cat, tags_encoded
    ], axis=1)

    for col in columns:
        if col not in X_input.columns:
            X_input[col] = 0
    X_input = X_input[columns]
    return X_input, None

def predict_trek(state, difficulty, length, temperature, windspeed, tags):
    user_input = {
        'State': state,
        'Best_Season': season_mapping.get(state, 'All Year'),
        'Difficulty': difficulty,
        'tags_list': tags if tags else ['hiking'],
        'Length': length,
        'windspeed': windspeed,
        'temperature': temperature
    }

    X_input, error = preprocess_user_input(user_input)
    if error:
        return {"error": error}

    try:
        proba = model.predict_proba(X_input)[0]
        trail_names = model.classes_
        top_indices = np.argsort(proba)[-10:][::-1]

        for idx in top_indices:
            predicted_trail = trail_names[idx]
            trek = df[df['Trail_name'] == predicted_trail]
            if trek.empty or trek.iloc[0]['State'] != state:
                continue

            trek = trek.iloc[0]
            max_proba = proba[idx]

            # Geocode the location to get latitude and longitude
            location = f"{trek['City']}, {trek['State']}, {trek['Country']}"
            latitude, longitude = geocode_location(location) if trek['City'] and trek['State'] else (None, None)

            # Mismatch warnings
            warnings = []
            if abs(length - trek['Length (in km)']) > 5:
                warnings.append(f"⚠️ Length differs significantly from your input ({trek['Length (in km)']} km).")
            if abs(windspeed - trek['current_windspeed']) > 5:
                warnings.append(f"⚠️ Windspeed differs significantly ({trek['current_windspeed']} km/h).")
            if difficulty != trek['Difficulty']:
                warnings.append(f"⚠️ Difficulty is different than selected ({trek['Difficulty']}).")

            return {
                "trail_name": trek['Trail_name'],
                "difficulty": trek['Difficulty'],
                "length_km": trek['Length (in km)'],
                "best_season": trek['Best_Season'],
                "state": trek['State'],
                "city": trek['City'],  # Added for clarity in location
                "country": trek['Country'],  # Added for clarity in location
                "tags": trek['Tags'],
                "windspeed": trek['current_windspeed'],
                "temperature": trek['current_temperature'],
                "description": trek['description'],
                "confidence": round(max_proba * 100, 2),
                "warnings": warnings,
                "latitude": latitude,
                "longitude": longitude
            }

        return {"error": f"No trek found in {state}. Try different inputs."}
    except Exception as e:
        return {"error": f"Prediction failed: {str(e)}"}

# ===============================
# 🔁 Routes
# ===============================
@app.route('/')
def index():
    return render_template(
        'index.html',
        states=states,
        difficulties=difficulties,
        all_locations=all_locations,
        filtered_data=filtered_data,
        all_tags=tag_options
    )

@app.route('/api/weather', methods=['GET'])
def get_weather():
    location = request.args.get('location')
    if not location:
        return jsonify({'error': 'Missing location'}), 400

    lat, lon = geocode_location(location)
    if lat is None or lon is None:
        return jsonify({'error': 'Location not found'}), 404

    weather = fetch_current_weather(lat, lon)
    if not weather:
        return jsonify({'error': 'Weather unavailable'}), 500

    return jsonify({
        "location": location,
        "latitude": lat,
        "longitude": lon,
        "weather": weather
    })

@app.route('/api/recommend', methods=['POST'])
def recommend():
    data = request.get_json()
    required = ['state', 'difficulty', 'length', 'temperature', 'windspeed', 'tags']
    if not all(k in data for k in required):
        return jsonify({"error": "Missing required fields"}), 400

    result = predict_trek(
        state=data['state'],
        difficulty=data['difficulty'],
        length=data['length'],
        temperature=data['temperature'],
        windspeed=data['windspeed'],
        tags=data['tags']
    )
    return jsonify(result)

# ===============================
# ▶️ Run the app
# ===============================
if __name__ == '__main__':
    app.run(debug=True)