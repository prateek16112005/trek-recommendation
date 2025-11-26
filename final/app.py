import streamlit as st
import pandas as pd
import joblib
import numpy as np
import requests
import urllib.parse

# ===============================
# 📦 Load model and data
# ===============================

df = pd.read_csv("final/processed_trek_data.csv")
model = joblib.load("final/model.pkl")
ohe = joblib.load("finalohe.pkl")
mlb = joblib.load("final/mlb.pkl")
columns = joblib.load("final/columns.pkl")

df['Location'] = df[['City', 'State', 'Country']].fillna('').agg(', '.join, axis=1)
states = sorted(df['State'].dropna().unique())
difficulties = sorted(df['Difficulty'].dropna().unique())

length_min, length_max = round(df['Length (in km)'].min(), 1), round(df['Length (in km)'].max(), 1)
wind_min, wind_max = round(df['current_windspeed'].min(), 1), round(df['current_windspeed'].max(), 1)

tag_options = ["hiking", "forest", "views", "waterfall", "wildlife", "snow", "sunset", "lake", "nature", "photography"]

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
# 🌐 Geocoding + Weather
# ===============================
def geocode_location(location):
    try:
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={urllib.parse.quote(location)}"
        response = requests.get(url, headers={"User-Agent": "TrekWeatherApp/1.0"}, timeout=10)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        return None, None
    except:
        return None, None

def fetch_current_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=10)
        return response.json().get("current_weather")
    except:
        return None

# ===============================
# 🧠 Preprocess & Predict
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

    encoded_cat = pd.DataFrame(
        ohe.transform(input_data[['Difficulty','Best_Season','State']]),
        columns=ohe.get_feature_names_out(['Difficulty','Best_Season','State'])
    )

    tags_encoded = pd.DataFrame(
        mlb.transform([user_input['tags_list']]),
        columns=mlb.classes_
    )

    X_input = pd.concat([
        input_data[['Length (in km)','current_windspeed','current_temperature',
                    'number_of_reviews','Est_time','current_weather_code']],
        encoded_cat, tags_encoded
    ], axis=1)

    for col in columns:
        if col not in X_input.columns:
            X_input[col] = 0

    X_input = X_input[columns]
    return X_input

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

    X_input = preprocess_user_input(user_input)

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

        location = f"{trek['City']}, {trek['State']}, {trek['Country']}"
        latitude, longitude = geocode_location(location) if trek['City'] else (None, None)

        warnings = []
        if abs(length - trek['Length (in km)']) > 5:
            warnings.append(f"⚠️ Length differs significantly ({trek['Length (in km)']} km)")
        if abs(windspeed - trek['current_windspeed']) > 5:
            warnings.append(f"⚠️ Windspeed differs significantly ({trek['current_windspeed']} km/h)")
        if difficulty != trek['Difficulty']:
            warnings.append(f"⚠️ Difficulty differs ({trek['Difficulty']})")

        return {
            "trail_name": trek['Trail_name'],
            "difficulty": trek['Difficulty'],
            "length_km": trek['Length (in km)'],
            "best_season": trek['Best_Season'],
            "state": trek['State'],
            "city": trek['City'],
            "country": trek['Country'],
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

# ===============================
# 🎨 STREAMLIT UI
# ===============================
st.title("🏔️ Trek Recommendation System")

state = st.selectbox("Select State", states)
difficulty = st.selectbox("Select Difficulty", difficulties)
length = st.slider("Length (km)", length_min, length_max, 10.0)
temperature = st.number_input("Temperature (°C)", value=20.0)
windspeed = st.number_input("Wind Speed (km/h)", value=5.0)
tags = st.multiselect("Select Tags", tag_options)

if st.button("Recommend Trek"):
    result = predict_trek(state, difficulty, length, temperature, windspeed, tags)

    if "error" in result:
        st.error(result["error"])
    else:
        st.success(f"✅ Recommended Trek: {result['trail_name']}")
        st.write(f"**Confidence:** {result['confidence']}%")
        st.write(f"**Difficulty:** {result['difficulty']}")
        st.write(f"**Length:** {result['length_km']} km")
        st.write(f"**Best Season:** {result['best_season']}")
        st.write(f"**Location:** {result['city']}, {result['state']}, {result['country']}")
        st.write(f"**Weather:** 🌡️ {result['temperature']}°C | 💨 {result['windspeed']} km/h")
        st.write(f"**Description:** {result['description']}")

        if result['warnings']:
            st.warning("\n".join(result['warnings']))

        if result['latitude'] and result['longitude']:
            st.map(pd.DataFrame({"lat":[result['latitude']], "lon":[result['longitude']]}))
