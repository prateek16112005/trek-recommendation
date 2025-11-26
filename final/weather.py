import pandas as pd
import requests
import time
import urllib.parse

# Load the dataset
df = pd.read_csv(r"C:\Users\HP\OneDrive\Desktop\final_project\final\trek_dataset.csv")

# Create empty columns
df['latitude'] = None
df['longitude'] = None
df['current_temperature'] = None
df['current_windspeed'] = None
df['current_winddirection'] = None
df['current_weather_code'] = None

# Geocode function using OpenStreetMap Nominatim API
def geocode_location(location):
    try:
        url = f"https://nominatim.openstreetmap.org/search?format=json&q={urllib.parse.quote(location)}"
        response = requests.get(url, headers={"User-Agent": "TrekWeatherApp/1.0"}, timeout=10)
        data = response.json()
        if data:
            return float(data[0]['lat']), float(data[0]['lon'])
        else:
            return None, None
    except Exception as e:
        print(f"Geocoding failed for '{location}': {e}")
        return None, None

# Fetch current weather using Open-Meteo API
def fetch_weather(lat, lon):
    try:
        url = (
            f"https://api.open-meteo.com/v1/forecast"
            f"?latitude={lat}&longitude={lon}&current_weather=true"
        )
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'current_weather' in data:
            weather = data['current_weather']
            return (
                weather.get("temperature"),
                weather.get("windspeed"),
                weather.get("winddirection"),
                weather.get("weathercode")
            )
        else:
            return None, None, None, None
    except Exception as e:
        print(f"Weather fetch failed for ({lat}, {lon}): {e}")
        return None, None, None, None

# Process each row
for idx, row in df.iterrows():
    location = row['Location']

    print(f"⛰️  Processing: {location}")

    # Step 1: Geocode
    lat, lon = geocode_location(location)
    if lat is None or lon is None:
        print(f"   ❌ Could not geocode: {location}")
        continue

    df.at[idx, 'latitude'] = lat
    df.at[idx, 'longitude'] = lon

    # Step 2: Get current weather
    temp, windspeed, winddir, code = fetch_weather(lat, lon)
    df.at[idx, 'current_temperature'] = temp
    df.at[idx, 'current_windspeed'] = windspeed
    df.at[idx, 'current_winddirection'] = winddir
    df.at[idx, 'current_weather_code'] = code

    # Respect API limits
    time.sleep(1)  # to avoid IP rate limiting from Nominatim

# Save final dataset
output_path = r"C:\Users\HP\OneDrive\Desktop\final_project\final\trek_dataset_with_current_weather.csv"
df.to_csv(output_path, index=False)
print(f"\n✅ Done! Weather data saved to: {output_path}")
