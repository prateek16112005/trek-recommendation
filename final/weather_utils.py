import requests
import urllib.parse

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

def fetch_current_weather(lat, lon):
    try:
        url = f"https://api.open-meteo.com/v1/forecast?latitude={lat}&longitude={lon}&current_weather=true"
        response = requests.get(url, timeout=10)
        data = response.json()
        if 'current_weather' in data:
            weather = data['current_weather']
            return {
                "temperature": weather.get("temperature"),
                "windspeed": weather.get("windspeed"),
                "winddirection": weather.get("winddirection"),
                "weathercode": weather.get("weathercode")
            }
        else:
            return None
    except Exception as e:
        print(f"Weather fetch failed: {e}")
        return None
