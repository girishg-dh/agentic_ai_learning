import os
import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# It's good practice to fetch the API key once and store it in a variable
OPENWEATHERMAP_API_KEY = os.getenv("OPEN_WEATHER_API_KEY")

def get_current_weather(location: str) -> str:
    """
    Get the current weather for a specific location.

    Args:
        location (str): The city name (e.g., "San Francisco").

    Returns:
        str: A string containing the weather information or an error message.
    """
    if not OPENWEATHERMAP_API_KEY:
        return "Error: OpenWeatherMap API key is not set."

    # The base URL for the OpenWeatherMap API
    url = f"http://api.openweathermap.org/data/2.5/weather"

    # Parameters for the API request
    params = {
        "q": location,
        "appid": OPENWEATHERMAP_API_KEY,
        "units": "metric"  # Use metric units (Celsius)
    }

    try:
        # Make the GET request to the API
        response = requests.get(url, params=params)
        # Raise an exception for bad status codes (4xx or 5xx)
        response.raise_for_status()
        
        # Parse the JSON response
        data = response.json()
        
        # Extract the relevant information
        weather_description = data['weather'][0]['description']
        temperature = data['main']['temp']
        city = data['name']
        country = data['sys']['country']
        
        return f"The current weather in {city}, {country} is {temperature}°C with {weather_description}."

    except requests.exceptions.HTTPError as http_err:
        # Handle specific HTTP errors, like 404 Not Found for an invalid city
        if response.status_code == 404:
            return f"Error: City '{location}' not found. Please check the spelling."
        else:
            return f"Error: HTTP error occurred: {http_err}"
    except Exception as e:
        # Handle other potential errors (e.g., network issues)
        return f"An unexpected error occurred: {e}"

# ---- Test the function ----
if __name__ == "__main__":
    # This block runs only when you execute this script directly
    print("Testing the weather tool...")
    test_city = "Berlin"
    weather_result = get_current_weather(test_city)
    print(weather_result)

    print("\nTesting with an invalid city...")
    invalid_city_result = get_current_weather("InvalidCityName123")
    print(invalid_city_result)