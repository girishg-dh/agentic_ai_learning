import os
import requests
from dotenv import load_dotenv
from langchain.tools import tool
from typing import List


# Load environment variables from .env file
load_dotenv("/Users/girish.gupta/work/learn/learning_reasources/agentic_ai_learning/.env")

# It's good practice to fetch the API key once and store it in a variable
OPENWEATHERMAP_API_KEY = os.getenv("OPEN_WEATHER_API_KEY")
NEWS_API_KEY = os.getenv("NEWS_API_KEY")
FMP_API_KEY = os.getenv("FMP_API_KEY")

@tool
def get_stock_price(ticker: str) -> str:
    """
    Get the current stock price for a given ticker symbol.

    Args:
        ticker (str): The stock ticker symbol (e.g., "AAPL" for Apple).

    Returns:
        str: A string containing the stock price information or an error message.
    """
    if not FMP_API_KEY:
        return "Error: Financial Modeling Prep API key is not set."

    url = f"https://financialmodelingprep.com/api/v3/quote-short/{ticker.upper()}"
    params = {
        "apikey": FMP_API_KEY
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        if not data:
            return f"Error: No data found for ticker '{ticker}'."
        price = data[0].get("price")
        volume = data[0].get("volume")
        return f"The current stock price of {ticker.upper()} is ${price} with a volume of {volume}."
    except requests.exceptions.HTTPError as http_err:
        return f"Error: HTTP error occurred: {http_err}"
    except Exception as e:
        return f"An unexpected error occurred: {e}"

@tool
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
            return (
                f"Error: City '{location}' not found. Please check the spelling."
            )
        else:
            return f"Error: HTTP error occurred: {http_err}"
    except Exception as e:
        # Handle other potential errors (e.g., network issues)
        return f"An unexpected error occurred: {e}"
    
@tool
def get_top_headlines(country: str) -> List[str]:
    """
    Get the top 5 news headlines for a specific country.

    Args:
        country (str): The two-letter country code (e.g., "us", "gb", "de").

    Returns:
        List[str]: A list of headline strings or an error message.
    """
    
    if not NEWS_API_KEY:
        return ["Error: News API key is not set."]

    url = f"https://newsapi.org/v2/top-headlines"
    params = {
        "country": country,
        "apiKey": NEWS_API_KEY,
        "pageSize": 5
    }

    try:
        response = requests.get(url, params=params)
        response.raise_for_status()
        data = response.json()
        articles = data.get("articles", [])
        if not articles:
            return [f"No news articles found for country code '{country}'."]
        return [article["title"] for article in articles if article.get("title")]
    except requests.exceptions.HTTPError as http_err:
        error_details = response.json().get("message", str(http_err))
        return [f"Error: HTTP error occurred: {http_err} - {error_details}"]
    except Exception as e:
        return [f"An unexpected error occurred: {e}"]

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

    # Testing the news tool
    print("\nTesting the news tool...")
    test_country = "us"
    news_result = get_top_headlines(test_country)
    print(news_result)

    print("\nTesting with an invalid country code...")
    invalid_country_result = get_top_headlines("invalid_country")
    print(invalid_country_result)

    # Testing the stock tool
    print("\nTesting the stock tool...")
    test_ticker = "AAPL"
    stock_result = get_stock_price(test_ticker)
    print(stock_result)

    print("\nTesting with an invalid ticker...")
    invalid_ticker_result = get_stock_price("INVALID")
    print(invalid_ticker_result)