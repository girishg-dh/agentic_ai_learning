from pydantic import BaseModel, Field
from typing import Optional, List

class WeatherResponse(BaseModel):
    """A structured representation of a weather report."""
    location: str = Field(..., description="The city and country name e.g. 'Berlin, Germany'")
    temperature: float = Field(..., description="The current temperature in Celsius")
    description: str = Field(..., description="A brief description of the weather conditions")
    error_message: Optional[str] = Field(None, description="An Optional field for error message if the API call failed")


class NewsArticle(BaseModel):
    """A structured representation of a news article."""
    title: str = Field(..., description="The title of the news article")
    url: str = Field(..., description="The URL of the full news article")
    error_message: Optional[str] = Field(None, description="An Optional field for error message if the API call failed")


class StockPriceResponse(BaseModel):
    """A structured representation of a stock price response."""
    ticker: str = Field(..., description="The stock ticker symbol")
    price: float = Field(..., description="The current stock price")
    volume: int = Field(..., description="The trading volume of the stock")
    error_message: Optional[str] = Field(None, description="An Optional field for error message if the API call failed")

class CityReport(BaseModel):
    """A full report for a city, including weather and top news headlines."""
    city: str = Field(..., description="The name of the city")
    country: str = Field(..., description="The country where the city is located")
    weather: Optional[WeatherResponse] = Field(None, description="The current weather report for the city")
    news: Optional[List[NewsArticle]] = Field(None, description="A list of top news headlines")
    error_message: Optional[str] = Field(None, description="An Optional field for error message if any API call failed")