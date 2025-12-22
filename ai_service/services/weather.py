"""
Weather Service (v2.1.0)
OpenWeatherMap integration for context-aware outfit recommendations.
"""
import os
import logging
from typing import Optional
from dataclasses import dataclass
import httpx

logger = logging.getLogger(__name__)

# Configuration
OPENWEATHER_API_KEY = os.getenv("OPENWEATHER_API_KEY", "")
OPENWEATHER_BASE_URL = "https://api.openweathermap.org/data/2.5/weather"
DEFAULT_UNITS = "metric"  # Celsius


@dataclass
class WeatherInfo:
    """Weather information for outfit recommendations."""
    city: str
    country: str
    temp_celsius: float
    feels_like: float
    humidity: int
    condition: str  # e.g., "clear", "rain", "snow"
    description: str  # e.g., "light rain"
    wind_speed: float  # m/s
    layer_hint: str  # "light", "medium", "heavy"
    
    def to_dict(self) -> dict:
        return {
            "city": self.city,
            "country": self.country,
            "temp_celsius": self.temp_celsius,
            "feels_like": self.feels_like,
            "humidity": self.humidity,
            "condition": self.condition,
            "description": self.description,
            "wind_speed": self.wind_speed,
            "layer_hint": self.layer_hint
        }
    
    def to_prompt_context(self) -> str:
        """Generate context string for LLM prompt."""
        return (
            f"Weather: {self.description} in {self.city}, "
            f"temperature {self.temp_celsius:.0f}°C (feels like {self.feels_like:.0f}°C), "
            f"humidity {self.humidity}%. "
            f"Recommended clothing weight: {self.layer_hint}."
        )


def _derive_layer_hint(temp: float, condition: str, wind_speed: float) -> str:
    """
    Derive clothing layer recommendation based on weather.
    
    Returns:
        "light", "medium", or "heavy"
    """
    # Wind chill effect
    effective_temp = temp - (wind_speed * 0.5) if wind_speed > 5 else temp
    
    # Rain/snow adjustment
    if condition in ["rain", "drizzle", "thunderstorm"]:
        effective_temp -= 3  # Rain feels colder
    elif condition in ["snow", "sleet"]:
        effective_temp -= 5
    
    # Determine layer
    if effective_temp >= 25:
        return "light"
    elif effective_temp >= 15:
        return "medium"
    else:
        return "heavy"


def is_configured() -> bool:
    """Check if weather service is configured."""
    return bool(OPENWEATHER_API_KEY)


async def get_weather(city: str) -> Optional[WeatherInfo]:
    """
    Fetch current weather for a city.
    
    Args:
        city: City name (e.g., "Istanbul", "London", "New York")
    
    Returns:
        WeatherInfo or None if failed
    """
    if not OPENWEATHER_API_KEY:
        logger.warning("OPENWEATHER_API_KEY not set - weather disabled")
        return None
    
    try:
        async with httpx.AsyncClient(timeout=10.0) as client:
            response = await client.get(
                OPENWEATHER_BASE_URL,
                params={
                    "q": city,
                    "appid": OPENWEATHER_API_KEY,
                    "units": DEFAULT_UNITS
                }
            )
            
            if response.status_code == 404:
                logger.warning(f"City not found: {city}")
                return None
            
            response.raise_for_status()
            data = response.json()
            
            # Extract weather data
            main = data.get("main", {})
            weather = data.get("weather", [{}])[0]
            wind = data.get("wind", {})
            sys = data.get("sys", {})
            
            temp = main.get("temp", 20)
            condition = weather.get("main", "Clear").lower()
            wind_speed = wind.get("speed", 0)
            
            weather_info = WeatherInfo(
                city=data.get("name", city),
                country=sys.get("country", ""),
                temp_celsius=temp,
                feels_like=main.get("feels_like", temp),
                humidity=main.get("humidity", 50),
                condition=condition,
                description=weather.get("description", "clear sky"),
                wind_speed=wind_speed,
                layer_hint=_derive_layer_hint(temp, condition, wind_speed)
            )
            
            logger.info(f"Weather: {weather_info.city} - {weather_info.temp_celsius}°C, {weather_info.layer_hint}")
            return weather_info
            
    except httpx.TimeoutException:
        logger.warning(f"Weather API timeout for {city}")
        return None
    except Exception as e:
        logger.error(f"Weather API error for {city}: {e}")
        return None


def get_weather_sync(city: str) -> Optional[WeatherInfo]:
    """
    Synchronous version of get_weather.
    Uses requests instead of httpx for sync context.
    """
    if not OPENWEATHER_API_KEY:
        return None
    
    try:
        import requests
        
        response = requests.get(
            OPENWEATHER_BASE_URL,
            params={
                "q": city,
                "appid": OPENWEATHER_API_KEY,
                "units": DEFAULT_UNITS
            },
            timeout=10
        )
        
        if response.status_code != 200:
            return None
        
        data = response.json()
        main = data.get("main", {})
        weather = data.get("weather", [{}])[0]
        wind = data.get("wind", {})
        sys = data.get("sys", {})
        
        temp = main.get("temp", 20)
        condition = weather.get("main", "Clear").lower()
        wind_speed = wind.get("speed", 0)
        
        return WeatherInfo(
            city=data.get("name", city),
            country=sys.get("country", ""),
            temp_celsius=temp,
            feels_like=main.get("feels_like", temp),
            humidity=main.get("humidity", 50),
            condition=condition,
            description=weather.get("description", "clear sky"),
            wind_speed=wind_speed,
            layer_hint=_derive_layer_hint(temp, condition, wind_speed)
        )
        
    except Exception as e:
        logger.error(f"Weather sync error: {e}")
        return None
