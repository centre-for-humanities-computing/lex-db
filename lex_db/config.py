"""Configuration settings for Lex DB."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings.
    
    Attributes:
        database_url: Path to the SQLite database file.
        app_name: Name of the application.
        debug: Whether to run in debug mode.
    """
    
    # Database settings
    DATABASE_URL: str
    
    # API settings
    APP_NAME: str = "Lex DB API"
    DEBUG: bool = False
    
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )


def get_settings() -> Settings:
    """Get application settings.
    
    Returns:
        Settings: Application settings.
    """
    return Settings()