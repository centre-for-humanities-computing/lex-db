"""Configuration settings for Lex DB."""

from dotenv import load_dotenv
from pydantic_settings import BaseSettings, SettingsConfigDict


# Load environment variables from .env file
load_dotenv()


class Settings(BaseSettings):
    """Application settings."""

    # Database settings
    DB_HOST: str = "localhost"
    DB_PORT: int = 5432
    DB_NAME: str = "lex_db"
    DB_USER: str = "lex_user"
    DB_PASSWORD: str = ""

    # API settings
    APP_NAME: str = "Lex DB API"
    DEBUG: bool = False

    # Connection pooling
    DB_POOL_MIN_SIZE: int = 2
    DB_POOL_MAX_SIZE: int = 10

    # Sitemap settings
    SITEMAP_BASE_URL: str = "https://lex.dk/.sitemap"
    SITEMAP_REQUEST_TIMEOUT: int = 30
    SITEMAP_MAX_RETRIES: int = 3
    SITEMAP_RATE_LIMIT: int = 10  # Max concurrent JSON requests

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    @property
    def DATABASE_URL(self) -> str:
        """Construct the database URL."""
        return f"postgresql://{self.DB_USER}:{self.DB_PASSWORD}@{self.DB_HOST}:{self.DB_PORT}/{self.DB_NAME}"


def get_settings() -> Settings:
    """Get application settings."""
    return Settings()
