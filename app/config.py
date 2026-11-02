"""Application settings, loaded from the environment (and `.env` locally)."""

from functools import lru_cache

from pydantic import Field, SecretStr
from pydantic_settings import BaseSettings, SettingsConfigDict


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    anthropic_api_key: SecretStr = Field(min_length=1)
    openai_api_key: SecretStr | None = None

    postgres_user: str = "app"
    postgres_password: SecretStr = SecretStr("change-me")
    postgres_db: str = "app"
    postgres_host: str = "localhost"
    postgres_port: int = 5432

    elasticsearch_url: str = "http://localhost:9200"

    @property
    def postgres_dsn(self) -> str:
        return (
            f"postgresql://{self.postgres_user}"
            f":{self.postgres_password.get_secret_value()}"
            f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
        )


@lru_cache
def get_settings() -> Settings:
    return Settings()
