"""Configuration management for the application"""

import os
from pathlib import Path
from pydantic_settings import BaseSettings, SettingsConfigDict
from pydantic import field_validator


class Settings(BaseSettings):
    """Application settings"""
    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        extra="ignore",
    )

    # Database
    database_url: str = os.getenv("DATABASE_URL", "sqlite:///./whatif.db")
    postgres_user: str = os.getenv("POSTGRES_USER", "")
    postgres_password: str = os.getenv("POSTGRES_PASSWORD", "")
    postgres_host: str = os.getenv("POSTGRES_HOST", "localhost")
    postgres_port: int = int(os.getenv("POSTGRES_PORT", "5432"))
    postgres_db: str = os.getenv("POSTGRES_DB", "whatif")

    # API Configuration
    debug: bool = True
    log_level: str = os.getenv("LOG_LEVEL", "INFO")

    # Storage
    storage_path: str = os.getenv("STORAGE_PATH", "./storage")
    storage_projects_path: Path = Path(storage_path) / "projects"
    storage_temp_path: Path = Path(storage_path) / "temp"

    # Chroma
    chroma_persist_path: str = os.getenv("CHROMA_PERSIST_PATH", "./storage/chroma")

    # CORS
    cors_origins: list[str] = [
        "http://localhost:3000",
        "http://localhost:5173",
        "http://localhost:5180",
        "http://127.0.0.1:3000",
        "http://127.0.0.1:5173",
        "http://127.0.0.1:5180",
    ]

    # Server
    host: str = os.getenv("HOST", "0.0.0.0")
    port: int = int(os.getenv("PORT", "8000"))
    frontend_dist_dir: Path = Path(os.getenv("FRONTEND_DIST_DIR", "./web/dist"))
    anet_token: str = os.getenv("ANET_TOKEN", "")

    # Primary LLM (OpenAI-compatible)
    openai_api_key: str = os.getenv("OPENAI_API_KEY", "")
    openai_base_url: str = os.getenv("OPENAI_BASE_URL", "")
    openai_model: str = os.getenv("OPENAI_MODEL", os.getenv("AUTOGEN_MODEL", ""))

    # Alternative LLM Providers
    siliconflow_api_key: str = os.getenv("SILICONFLOW_API_KEY", "")
    siliconflow_base_url: str = os.getenv("SILICONFLOW_BASE_URL", "https://api.siliconflow.cn/v1")
    deepseek_api_key: str = os.getenv("DEEPSEEK_API_KEY", "")
    deepseek_base_url: str = os.getenv("DEEPSEEK_BASE_URL", "https://api.deepseek.com/v1")
    zhipu_api_key: str = os.getenv("ZHIPU_API_KEY", "")
    zhipu_base_url: str = os.getenv("ZHIPU_BASE_URL", "https://open.bigmodel.cn/api/paas/v4")

    # Anthropic (Claude)
    anthropic_api_key: str = os.getenv("ANTHROPIC_API_KEY", "")
    anthropic_base_url: str = os.getenv("ANTHROPIC_BASE_URL", "https://api.anthropic.com")

    # Video Generation
    happyhorse_api_key: str = os.getenv("HAPPYHORSE_API_KEY", "")
    happyhorse_base_url: str = os.getenv("HAPPYHORSE_BASE_URL", "https://dashscope.aliyuncs.com")
    kling_api_key: str = os.getenv("KLING_API_KEY", "")
    kling_base_url: str = os.getenv("KLING_BASE_URL", "https://api.kling.kuaishou.com")
    wan_api_key: str = os.getenv("WAN_API_KEY", "")
    wan_base_url: str = os.getenv("WAN_BASE_URL", "")
    seedance_api_key: str = os.getenv("SEEDANCE_API_KEY", "")
    seedance_api_url: str = os.getenv("SEEDANCE_API_URL", "http://localhost:8000")
    # Local video model (Wan2.1-VACE-14B or similar)
    local_video_model_path: str = os.getenv("LOCAL_VIDEO_MODEL_PATH", "")
    local_video_model_name: str = os.getenv("LOCAL_VIDEO_MODEL_NAME", "Wan-AI/Wan2.1-VACE-14B")

    # OpenAI-Next API (video generation & editing)
    openai_next_api_key: str = os.getenv("OPENAI_NEXT_API_KEY", "")
    openai_next_base_url: str = os.getenv("OPENAI_NEXT_BASE_URL", "https://draw.openai-next.com")
    openai_next_video_model: str = os.getenv("OPENAI_NEXT_VIDEO_MODEL", "wan2.7-videoedit")
    # Public HTTPS base URL for serving video assets to external APIs
    public_base_url: str = os.getenv("PUBLIC_BASE_URL", "")

    # Memory
    mem0_api_key: str = os.getenv("MEM0_API_KEY", "")

    @field_validator("debug", mode="before")
    @classmethod
    def normalize_debug(cls, v):
        if isinstance(v, bool):
            return v
        raw = str(v or "").strip().lower()
        if raw in {"1", "true", "yes", "on", "dev", "debug"}:
            return True
        if raw in {"0", "false", "no", "off", "release", "prod", "production"}:
            return False
        return True

    @property
    def resolved_database_url(self) -> str:
        """Return PostgreSQL URL when postgres_user is set, otherwise SQLite fallback."""
        if self.postgres_user:
            return (
                f"postgresql://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}"
            )
        return self.database_url

    def ensure_storage_paths(self) -> None:
        """Create storage directories if they don't exist"""
        self.storage_projects_path.mkdir(parents=True, exist_ok=True)
        self.storage_temp_path.mkdir(parents=True, exist_ok=True)

    def validate_openai_env(self) -> None:
        """Check OpenAI configuration. Logs warning if missing; non-fatal since users can configure their own models."""
        import logging
        logger = logging.getLogger(__name__)
        required = {
            "OPENAI_API_KEY": self.openai_api_key,
            "OPENAI_BASE_URL": self.openai_base_url,
            "OPENAI_MODEL": self.openai_model,
        }
        missing = [name for name, value in required.items() if not str(value or "").strip()]
        if missing:
            missing_text = ", ".join(missing)
            logger.warning(
                "Missing OpenAI environment variables: %s. "
                "Some LLM features may not work until configured in .env",
                missing_text,
            )


# Global settings instance
settings = Settings()
