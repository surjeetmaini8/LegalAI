
import logging
import os
from typing import Optional

from dotenv import load_dotenv

logger = logging.getLogger(__name__)


class Config:

    def __init__(self):
        # Try to load from .env.mongodb first, then fall back to .env
        env_files = [".env.mongodb", ".env"]
        loaded = False

        for env_file in env_files:
            if os.path.exists(env_file):
                load_dotenv(env_file)
                logger.info(f" Loaded configuration from {env_file}")
                loaded = True
                break

        if not loaded:
            logger.warning("⚠️ No .env file found, using default values")

    
    @property
    def MONGODB_URL(self) -> str:
        url = os.getenv("MONGODB_URL")
        if not url:
            raise ValueError(
                "MONGODB_URL environment variable is required. Please set it in your .env file."
            )
        return url

    @property
    def DATABASE_NAME(self) -> str:
        return os.getenv("MONGODB_DATABASE", "dcm_system")

    @property
    def MONGODB_USERNAME(self) -> str:
        """MongoDB username (required for Atlas, optional for local)"""
        username = os.getenv("MONGODB_USERNAME", "")
        # If using Atlas (mongodb+srv), username is required
        if "mongodb+srv" in self.MONGODB_URL and not username:
            raise ValueError(
                "MONGODB_USERNAME is required for MongoDB Atlas connection"
            )
        return username

    @property
    def MONGODB_PASSWORD(self) -> str:
        """MongoDB password (required for Atlas, optional for local)"""
        password = os.getenv("MONGODB_PASSWORD", "")
        # If using Atlas (mongodb+srv), password is required
        if "mongodb+srv" in self.MONGODB_URL and not password:
            raise ValueError(
                "MONGODB_PASSWORD is required for MongoDB Atlas connection"
            )
        return password

    @property
    def JWT_SECRET_KEY(self) -> str:
        return os.getenv("JWT_SECRET_KEY", "dcm-super-secure-mongodb-key-2024")

    @property
    def JWT_ALGORITHM(self) -> str:
        return os.getenv("JWT_ALGORITHM", "HS256")

    @property
    def JWT_EXPIRATION_HOURS(self) -> int:
        return int(os.getenv("JWT_EXPIRATION_HOURS", "24"))

    @property
    def APP_NAME(self) -> str:
        return os.getenv("APP_NAME", "DCM System - MongoDB Edition")

    @property
    def APP_VERSION(self) -> str:
        return os.getenv("APP_VERSION", "2.0.0")

    @property
    def DEBUG(self) -> bool:
        return os.getenv("DEBUG", "true").lower() in ("true", "1", "yes", "on")

    @property
    def PORT(self) -> int:
        return int(os.getenv("PORT", "8001"))

    @property
    def HOST(self) -> str:
        return os.getenv("HOST", "0.0.0.0")

    @property
    def CORS_ORIGINS(self) -> list:
        origins_str = os.getenv(
            "CORS_ORIGINS",
            "http://localhost:3000,http://localhost:3001,http://localhost:3002,http://127.0.0.1:3000,http://127.0.0.1:3001,http://127.0.0.1:3002",
        )
        return [origin.strip() for origin in origins_str.split(",")]
    @property
    def CREATE_DEMO_DATA(self) -> bool:
        return os.getenv("CREATE_DEMO_DATA", "true").lower() in (
            "true",
            "1",
            "yes",
            "on",
        )

    @property
    def DEMO_USERS_COUNT(self) -> int:
        return int(os.getenv("DEMO_USERS_COUNT", "4"))

    @property
    def DEMO_CASES_COUNT(self) -> int:
        return int(os.getenv("DEMO_CASES_COUNT", "10"))

    @property
    def log_level(self) -> str:
        return os.getenv("LOG_LEVEL", "INFO")

    @property
    def EMAIL_ENABLED(self) -> bool:
        return os.getenv("EMAIL_ENABLED", "false").lower() in ("true", "1", "yes", "on")

    @property
    def SMTP_SERVER(self) -> Optional[str]:
        return os.getenv("SMTP_SERVER")

    @property
    def SMTP_PORT(self) -> int:
        return int(os.getenv("SMTP_PORT", "587"))

    @property
    def FROM_EMAIL(self) -> Optional[str]:
        return os.getenv("FROM_EMAIL")

    def validate(self) -> bool:
        """Validate that all required configuration is present"""
        required_configs = [
            ("MongoDB URL", self.MONGODB_URL),
            ("MongoDB Database", self.DATABASE_NAME),
            ("JWT Secret Key", self.JWT_SECRET_KEY),
        ]

        missing_configs = []
        for name, value in required_configs:
            if not value or value.strip() == "":
                missing_configs.append(name)

        if missing_configs:
            logger.error(
                f" Missing required configuration: {', '.join(missing_configs)}"
            )
            return False

        # Check if MongoDB password is still the placeholder
        if (
            "YOUR_PASSWORD_HERE" in self.MONGODB_URL
            or "<db_password>" in self.MONGODB_URL
        ):
            logger.error(" Please update MongoDB password in .env.mongodb file")
            return False

        logger.info(" Configuration validation passed")
        return True

    def print_config_summary(self):
       

config = Config()

if not config.validate():
    raise Exception(
        " Configuration validation failed. Please check your .env.mongodb file."
    )
