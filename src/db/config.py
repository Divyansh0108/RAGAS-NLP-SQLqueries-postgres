"""
Database configuration - now centralized in src.config.settings.
This module is kept for backward compatibility.
"""

from src.config import get_settings

# Get settings instance
_settings = get_settings()

# Expose database config for backward compatibility
db_config = type(
    "DatabaseConfig",
    (),
    {
        "host": _settings.db_host,
        "port": _settings.db_port,
        "user": _settings.db_user,
        "password": _settings.db_password,
        "database": _settings.db_name,
        "connection_string": _settings.db_connection_string,
    },
)()
