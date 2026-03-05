"""
Unit tests for configuration management.
"""

import pytest
from pathlib import Path
from pydantic import ValidationError

from src.config.settings import Settings, get_settings


def test_settings_default_values():
    """Test that settings have sensible default values."""
    settings = Settings()

    assert settings.db_host == "localhost"
    assert settings.db_port == 5432
    assert settings.default_llm_model in ["qwen2.5-coder", "codellama"]
    assert settings.llm_temperature == 0.0
    assert settings.n_schema_results == 5
    assert settings.n_example_results == 3


def test_settings_connection_string():
    """Test database connection string generation."""
    settings = Settings(
        db_user="testuser",
        db_password="testpass",
        db_host="testhost",
        db_port=5433,
        db_name="testdb",
    )

    expected = "postgresql://testuser:testpass@testhost:5433/testdb"
    assert settings.db_connection_string == expected


def test_settings_path_properties():
    """Test that path properties work correctly."""
    settings = Settings()

    assert isinstance(settings.project_root, Path)
    assert isinstance(settings.data_dir, Path)
    assert isinstance(settings.chroma_path, Path)
    assert isinstance(settings.schema_path, Path)
    assert isinstance(settings.examples_path, Path)

    # Verify path relationships
    assert settings.data_dir == settings.project_root / "data"
    assert settings.chroma_path == settings.data_dir / "chroma_db"


def test_settings_port_validation():
    """Test that invalid ports are rejected."""
    with pytest.raises(ValidationError):
        Settings(db_port=70000)  # Too high

    with pytest.raises(ValidationError):
        Settings(db_port=0)  # Too low


def test_settings_temperature_validation():
    """Test that invalid temperatures are rejected."""
    with pytest.raises(ValidationError):
        Settings(llm_temperature=3.0)  # Too high

    with pytest.raises(ValidationError):
        Settings(llm_temperature=-1.0)  # Too low


def test_settings_similarity_threshold_validation():
    """Test that invalid similarity thresholds are rejected."""
    with pytest.raises(ValidationError):
        Settings(similarity_threshold=1.5)  # Too high

    with pytest.raises(ValidationError):
        Settings(similarity_threshold=-0.5)  # Too low


def test_get_settings_singleton():
    """Test that get_settings returns the same instance."""
    settings1 = get_settings()
    settings2 = get_settings()

    assert settings1 is settings2


def test_settings_from_env(monkeypatch):
    """Test loading settings from environment variables."""
    monkeypatch.setenv("DB_HOST", "custom-host")
    monkeypatch.setenv("DB_PORT", "5433")
    monkeypatch.setenv("LLM_TEMPERATURE", "0.5")

    # Need to clear the cache to reload settings
    get_settings.cache_clear()
    settings = Settings()

    assert settings.db_host == "custom-host"
    assert settings.db_port == 5433
    assert settings.llm_temperature == 0.5

    # Clean up
    get_settings.cache_clear()
