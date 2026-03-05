"""
Unit tests for rate limiter.
"""

import pytest
import time

from src.exceptions import RateLimitError
from src.utils.rate_limiter import RateLimiter


class TestRateLimiter:
    """Tests for rate limiter functionality."""

    def test_rate_limiter_initialization(self):
        """Test rate limiter initializes correctly."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        assert limiter.max_requests == 5
        assert limiter.window_seconds == 60

    def test_allow_requests_within_limit(self):
        """Test that requests within limit are allowed."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        identifier = "user_123"

        for i in range(5):
            assert limiter.check_rate_limit(identifier) is True

    def test_block_requests_exceeding_limit(self):
        """Test that requests exceeding limit are blocked."""
        limiter = RateLimiter(max_requests=3, window_seconds=60)
        identifier = "user_123"

        # Use up allowance
        for i in range(3):
            limiter.check_rate_limit(identifier)

        # Next request should be blocked
        with pytest.raises(RateLimitError):
            limiter.check_rate_limit(identifier)

    def test_different_identifiers_independent(self):
        """Test that different identifiers have independent limits."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        assert limiter.check_rate_limit("user_1") is True
        assert limiter.check_rate_limit("user_1") is True

        assert limiter.check_rate_limit("user_2") is True
        assert limiter.check_rate_limit("user_2") is True

        # user_1 should be blocked
        with pytest.raises(RateLimitError):
            limiter.check_rate_limit("user_1")

        # user_2 should be blocked
        with pytest.raises(RateLimitError):
            limiter.check_rate_limit("user_2")

    def test_requests_expire_after_window(self):
        """Test that old requests are cleaned up after window."""
        limiter = RateLimiter(max_requests=2, window_seconds=1)
        identifier = "user_123"

        # Use up allowance
        limiter.check_rate_limit(identifier)
        limiter.check_rate_limit(identifier)

        # Wait for window to expire
        time.sleep(1.1)

        # Should be allowed again
        assert limiter.check_rate_limit(identifier) is True

    def test_reset_specific_identifier(self):
        """Test resetting rate limit for specific identifier."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        limiter.check_rate_limit("user_1")
        limiter.check_rate_limit("user_1")
        limiter.check_rate_limit("user_2")

        # Reset user_1
        limiter.reset("user_1")

        # user_1 should be allowed again
        assert limiter.check_rate_limit("user_1") is True

        # user_2 should still have 1 request used
        assert limiter.get_remaining("user_2") == 1

    def test_reset_all_identifiers(self):
        """Test resetting rate limit for all identifiers."""
        limiter = RateLimiter(max_requests=2, window_seconds=60)

        limiter.check_rate_limit("user_1")
        limiter.check_rate_limit("user_2")

        # Reset all
        limiter.reset()

        # Both should have full allowance
        assert limiter.get_remaining("user_1") == 2
        assert limiter.get_remaining("user_2") == 2

    def test_get_remaining_requests(self):
        """Test getting remaining requests count."""
        limiter = RateLimiter(max_requests=5, window_seconds=60)
        identifier = "user_123"

        assert limiter.get_remaining(identifier) == 5

        limiter.check_rate_limit(identifier)
        assert limiter.get_remaining(identifier) == 4

        limiter.check_rate_limit(identifier)
        limiter.check_rate_limit(identifier)
        assert limiter.get_remaining(identifier) == 2

    def test_rate_limit_error_message(self):
        """Test that rate limit error has useful message."""
        limiter = RateLimiter(max_requests=1, window_seconds=60)
        identifier = "user_123"

        limiter.check_rate_limit(identifier)

        try:
            limiter.check_rate_limit(identifier)
            assert False, "Should have raised RateLimitError"
        except RateLimitError as e:
            assert "Rate limit exceeded" in str(e)
            assert "wait" in str(e).lower()
