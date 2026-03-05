"""
Rate limiting functionality to prevent abuse.
"""

import time
from collections import defaultdict
from typing import Dict, Optional

from src.exceptions import RateLimitError
from src.utils import get_logger

logger = get_logger(__name__)


class RateLimiter:
    """
    Simple in-memory rate limiter using sliding window algorithm.
    """

    def __init__(self, max_requests: int = 10, window_seconds: int = 60):
        """
        Initialize rate limiter.

        Args:
            max_requests: Maximum requests allowed in window
            window_seconds: Time window in seconds
        """
        self.max_requests = max_requests
        self.window_seconds = window_seconds
        self.requests: Dict[str, list] = defaultdict(list)
        logger.info(
            f"Rate limiter initialized: {max_requests} requests per {window_seconds}s"
        )

    def check_rate_limit(self, identifier: str) -> bool:
        """
        Check if request is allowed for given identifier.

        Args:
            identifier: Unique identifier (e.g., user_id, session_id)

        Returns:
            True if allowed, False if rate limit exceeded

        Raises:
            RateLimitError: If rate limit exceeded
        """
        current_time = time.time()

        # Clean up old requests
        self.requests[identifier] = [
            req_time
            for req_time in self.requests[identifier]
            if current_time - req_time < self.window_seconds
        ]

        # Check if limit exceeded
        if len(self.requests[identifier]) >= self.max_requests:
            wait_time = self.window_seconds - (
                current_time - self.requests[identifier][0]
            )
            logger.warning(
                f"Rate limit exceeded for {identifier}: "
                f"{len(self.requests[identifier])}/{self.max_requests} requests"
            )
            raise RateLimitError(
                f"Rate limit exceeded. Please wait {int(wait_time)}s before trying again."
            )

        # Record request
        self.requests[identifier].append(current_time)
        logger.debug(
            f"Request allowed for {identifier}: "
            f"{len(self.requests[identifier])}/{self.max_requests} requests used"
        )
        return True

    def reset(self, identifier: Optional[str] = None):
        """
        Reset rate limit for identifier or all identifiers.

        Args:
            identifier: Specific identifier to reset, or None for all
        """
        if identifier:
            if identifier in self.requests:
                del self.requests[identifier]
                logger.info(f"Rate limit reset for {identifier}")
        else:
            self.requests.clear()
            logger.info("Rate limit reset for all identifiers")

    def get_remaining(self, identifier: str) -> int:
        """
        Get remaining requests for identifier.

        Args:
            identifier: Unique identifier

        Returns:
            Number of remaining requests
        """
        current_time = time.time()

        # Clean up old requests
        self.requests[identifier] = [
            req_time
            for req_time in self.requests[identifier]
            if current_time - req_time < self.window_seconds
        ]

        return max(0, self.max_requests - len(self.requests[identifier]))


# Global rate limiter instance
rate_limiter = RateLimiter(max_requests=10, window_seconds=60)
