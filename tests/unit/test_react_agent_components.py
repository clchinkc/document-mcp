"""
Unit tests for React Agent components.

This module tests individual React Agent components like error classification,
circuit breakers, retry managers, and other isolated functionality.
"""

import pytest

from src.agents.react_agent.main import (
    ErrorClassifier,
    ErrorType,
    RetryManager,
    ServiceCircuitBreaker,
    get_circuit_breaker,
)


class TestErrorClassifier:
    """Test the error classification functionality."""

    def test_network_error_classification(self):
        """Test network error classification."""
        classifier = ErrorClassifier()
        network_error = Exception("Connection timeout occurred")
        error_info = classifier.classify(network_error)

        assert error_info.error_type == ErrorType.NETWORK_ERROR
        assert error_info.is_retryable is True
        assert error_info.max_retries > 0

    def test_authentication_error_classification(self):
        """Test authentication error classification."""
        classifier = ErrorClassifier()
        auth_error = Exception("Invalid API key provided")
        error_info = classifier.classify(auth_error)

        assert error_info.error_type == ErrorType.AUTHENTICATION_ERROR
        assert error_info.is_retryable is False
        assert error_info.max_retries == 0

    def test_rate_limit_error_classification(self):
        """Test rate limit error classification."""
        classifier = ErrorClassifier()
        rate_error = Exception("Rate limit exceeded, too many requests")
        error_info = classifier.classify(rate_error)

        assert error_info.error_type == ErrorType.RATE_LIMIT_ERROR
        assert error_info.is_retryable is True
        assert error_info.max_retries > 0

    def test_validation_error_classification(self):
        """Test validation error classification."""
        classifier = ErrorClassifier()
        validation_error = Exception("Invalid input format provided")
        error_info = classifier.classify(validation_error)

        assert error_info.error_type == ErrorType.VALIDATION_ERROR
        assert error_info.is_retryable is False

    def test_unknown_error_classification(self):
        """Test unknown error classification."""
        classifier = ErrorClassifier()
        unknown_error = Exception("Something unexpected happened")
        error_info = classifier.classify(unknown_error)

        assert error_info.error_type == ErrorType.UNKNOWN_ERROR
        assert error_info.is_retryable is True


class TestServiceCircuitBreaker:
    """Test the service circuit breaker functionality."""

    @pytest.mark.asyncio
    async def test_successful_calls(self):
        """Test circuit breaker with successful calls."""
        circuit_breaker = ServiceCircuitBreaker("test_service", failure_threshold=3)

        async def successful_func():
            return "success"

        result = await circuit_breaker.call(successful_func)
        assert result == "success"
        assert circuit_breaker.state == "CLOSED"

    @pytest.mark.asyncio
    async def test_failure_accumulation(self):
        """Test circuit breaker failure accumulation and state change."""
        circuit_breaker = ServiceCircuitBreaker("test_service", failure_threshold=3)

        async def failing_func():
            raise Exception("Service unavailable")

        # Accumulate failures to trigger circuit breaker
        for i in range(3):
            try:
                await circuit_breaker.call(failing_func)
            except Exception:
                pass

        # Circuit should now be OPEN
        assert circuit_breaker.state == "OPEN"

    @pytest.mark.asyncio
    async def test_circuit_breaker_blocks_when_open(self):
        """Test that circuit breaker blocks calls when OPEN."""
        circuit_breaker = ServiceCircuitBreaker("test_service", failure_threshold=3)

        async def failing_func():
            raise Exception("Service unavailable")

        async def successful_func():
            return "success"

        # Trigger circuit breaker to OPEN state
        for i in range(3):
            try:
                await circuit_breaker.call(failing_func)
            except Exception:
                pass

        # Circuit should now block calls
        try:
            await circuit_breaker.call(successful_func)
            assert False, "Circuit breaker should have blocked the call"
        except Exception as e:
            assert "Circuit breaker" in str(e) and "OPEN" in str(e)


class TestRetryManager:
    """Test the retry manager functionality."""

    @pytest.mark.asyncio
    async def test_successful_function_no_retries(self):
        """Test retry manager with function that succeeds on first try."""
        retry_manager = RetryManager()

        call_count = 0

        async def successful_func():
            nonlocal call_count
            call_count += 1
            return "success"

        result = await retry_manager.execute_with_retry(successful_func)
        assert result == "success"
        assert call_count == 1

    @pytest.mark.asyncio
    async def test_eventually_successful_function(self):
        """Test retry manager with function that fails then succeeds."""
        retry_manager = RetryManager()

        call_count = 0

        async def eventually_successful_func():
            nonlocal call_count
            call_count += 1
            if (
                call_count < 2
            ):  # Succeed on second attempt to match retry manager's logic
                raise Exception("Connection timeout occurred")  # Use a retryable error
            return "success"

        result = await retry_manager.execute_with_retry(eventually_successful_func)
        assert result == "success"
        assert call_count == 2

    @pytest.mark.asyncio
    async def test_always_failing_function(self):
        """Test retry manager with function that always fails."""
        retry_manager = RetryManager()

        call_count = 0

        async def always_failing_func():
            nonlocal call_count
            call_count += 1
            raise Exception("Permanent failure")

        try:
            await retry_manager.execute_with_retry(always_failing_func)
            assert False, "Should have raised an exception"
        except Exception as e:
            assert "Permanent failure" in str(e)
            # Should have tried multiple times
            assert call_count > 1


class TestCircuitBreakerSingleton:
    """Test circuit breaker singleton management."""

    def test_same_service_returns_same_instance(self):
        """Test that same service name returns same circuit breaker instance."""
        cb1 = get_circuit_breaker("test_service")
        cb2 = get_circuit_breaker("test_service")
        assert cb1 is cb2

    def test_different_services_return_different_instances(self):
        """Test that different service names return different instances."""
        cb1 = get_circuit_breaker("test_service")
        cb3 = get_circuit_breaker("other_service")
        assert cb1 is not cb3
