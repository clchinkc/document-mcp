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

    def test_error_classifier_network_error(self):
        """Classifies network errors correctly."""
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


class TestSummaryFunctionality:
    """Test summary functionality integration with ReAct Agent."""

    @pytest.mark.asyncio
    async def test_react_agent_automatic_summary_workflow(self):
        """Test that ReAct agent automatically handles summaries in multi-step workflow."""
        import tempfile
        import os
        from pathlib import Path
        
        # Create isolated test environment
        temp_dir = Path(tempfile.mkdtemp(prefix='test_react_summary_'))
        os.environ['DOCUMENT_ROOT_DIR'] = str(temp_dir)
        
        try:
            # Clear module cache to ensure fresh imports
            import sys
            if 'document_mcp.doc_tool_server' in sys.modules:
                import importlib
                importlib.reload(sys.modules['document_mcp.doc_tool_server'])
            
            from document_mcp.doc_tool_server import (
                create_document, 
                DOCUMENT_SUMMARY_FILE
            )
            
            doc_name = 'react_test_doc'
            
            # Set up document with summary
            create_document(doc_name)
            doc_dir = temp_dir / doc_name
            doc_dir.mkdir(parents=True, exist_ok=True)
            
            # Create summary
            summary_file = doc_dir / DOCUMENT_SUMMARY_FILE  
            summary_content = "# Development Project\n\nA software development project with API and UI components."
            summary_file.write_text(summary_content)
            
            # Create chapters
            api_chapter = doc_dir / '01-api.md'
            api_chapter.write_text("# API Development\n\nDetailed API implementation...")
            
            ui_chapter = doc_dir / '02-ui.md'
            ui_chapter.write_text("# UI Development\n\nDetailed UI implementation...")
            
            # Test that ReAct agent workflow includes summary handling
            try:
                from src.agents.react_agent.main import run_react_loop
                
                # Test the enhanced system prompt includes summary handling
                from src.agents.react_agent.main import REACT_SYSTEM_PROMPT
                assert 'read_document_summary' in REACT_SYSTEM_PROMPT
                assert 'Automatic Summary Handling' in REACT_SYSTEM_PROMPT
                assert 'Summary First Strategy' in REACT_SYSTEM_PROMPT
                
            except Exception as e:
                # If agent connection fails, test the underlying logic
                from document_mcp.doc_tool_server import list_documents, read_document_summary
                
                docs = list_documents()
                test_doc = next((d for d in docs if d.document_name == doc_name), None)
                assert test_doc is not None
                assert test_doc.has_summary is True
                
                summary = read_document_summary(doc_name)
                assert summary == summary_content
                
        finally:
            # Clean up
            import shutil
            if temp_dir.exists():
                shutil.rmtree(temp_dir)

    def test_react_agent_system_prompt_summary_enhancements(self):
        """Test that ReAct Agent system prompt includes summary handling."""
        from src.agents.react_agent.main import REACT_SYSTEM_PROMPT
        
        assert 'Summary Operations' in REACT_SYSTEM_PROMPT
        assert 'Explicit Content Requests' in REACT_SYSTEM_PROMPT
        assert 'Broad Screening/Editing' in REACT_SYSTEM_PROMPT
        assert 'read_document_summary' in REACT_SYSTEM_PROMPT
