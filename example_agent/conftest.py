import pytest
import asyncio
import sys
import os
from pathlib import Path

# Ensure each test has a clean event loop
@pytest.fixture(scope="function")
def event_loop():
    """Create an instance of the default event loop for each test."""
    loop = asyncio.new_event_loop()
    asyncio.set_event_loop(loop)
    yield loop
    
    # Clean up pending tasks
    pending_tasks = asyncio.all_tasks(loop)
    if pending_tasks:
        # Cancel all tasks
        for task in pending_tasks:
            task.cancel()
        
        # Allow cancelled tasks to complete their cancellation
        if sys.version_info >= (3, 7):
            loop.run_until_complete(asyncio.gather(*pending_tasks, return_exceptions=True))
    
    # Close the loop explicitly
    loop.run_until_complete(loop.shutdown_asyncgens())
    loop.close()

# Set a test timeout for asyncio tests
@pytest.fixture(autouse=True)
def set_asyncio_timeout(monkeypatch):
    """Set timeout for asyncio tests to handle subprocess creation/teardown"""
    monkeypatch.setenv("PYTEST_TIMEOUT", "60") 