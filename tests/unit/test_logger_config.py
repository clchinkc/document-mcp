import pytest
import logging
from document_mcp import logger_config


def test_log_mcp_call_decorator_basic(monkeypatch, caplog):
    # Ensure metrics are disabled to simplify behavior
    monkeypatch.setattr(logger_config, 'METRICS_AVAILABLE', False)

    logger = logging.getLogger('mcp_call_logger')
    monkeypatch.setattr(logger, 'propagate', True)

    caplog.set_level(logging.INFO)

    @logger_config.log_mcp_call
    def dummy(a, b=2):
        return a + b

    result = dummy(3, b=4)
    assert result == 7
    # Validate log messages
    records = [r for r in caplog.records if r.name == 'mcp_call_logger']
    assert len(records) >= 2
    assert 'Calling tool: dummy' in records[0].getMessage()
    assert 'Tool dummy returned:' in records[1].getMessage()


def test_log_mcp_call_exception(monkeypatch, caplog):
    # Disable metrics for simplicity
    monkeypatch.setattr(logger_config, 'METRICS_AVAILABLE', False)
    
    logger = logging.getLogger('mcp_call_logger')
    monkeypatch.setattr(logger, 'propagate', True)

    caplog.set_level(logging.INFO)

    @logger_config.log_mcp_call
    def dummy_err():
        raise ValueError('fail')

    with pytest.raises(ValueError):
        dummy_err()

    # Check that error was logged
    records = [r for r in caplog.records if r.name == 'mcp_call_logger' and r.levelno == logging.ERROR]
    assert len(records) == 1
    assert 'Tool dummy_err raised exception: fail' in records[0].getMessage() 