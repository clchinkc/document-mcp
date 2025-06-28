import pytest
import time
from document_mcp import metrics_config


@pytest.fixture
def disabled_metrics(monkeypatch):
    """Fixture to ensure metrics are in disabled state for testing."""
    # Reset global state to disabled
    monkeypatch.setattr(metrics_config, 'METRICS_ENABLED', False)
    monkeypatch.setattr(metrics_config, 'meter', None)
    monkeypatch.setattr(metrics_config, 'tool_calls_counter', None)
    monkeypatch.setattr(metrics_config, 'tool_duration_histogram', None)
    monkeypatch.setattr(metrics_config, 'tool_errors_counter', None)
    monkeypatch.setattr(metrics_config, 'tool_argument_sizes_histogram', None)
    monkeypatch.setattr(metrics_config, 'concurrent_operations_gauge', None)
    monkeypatch.setattr(metrics_config, 'server_info_counter', None)
    monkeypatch.setattr(metrics_config, 'prometheus_reader', None)
    metrics_config._active_operations.clear()


@pytest.fixture
def mock_otel(monkeypatch, mocker):
    """Mocks all opentelemetry and prometheus objects for testing enabled state."""
    monkeypatch.setattr(metrics_config, 'OTEL_AVAILABLE', True)
    monkeypatch.setattr(metrics_config, 'METRICS_ENABLED', True)
    
    # Mock the top-level 'metrics' object and the 'meter' it returns
    mock_metrics_module = mocker.Mock()
    mock_meter = mocker.Mock()
    mock_metrics_module.get_meter.return_value = mock_meter
    monkeypatch.setattr(metrics_config, 'metrics', mock_metrics_module)
    
    # Since 'meter' is global, we patch it after initialization is simulated
    monkeypatch.setattr(metrics_config, 'meter', mock_meter)
    
    # Mock the classes that are conditionally imported
    monkeypatch.setattr(metrics_config, 'MeterProvider', mocker.Mock())
    monkeypatch.setattr(metrics_config, 'PrometheusMetricReader', mocker.Mock())
    monkeypatch.setattr(metrics_config, 'OTLPMetricExporter', mocker.Mock())
    monkeypatch.setattr(metrics_config, 'PeriodicExportingMetricReader', mocker.Mock())
    monkeypatch.setattr(metrics_config, 'FastAPIInstrumentor', mocker.Mock())
    monkeypatch.setattr(metrics_config, 'RequestsInstrumentor', mocker.Mock())
    monkeypatch.setattr(metrics_config, 'generate_latest', mocker.Mock(return_value=b"prometheus_data"))
    monkeypatch.setattr(metrics_config, 'CONTENT_TYPE_LATEST', "text/prometheus")

    # Mock the instrument objects that are created
    tool_calls_counter = mocker.Mock()
    tool_duration_histogram = mocker.Mock()
    tool_errors_counter = mocker.Mock()
    tool_argument_sizes_histogram = mocker.Mock()
    concurrent_operations_gauge = mocker.Mock()

    # Link mocks to the meter's creation methods
    def counter_side_effect(name, **kwargs):
        if name == "mcp_tool_calls_total": return tool_calls_counter
        if name == "mcp_tool_errors_total": return tool_errors_counter
        return mocker.Mock()
        
    mock_meter.create_counter.side_effect = counter_side_effect
    mock_meter.create_histogram.return_value = tool_argument_sizes_histogram
    mock_meter.create_up_down_counter.return_value = concurrent_operations_gauge
    
    # Set the global instrument variables to our mocks
    monkeypatch.setattr(metrics_config, 'tool_calls_counter', tool_calls_counter)
    monkeypatch.setattr(metrics_config, 'tool_duration_histogram', tool_duration_histogram)
    monkeypatch.setattr(metrics_config, 'tool_errors_counter', tool_errors_counter)
    monkeypatch.setattr(metrics_config, 'tool_argument_sizes_histogram', tool_argument_sizes_histogram)
    monkeypatch.setattr(metrics_config, 'concurrent_operations_gauge', concurrent_operations_gauge)
    
    yield
    
    metrics_config._active_operations.clear()


# --- Tests for Disabled State ---

def test_calculate_argument_size_json():
    args = ("a", 1, {"key": "value"})
    kwargs = {"x": True, "y": None}
    size = metrics_config.calculate_argument_size(args, kwargs)
    assert isinstance(size, int) and size > 0

def test_calculate_argument_size_fallback(monkeypatch, mocker):
    class BadJSON:
        def __repr__(self): return "bad"
    monkeypatch.setattr(metrics_config, 'json', type('J', (), {'dumps': staticmethod(lambda *a, **k: exec("raise TypeError"))}))
    size = metrics_config.calculate_argument_size((BadJSON(),), {})
    assert size == len(repr((BadJSON(),)).encode('utf-8')) + len(repr({}).encode('utf-8'))

def test_is_metrics_enabled_initially_false(disabled_metrics):
    assert not metrics_config.is_metrics_enabled()

def test_get_metrics_export_disabled(disabled_metrics):
    data, content_type = metrics_config.get_metrics_export()
    assert "not available" in data
    assert content_type == "text/plain"

def test_get_metrics_summary_disabled(disabled_metrics):
    summary = metrics_config.get_metrics_summary()
    assert summary["status"] == "disabled"

def test_instrument_tool_decorator(disabled_metrics, monkeypatch, mocker):
    start_mock = mocker.Mock(return_value=None)  # Return None for disabled state
    success_mock = mocker.Mock()
    error_mock = mocker.Mock()
    monkeypatch.setattr(metrics_config, 'record_tool_call_start', start_mock)
    monkeypatch.setattr(metrics_config, 'record_tool_call_success', success_mock)
    monkeypatch.setattr(metrics_config, 'record_tool_call_error', error_mock)

    @metrics_config.instrument_tool
    def success_func(a, b): return a + b
    
    @metrics_config.instrument_tool
    def error_func(): raise ValueError("test")

    # Test success path
    assert success_func(1, 2) == 3
    start_mock.assert_called_with('success_func', (1, 2), {})
    success_mock.assert_called_with('success_func', None, 3)
    error_mock.assert_not_called()

    # Test error path
    start_mock.reset_mock()
    success_mock.reset_mock()
    with pytest.raises(ValueError):
        error_func()
    start_mock.assert_called_with('error_func', (), {})
    success_mock.assert_not_called()
    error_mock.assert_called_once()


# --- Tests for Enabled State ---

def test_initialize_metrics_enabled_no_otlp(mock_otel):
    metrics_config.initialize_metrics()
    metrics_config.PrometheusMetricReader.assert_called_once()
    metrics_config.MeterProvider.assert_called_once()
    metrics_config.metrics.set_meter_provider.assert_called_once()

def test_initialize_metrics_enabled_with_otlp(mock_otel, monkeypatch):
    monkeypatch.setattr(metrics_config, 'OTEL_ENDPOINT', "http://test.com")
    metrics_config.initialize_metrics()
    metrics_config.OTLPMetricExporter.assert_called_with(endpoint="http://test.com", headers={})
    metrics_config.PeriodicExportingMetricReader.assert_called_once()
    assert len(metrics_config.MeterProvider.call_args.kwargs['metric_readers']) == 2

def test_record_tool_calls(mock_otel, mocker):
    mocker.patch('time.time', side_effect=[100.0, 101.0, 200.0, 202.0])
    # Success
    start_time = metrics_config.record_tool_call_start("tool1", (), {})
    metrics_config.record_tool_call_success("tool1", start_time)
    metrics_config.tool_calls_counter.add.assert_called_with(1, {"tool_name": "tool1", "status": "success", "environment": "development"})
    metrics_config.tool_duration_histogram.record.assert_called_with(1.0, {"tool_name": "tool1", "status": "success"})
    
    # Error
    start_time_err = metrics_config.record_tool_call_start("tool2", (), {})
    metrics_config.record_tool_call_error("tool2", start_time_err, ValueError())
    metrics_config.tool_errors_counter.add.assert_called_with(1, {"tool_name": "tool2", "error_type": "ValueError", "environment": "development"})

def test_concurrent_gauge(mock_otel):
    start1 = metrics_config.record_tool_call_start("task1", (), {})
    metrics_config.concurrent_operations_gauge.add.assert_called_with(1, {"tool_name": "task1"})
    start2 = metrics_config.record_tool_call_start("task2", (), {})
    metrics_config.concurrent_operations_gauge.add.assert_called_with(1, {"tool_name": "task2"})
    metrics_config.record_tool_call_success("task1", start1)
    metrics_config.concurrent_operations_gauge.add.assert_called_with(-1, {"tool_name": "task1"})
    metrics_config.record_tool_call_error("task2", start2, Exception())
    metrics_config.concurrent_operations_gauge.add.assert_called_with(-1, {"tool_name": "task2"})

def test_get_metrics_export_enabled(mock_otel):
    data, content_type = metrics_config.get_metrics_export()
    metrics_config.generate_latest.assert_called_once()
    assert data == "prometheus_data"
    assert content_type == "text/prometheus"

def test_get_metrics_summary_enabled(mock_otel):
    summary = metrics_config.get_metrics_summary()
    assert summary["status"] == "enabled" 