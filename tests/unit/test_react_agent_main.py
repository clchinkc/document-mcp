import pytest
import os
import random
import asyncio
from src.agents.react_agent.main import (
    parse_action_string,
    HistoryContextBuilder,
    get_circuit_breaker,
    ServiceCircuitBreaker,
    ErrorClassifier,
    RetryManager,
    execute_mcp_tool_directly,
    get_cached_agent,
)
from pydantic_ai import Agent


def test_parse_action_string_no_args():
    name, kwargs = parse_action_string('foo()')
    assert name == 'foo'
    assert kwargs == {}


def test_parse_action_string_with_args():
    action = 'tool(arg1="val1",arg2=2,flag=true,none=None)'
    name, kwargs = parse_action_string(action)
    assert name == 'tool'
    assert kwargs.get('arg1') == 'val1'
    assert kwargs.get('arg2') == 2
    assert kwargs.get('flag') is True
    assert kwargs.get('none') is None


def test_parse_action_string_invalid():
    with pytest.raises(ValueError):
        parse_action_string('not a func')


def test_history_context_builder():
    builder = HistoryContextBuilder()
    assert builder.get_context() == ''
    step_data = {'step': 1, 'thought': 't', 'action': 'a', 'observation': 'o'}
    builder.add_step(step_data)
    ctx = builder.get_context()
    assert '**Previous Step 1:**' in ctx
    # cached
    assert builder.get_context() == ctx
    builder.clear()
    assert builder.get_context() == ''


def test_get_circuit_breaker_singleton():
    cb1 = get_circuit_breaker('svc')
    cb2 = get_circuit_breaker('svc')
    assert cb1 is cb2
    assert isinstance(cb1, ServiceCircuitBreaker)


def test_error_classifier_patterns():
    classifier = ErrorClassifier()
    cases = [
        ('connection timed out', 'NETWORK_ERROR'),
        ('401 unauthorized', 'AUTHENTICATION_ERROR'),
        ('rate limit reached', 'RATE_LIMIT_ERROR'),
        ('invalid format', 'VALIDATION_ERROR'),
        ('tool execution failure', 'TOOL_ERROR'),
        ('llm generation error', 'LLM_ERROR'),
        ('some other error', 'UNKNOWN_ERROR'),
    ]
    for msg, expected in cases:
        info = classifier.classify(Exception(msg))
        assert info.error_type.name == expected


def test_retry_manager_calculate_delay():
    random.seed(0)
    rm = RetryManager()
    delay = rm._calculate_delay(2, initial_delay=1.0, max_delay=5.0)
    assert delay >= 0.1 and delay <= 5.0

@pytest.mark.asyncio
async def test_execute_mcp_tool_directly_test_env(monkeypatch):
    # Test test environment path
    os.environ['PYTEST_CURRENT_TEST'] = '1'
    class DummyAgent:
        async def run(self, prompt):
            class Result:
                class Output:
                    def model_dump(self):
                        return {'ok': True}
                output = Output()
            return Result()
    result = await execute_mcp_tool_directly(DummyAgent(), 'tool', {'a': 'b'})
    assert 'ok' in result
    del os.environ['PYTEST_CURRENT_TEST']

@pytest.mark.asyncio
async def test_get_cached_agent(monkeypatch):
    # Clear cache
    from src.agents.react_agent.main import _agent_cache
    _agent_cache.clear()
    dummy_server = object()
    # Monkeypatch load_llm_config
    async def fake_load(): return 'model'
    monkeypatch.setattr('src.agents.react_agent.main.load_llm_config', fake_load)
    # Monkeypatch Agent to capture args
    class DummyAgentClass:
        def __init__(self, model, mcp_servers, system_prompt, output_type):
            self.model = model
            self.mcp_servers = mcp_servers
            self.system_prompt = system_prompt
            self.output_type = output_type
    monkeypatch.setattr('src.agents.react_agent.main.Agent', DummyAgentClass)
    agent1 = await get_cached_agent('type', 'prompt', dummy_server)
    agent2 = await get_cached_agent('type', 'prompt', dummy_server)
    assert agent1 is agent2
    assert agent1.model == 'model'
    assert agent1.mcp_servers == [dummy_server] 