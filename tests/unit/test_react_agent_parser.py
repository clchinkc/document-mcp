import pytest
from src.agents.react_agent.parser import ActionParser

@pytest.fixture
def parser():
    return ActionParser()

def test_simple_action(parser):
    action_string = 'create_document(document_name="My Book")'
    tool_name, kwargs = parser.parse(action_string)
    assert tool_name == "create_document"
    assert kwargs == {"document_name": "My Book"}

def test_action_with_multiple_args(parser):
    action_string = 'create_chapter(document_name="My Book", chapter_name="01-introduction.md")'
    tool_name, kwargs = parser.parse(action_string)
    assert tool_name == "create_chapter"
    assert kwargs == {"document_name": "My Book", "chapter_name": "01-introduction.md"}

def test_action_with_no_args(parser):
    action_string = 'list_documents()'
    tool_name, kwargs = parser.parse(action_string)
    assert tool_name == "list_documents"
    assert kwargs == {}

def test_malformed_action_no_parentheses(parser):
    with pytest.raises(ValueError):
        parser.parse("create_document")

def test_malformed_action_missing_closing_parenthesis(parser):
    with pytest.raises(ValueError):
        parser.parse('create_document(document_name="My Book"')

def test_action_with_integer_arg(parser):
    action_string = 'delete_paragraph(document_name="My Book", chapter_name="01-intro.md", paragraph_index=2)'
    tool_name, kwargs = parser.parse(action_string)
    assert tool_name == "delete_paragraph"
    assert kwargs == {"document_name": "My Book", "chapter_name": "01-intro.md", "paragraph_index": 2}

def test_action_with_boolean_arg(parser):
    action_string = 'find_text_in_chapter(document_name="My Book", chapter_name="01-intro.md", query="search term", case_sensitive=false)'
    tool_name, kwargs = parser.parse(action_string)
    assert tool_name == "find_text_in_chapter"
    assert kwargs == {"document_name": "My Book", "chapter_name": "01-intro.md", "query": "search term", "case_sensitive": False}

def test_action_with_nested_quotes(parser):
    action_string = 'replace_text_in_chapter(document_name="My Book", chapter_name="01-intro.md", text_to_find="old \'text\'", replacement_text="new \'text\'")'
    tool_name, kwargs = parser.parse(action_string)
    assert tool_name == "replace_text_in_chapter"
    assert kwargs == {"document_name": "My Book", "chapter_name": "01-intro.md", "text_to_find": "old 'text'", "replacement_text": "new 'text'"}


