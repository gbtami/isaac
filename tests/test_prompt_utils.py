from acp.schema import EmbeddedResourceContentBlock, TextResourceContents

from isaac.agent.prompt_utils import coerce_user_text, extract_prompt_text


def test_extract_prompt_text_with_embedded_resource():
    resource = TextResourceContents(text="hello world", uri="file:///tmp/demo.txt")
    block = EmbeddedResourceContentBlock(resource=resource, type="resource")
    text = extract_prompt_text([block])
    assert "hello world" in text
    assert "[resource:" not in text


def test_coerce_user_text_with_embedded_resource():
    resource = TextResourceContents(text="hello world", uri="file:///tmp/demo.txt")
    block = EmbeddedResourceContentBlock(resource=resource, type="resource")
    text = coerce_user_text(block)
    assert text == "hello world"
