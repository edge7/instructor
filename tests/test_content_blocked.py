"""Tests for Model Armor / content blocking support in GenAI parsers."""

import pytest

genai = pytest.importorskip("google.genai")

from unittest.mock import MagicMock
from pydantic import BaseModel
from instructor.core.exceptions import ContentBlockedError
from instructor.processing.function_calls import OpenAISchema, _check_genai_blocked


class SimpleModel(OpenAISchema, BaseModel):
    name: str


def _make_blocked_response(
    block_reason="MODEL_ARMOR", block_message="Blocked by policy"
):
    """Create a mock GenAI response that simulates a Model Armor block."""
    response = MagicMock()
    response.candidates = []  # empty when blocked
    response.prompt_feedback = MagicMock()
    response.prompt_feedback.block_reason = block_reason
    response.prompt_feedback.block_reason_message = block_message
    return response


def _make_blocked_response_no_feedback():
    """Create a mock blocked response without prompt_feedback."""
    response = MagicMock()
    response.candidates = []
    response.prompt_feedback = None
    return response


# --- _check_genai_blocked tests ---


def test_check_genai_blocked_raises_on_empty_candidates():
    response = _make_blocked_response()
    with pytest.raises(ContentBlockedError) as exc_info:
        _check_genai_blocked(response)
    assert exc_info.value.block_reason == "MODEL_ARMOR"
    assert exc_info.value.block_message == "Blocked by policy"


def test_check_genai_blocked_raises_without_feedback():
    response = _make_blocked_response_no_feedback()
    with pytest.raises(ContentBlockedError) as exc_info:
        _check_genai_blocked(response)
    assert exc_info.value.block_reason is None


def test_check_genai_blocked_passes_valid_response():
    response = MagicMock()
    response.candidates = [MagicMock()]  # has candidates
    # Should not raise
    _check_genai_blocked(response)


# --- Parser tests ---


def test_parse_genai_tools_raises_content_blocked():
    response = _make_blocked_response()
    # Make it pass the isinstance check
    from google.genai import types

    response.__class__ = types.GenerateContentResponse
    with pytest.raises(ContentBlockedError):
        SimpleModel.parse_genai_tools(response)


def test_parse_genai_structured_outputs_raises_content_blocked():
    response = _make_blocked_response()
    with pytest.raises(ContentBlockedError):
        SimpleModel.parse_genai_structured_outputs(response)


def test_parse_gemini_json_raises_content_blocked():
    response = _make_blocked_response()
    with pytest.raises(ContentBlockedError):
        SimpleModel.parse_gemini_json(response)
