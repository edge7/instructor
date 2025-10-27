import os
from typing import Literal, Union
from pydantic import BaseModel
import pytest
import instructor

from .util import modes


class UserExtract(BaseModel):
    name: str
    age: int


@pytest.mark.parametrize("mode", modes)
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_iterable_model(mode):
    client = instructor.from_provider("xai/grok-3-mini", mode=mode)
    model = client.chat.create_iterable(
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    count = 0
    for m in model:
        assert isinstance(m, UserExtract)
        count += 1
    assert count == 2


@pytest.mark.parametrize("mode", modes)
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
async def test_iterable_model_async(mode):
    client = instructor.from_provider("xai/grok-3-mini", mode=mode, async_client=True)
    model = client.chat.create_iterable(
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Make two up people"},
        ],
    )
    count = 0
    async for m in model:
        assert isinstance(m, UserExtract)
        count += 1
    assert count == 2


@pytest.mark.parametrize("mode", modes)
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_partial_model(mode):
    client = instructor.from_provider("xai/grok-3-mini", mode=mode)
    model = client.chat.create_partial(
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )
    count = 0
    for m in model:
        assert isinstance(m, UserExtract)
        count += 1
    assert count >= 1


@pytest.mark.parametrize("mode", modes)
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
async def test_partial_model_async(mode):
    client = instructor.from_provider("xai/grok-3-mini", mode=mode, async_client=True)
    model = client.chat.create_partial(
        response_model=UserExtract,
        max_retries=2,
        messages=[
            {"role": "user", "content": "Jason Liu is 12 years old"},
        ],
    )
    count = 0
    async for m in model:
        assert isinstance(m, UserExtract)
        count += 1
    assert count >= 1


class Weather(BaseModel):
    location: str
    units: Literal["imperial", "metric"]


class GoogleSearch(BaseModel):
    query: str


@pytest.mark.parametrize("mode", [instructor.Mode.XAI_JSON])
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
def test_iterable_create_union_model(mode):
    client = instructor.from_provider("xai/grok-3-mini", mode=mode)
    model = client.chat.create_iterable(
        max_retries=2,
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Union[Weather, GoogleSearch],
    )
    count = 0
    for m in model:
        assert isinstance(m, (Weather, GoogleSearch))
        count += 1
    assert count >= 1


@pytest.mark.parametrize("mode", [instructor.Mode.XAI_JSON])
@pytest.mark.asyncio
@pytest.mark.skipif(
    not os.environ.get("XAI_API_KEY") or os.environ.get("XAI_API_KEY") == "test",
    reason="XAI_API_KEY not set or invalid",
)
async def test_iterable_create_union_model_async(mode):
    client = instructor.from_provider("xai/grok-3-mini", mode=mode, async_client=True)
    model = client.chat.create_iterable(
        max_retries=2,
        messages=[
            {"role": "system", "content": "You must always use tools"},
            {
                "role": "user",
                "content": "What is the weather in toronto and dallas and who won the super bowl?",
            },
        ],
        response_model=Union[Weather, GoogleSearch],
    )
    count = 0
    async for m in model:
        assert isinstance(m, (Weather, GoogleSearch))
        count += 1
    assert count >= 1
