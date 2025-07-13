import pytest
from pydantic import BaseModel
import instructor

from .util import models, modes


class Listing(BaseModel):
    title: str
    url: str


@pytest.mark.parametrize("model", models)
@pytest.mark.parametrize("mode", modes)
def test_create_with_completion_list(client, model, mode):
    inst = instructor.from_genai(client, mode=mode)
    items, raw = inst.chat.completions.create_with_completion(
        model=model,
        messages=[
            {"role": "user", "content": "Give two blog post titles with URLs about AI"}
        ],
        response_model=list[Listing],
    )
    assert isinstance(items, list)
    assert len(items) == 2
    assert all(isinstance(i, Listing) for i in items)
    assert raw is not None
