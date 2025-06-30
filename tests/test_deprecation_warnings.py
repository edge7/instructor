import sys
import types
import warnings

import pytest  # type: ignore

# Import the module under test with optional stubbing of the external dependency.


def _ensure_vertexai_stub():
    """Create a minimal stub for the `vertexai.generative_models` package if it's absent."""
    if "vertexai.generative_models" in sys.modules:
        return sys.modules["vertexai.generative_models"]

    # Build stub hierarchy: vertexai -> vertexai.generative_models
    vertexai_mod = types.ModuleType("vertexai")
    gm_mod = types.ModuleType("vertexai.generative_models")

    class _StubGenerativeModel:  # noqa: D401, E501
        def __init__(self, *args, **kwargs):
            pass

        def generate_content(self, *args, **kwargs):  # noqa: D401
            return None

        async def generate_content_async(self, *args, **kwargs):  # noqa: D401
            return None

    gm_mod.GenerativeModel = _StubGenerativeModel  # type: ignore[attr-defined]

    # Register modules
    sys.modules["vertexai"] = vertexai_mod
    sys.modules["vertexai.generative_models"] = gm_mod


_ensure_vertexai_stub()


from instructor.client_vertexai import from_vertexai  # noqa: E402
import vertexai.generative_models as gm  # type: ignore # noqa: E402, E501


pytestmark = pytest.mark.filterwarnings("ignore::FutureWarning")


@pytest.fixture()
def vertex_model():
    """Return a minimal GenerativeModel instance (real or stub)."""
    return gm.GenerativeModel("dummy-model")  # type: ignore[call-arg]


def test_async_deprecated__async(vertex_model):
    """Using `_async=True` should raise a DeprecationWarning."""
    with pytest.warns(DeprecationWarning):
        from_vertexai(vertex_model, _async=True)


def test_async_deprecated_use_async(vertex_model):
    """Using `use_async=True` should raise a DeprecationWarning."""
    with pytest.warns(DeprecationWarning):
        from_vertexai(vertex_model, use_async=True)


def test_async_client_no_warning(vertex_model):
    """Preferred flag should not trigger any deprecation warnings."""
    with warnings.catch_warnings(record=True) as rec:
        warnings.simplefilter("always")
        from_vertexai(vertex_model, async_client=True)
        assert not any(isinstance(w.message, DeprecationWarning) for w in rec)