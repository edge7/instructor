"""Plugin-like registry for provider implementations.

Sub-modules register themselves via the :pyfunc:`providers.base.register_provider`
decorator.  End-users generally do not need to import from this package
directly – it exists mainly so that :pycode:`import providers.client_openai` (and
friends) works.
"""