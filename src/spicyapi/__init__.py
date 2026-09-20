"""Official SpicyAPI client for Python.

The public surface is re-exported here, so ``from spicyapi import SpicyClient``
is the only import most programs need.
"""

from ._client import (
    ACTIVE_STATES,
    API_BASE_URL,
    TERMINAL_STATES,
    SpicyApiError,
    SpicyClient,
    SpicyTimeoutError,
    SpicyUploadError,
    SpicyWebhookError,
    compute_webhook_signature,
    is_terminal,
    output_assets,
    output_text,
    verify_webhook,
)

__all__ = [
    "ACTIVE_STATES",
    "API_BASE_URL",
    "TERMINAL_STATES",
    "SpicyApiError",
    "SpicyClient",
    "SpicyTimeoutError",
    "compute_webhook_signature",
    "SpicyUploadError",
    "SpicyWebhookError",
    "is_terminal",
    "output_assets",
    "output_text",
    "verify_webhook",
]

# _client.py is the single source of the version; this only re-exports it, and pyproject reads it
# from there too. Writing it in two places guarantees drift, and the symptom - a PyPI version that
# disagrees with __version__ - is among the most misleading things to debug.
from ._client import __version__ as __version__
