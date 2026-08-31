"""Engine V2 research components.

Nothing in this package is imported by the frozen V1 API.  Promotion into a
live service requires a separate, explicit deployment decision.
"""

from .validity_boundary import (  # noqa: F401
    BoundaryDecision,
    EngineV2Request,
    ModuleDecision,
    evaluate_request,
)
