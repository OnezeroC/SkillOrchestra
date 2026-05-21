from .pool import POOL_MODEL_DISPLAY_NAMES


def resolve_model(agent_id: str) -> str:
    """Return agent_id unchanged (model routing uses model keys directly)."""
    return agent_id


__all__ = ["resolve_model", "POOL_MODEL_DISPLAY_NAMES"]
