from __future__ import annotations

from typing import Any, Dict

from langchain_core.messages import AIMessage


class AgentMessage(AIMessage):
    """
    AIMessage variant that keeps structured_output inside additional kwargs while exposing
    a user-friendly accessor.
    """

    def __init__(
        self,
        *,
        name: str,
        structured_output: Dict[str, Any] | None = None,
        **kwargs: Any,
    ):
        structured_output = structured_output or {}
        init_kwargs = dict(kwargs)
        import json
        # Use json.dumps to ensure the model sees a clear string representation of the dict
        content = init_kwargs.pop("content", json.dumps(structured_output, indent=2))
        additional_kwargs = dict(init_kwargs.pop("additional_kwargs", {}))
        additional_kwargs["structured_output"] = structured_output
        if additional_kwargs:
            init_kwargs["additional_kwargs"] = additional_kwargs
        super().__init__(
            content=content,
            name=name,
            **init_kwargs,
        )

    @property
    def structured_output(self) -> Dict[str, Any]:
        additional_kwargs = getattr(self, "additional_kwargs", {}) or {}
        structured = additional_kwargs.get("structured_output")
        if isinstance(structured, dict):
            return dict(structured)
        return {}
