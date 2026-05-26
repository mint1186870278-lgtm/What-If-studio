"""Multi-agent discussion orchestration — AutoGen (legacy) + LangGraph (new)."""

from .autogen_service import (
    dispatch_agent,
    dispatch_autogen_service,
    run_autogen_discussion,
    run_autogen_discussion_stream,
    run_debate,
    run_debate_stream,
)

try:
    from .langgraph_service import (
        run_langgraph_discussion_stream,
        resume_langgraph_discussion_stream,
        build_discussion_graph,
        get_discussion_state,
    )
    _langgraph_ok = True
except Exception:
    _langgraph_ok = False
    run_langgraph_discussion_stream = None  # type: ignore[assignment]
    resume_langgraph_discussion_stream = None  # type: ignore[assignment]
    build_discussion_graph = None  # type: ignore[assignment]
    get_discussion_state = None  # type: ignore[assignment]


__all__ = [
    # Legacy AutoGen
    "dispatch_autogen_service",
    "run_autogen_discussion",
    "run_autogen_discussion_stream",
    "dispatch_agent",
    "run_debate",
    "run_debate_stream",
    # New LangGraph (may be None if unavailable)
    "run_langgraph_discussion_stream",
    "resume_langgraph_discussion_stream",
    "build_discussion_graph",
    "get_discussion_state",
]
