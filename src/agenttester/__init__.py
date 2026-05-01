"""AgentTester: Multi-agent comparison tool.

Usable as a CLI (``agenttester run …``), a Docker container, or a Python
library::

    from agenttester import Orchestrator, AgentConfig, load_config
"""

from .agent_runner import AgentResult, run_agent
from .config import AgentConfig, load_config
from .git_manager import DiffStats, GitManager
from .orchestrator import Orchestrator

__all__ = [
    "AgentConfig",
    "AgentResult",
    "DiffStats",
    "GitManager",
    "Orchestrator",
    "load_config",
    "run_agent",
]

__version__ = "0.1.0"
