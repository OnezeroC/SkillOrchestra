"""Prompt templates for SkillOrchestra.

Centralized prompts for:
- learning: handbook discovery, refinement, profiler
- model_routing: QA benchmarks (skill-based and baseline routing)
"""

from .learning import (
    SKILL_DISCOVERY_PROMPT,
    SKILL_IDENTIFICATION_PROMPT,
    MODE_INSIGHT_PROMPT,
    PROFILE_SUMMARY_PROMPT,
    SKILL_SPLIT_PROMPT,
    SKILL_MERGE_PROMPT,
    FAILURE_DRIVEN_REFINEMENT_PROMPT,
)
from .model_routing import SKILL_ANALYSIS_PROMPT, BASELINE_PROMPT

__all__ = [
    # Learning
    "SKILL_DISCOVERY_PROMPT",
    "SKILL_IDENTIFICATION_PROMPT",
    "MODE_INSIGHT_PROMPT",
    "PROFILE_SUMMARY_PROMPT",
    "SKILL_SPLIT_PROMPT",
    "SKILL_MERGE_PROMPT",
    "FAILURE_DRIVEN_REFINEMENT_PROMPT",
    # Model routing
    "SKILL_ANALYSIS_PROMPT",
    "BASELINE_PROMPT",
]
