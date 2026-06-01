from enum import Enum


class Domain(str, Enum):
    CODING_AGENT_DEBUGGING = "coding_agent_debugging"
    RESEARCH_AGENT_MEMORY = "research_agent_memory"
    PERSONAL_PREFERENCE_MEMORY = "personal_preference_memory"
    CALENDAR_WORKFLOW = "calendar_workflow"
    TOOL_USE_ASSISTANT = "tool_use_assistant"
    MULTI_AGENT_KNOWLEDGE_BASE = "multi_agent_knowledge_base"


class ProbeType(str, Enum):
    STATE_RESOLUTION = "state_resolution"
    PREMISE_RESISTANCE = "premise_resistance"
    POLICY_ADAPTATION = "policy_adaptation"
    AUDIT_LOCALIZATION = "audit_localization"


class RevisionFamily(str, Enum):
    SUPERSEDES = "supersedes"
    BLOCKS = "blocks"
    RELEASES = "releases"
    UNCERTAIN = "uncertain"
    REAFFIRMS = "reaffirms"
    NO_REVISION = "no_revision"
    MIXED_MULTI_ACTION = "mixed_multi_action"


class RevisionActionType(str, Enum):
    SUPERSEDES = "SUPERSEDES"
    BLOCKS = "BLOCKS"
    RELEASES = "RELEASES"
    UNCERTAIN = "UNCERTAIN"
    REAFFIRMS = "REAFFIRMS"
    NO_REVISION = "NO_REVISION"


class FinalStatus(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    SUPERSEDED = "SUPERSEDED"
    BLOCKED = "BLOCKED"
    UNRESOLVED = "UNRESOLVED"
