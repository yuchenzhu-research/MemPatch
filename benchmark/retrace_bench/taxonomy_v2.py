from enum import Enum


class DomainV2(str, Enum):
    SOFTWARE_ENGINEERING_AGENT = "software_engineering_agent"
    ENTERPRISE_MULTI_TOOL_WORKFLOW = "enterprise_multi_tool_workflow"
    CUSTOMER_SUPPORT_CRM = "customer_support_crm"
    CALENDAR_TASK_WORKFLOW = "calendar_task_workflow"
    RESEARCH_KNOWLEDGE_WORK = "research_knowledge_work"
    PERSONAL_ASSISTANT_PREFERENCE = "personal_assistant_preference"
    ECOMMERCE_RECOMMENDATION = "ecommerce_recommendation"
    DATA_ANALYSIS_BI = "data_analysis_bi"


class ReliabilityFailureModeV2(str, Enum):
    STALE_MEMORY_REUSE = "stale_memory_reuse"
    UNDER_UPDATE = "under_update"
    OVER_UPDATE = "over_update"
    CONFLICT_COLLAPSE = "conflict_collapse"
    SCOPE_LEAKAGE = "scope_leakage"
    POLICY_VIOLATION = "policy_violation"
    WRONG_SOURCE_ATTRIBUTION = "wrong_source_attribution"
    MEMORY_HALLUCINATION = "memory_hallucination"
    UNNECESSARY_MEMORY_WRITE = "unnecessary_memory_write"
    FAILURE_TO_FORGET = "failure_to_forget"
    FAILURE_TO_RELEASE_OR_RESTORE = "failure_to_release_or_restore"


class TaskProtocolV2(str, Enum):
    BLACK_BOX_TASK = "black_box_task"
    MEMORY_STATE_TASK = "memory_state_task"
    STRUCTURED_REVISION_TASK = "structured_revision_task"
    ORACLE_DIAGNOSTIC_TASK = "oracle_diagnostic_task"


# ProbeTypeV2 maps/extends v1 probe types to cover new aspects
class ProbeTypeV2(str, Enum):
    STATE_RESOLUTION = "state_resolution"
    PREMISE_RESISTANCE = "premise_resistance"
    POLICY_ADAPTATION = "policy_adaptation"
    AUDIT_LOCALIZATION = "audit_localization"
    TEMPORAL_ORDERING = "temporal_ordering"
    MULTI_AGENT_SYNCHRONY = "multi_agent_synchrony"


class EventSourceV2(str, Enum):
    USER = "user"
    AGENT = "agent"
    SYSTEM = "system"


class EventTypeV2(str, Enum):
    USER_MESSAGE = "user_message"
    AGENT_MESSAGE = "agent_message"
    MEMORY_READ = "memory_read"
    MEMORY_WRITE = "memory_write"
    TOOL_CALL = "tool_call"
    SYSTEM_NOTIFICATION = "system_notification"
    AUXILIARY = "auxiliary"


class MemoryLifecycleOperationTypeV2(str, Enum):
    CREATE = "CREATE"
    UPDATE = "UPDATE"
    SUPERSEDE = "SUPERSEDE"
    BLOCK = "BLOCK"
    UNBLOCK = "UNBLOCK"
    FORGET = "FORGET"
    REACTIVATE = "REACTIVATE"


class MemoryStatusV2(str, Enum):
    AUTHORIZED = "AUTHORIZED"
    BLOCKED = "BLOCKED"
    SUPERSEDED = "SUPERSEDED"
    UNRESOLVED = "UNRESOLVED"


class StructuredRevisionActionTypeV2(str, Enum):
    SUPERSEDES = "SUPERSEDES"
    BLOCKS = "BLOCKS"
    RELEASES = "RELEASES"
    REAFFIRMS = "REAFFIRMS"
    UNCERTAIN = "UNCERTAIN"
    NO_REVISION = "NO_REVISION"


class TrustLevelV2(str, Enum):
    TRUSTED = "trusted"
    UNTRUSTED = "untrusted"
    VERIFIED = "verified"


class SplitV2(str, Enum):
    TRAIN = "train"
    DEV = "dev"
    TEST = "test"
