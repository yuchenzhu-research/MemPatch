# Prompt templates for judges and proposers

DIRECT_JUDGE_SYSTEM_PROMPT = """You are an expert evaluation judge. You are auditing a shared-memory revision authorization task.
Your job is to examine the history of dialogue updates and the current memory snapshot, and output the correct status and answers.
"""

PROPOSAL_SYSTEM_PROMPT = """You are a multi-agent shared-memory controller.
Your task is to analyze the dialogue updates and propose structured revision actions.
"""
