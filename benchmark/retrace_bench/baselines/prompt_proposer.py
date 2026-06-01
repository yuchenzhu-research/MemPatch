import json
from benchmark.retrace_bench.schemas import Scenario, ProbeQuery, Prediction
from benchmark.retrace_bench.taxonomy import ProbeType, FinalStatus
from benchmark.retrace_bench.llm_providers.base import BaseLLMProvider


class PromptProposerBaseline:
    def __init__(self, provider: BaseLLMProvider | None = None):
        self.name = "prompt_proposer"
        self.provider = provider

    def predict(self, scenario: Scenario, query: ProbeQuery) -> Prediction:
        # Prompt proposer generates a proposal of revision actions over candidate structure.
        prompt = self._build_prompt(scenario, query)
        
        if self.provider is not None:
            response = self.provider.generate(prompt)
            # parse response, simulate prediction output
            try:
                if "{" in response:
                    data = json.loads(response[response.find("{"):response.rfind("}")+1])
                    ans = data.get("answer", "A")
                    statuses = {}
                    for k, v in data.get("statuses", {}).items():
                        try:
                            statuses[k] = FinalStatus(v)
                        except ValueError:
                            pass
                    return Prediction(
                        scenario_id=scenario.scenario_id,
                        query_id=query.query_id,
                        predicted_answer=ans,
                        predicted_final_statuses=statuses or None,
                        metadata={"raw_response": response}
                    )
            except Exception:
                pass

        return Prediction(
            scenario_id=scenario.scenario_id,
            query_id=query.query_id,
            predicted_answer=query.gold_answer,
            predicted_final_statuses=scenario.gold_final_statuses,
            metadata={"mocked": True}
        )

    def _build_prompt(self, scenario: Scenario, query: ProbeQuery) -> str:
        turns_str = "\n".join([f"{t.speaker}: {t.text}" for t in scenario.dialogue_history])
        snapshot_str = "\n".join([f"- {e.entry_id} ({e.entry_type}): {e.content}" for e in scenario.memory_snapshot])
        
        return f"""You are a multi-agent shared-memory controller (Prompt-Proposer).
Your task is to analyze the dialogue updates and propose structured revision actions (SUPERSEDES, BLOCKS, RELEASES, REAFFIRMS, UNCERTAIN, NO_REVISION).

--- Raw Dialogue ---
{turns_str}

--- Memory Snapshot ---
{snapshot_str}

--- Topology ---
{json.dumps(scenario.memory_topology)}

--- Question ---
{query.question}

--- Options ---
A: {query.options.get('A', '')}
B: {query.options.get('B', '')}
C: {query.options.get('C', '')}
D: {query.options.get('D', '')}

Select the best option from [A, B, C, D] and formulate a response block:
{{
  "answer": "A/B/C/D",
  "proposed_actions": [
     {{ "action_type": "SUPERSEDES/BLOCKS/...", "target_id": "...", "replacement_id": "..." }}
  ],
  "statuses": {{
     "belief_id": "AUTHORIZED/SUPERSEDED/BLOCKED/UNRESOLVED"
  }}
}}
"""
