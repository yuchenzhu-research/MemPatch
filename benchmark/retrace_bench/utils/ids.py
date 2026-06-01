import hashlib


def generate_scenario_id(domain: str, index: int, seed: int = 7) -> str:
    # Use deterministic hash of domain, index, seed to create a clean unique ID
    hasher = hashlib.md5(f"{domain}_{index}_{seed}".encode("utf-8"))
    return f"scen_{domain[:6]}_{hasher.hexdigest()[:8]}"


def generate_query_id(scenario_id: str, probe_type: str) -> str:
    hasher = hashlib.md5(f"{scenario_id}_{probe_type}".encode("utf-8"))
    return f"q_{hasher.hexdigest()[:8]}"
