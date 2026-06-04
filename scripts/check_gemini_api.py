#!/usr/bin/env python3
"""Minimal Gemini API connectivity check."""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from scripts.gemini_api import call_gemini_generate, redact_secrets, require_api_key, resolve_model


def main() -> int:
    api_key = require_api_key()
    model = resolve_model()
    prompt = 'Return exactly this JSON: {"ok": true}'
    try:
        raw = call_gemini_generate(prompt=prompt, model=model, temperature=0.0)
        parsed = json.loads(raw)
        if parsed.get("ok") is True:
            print(json.dumps({"ok": True, "model": model}, ensure_ascii=False))
            return 0
        print(json.dumps({"ok": False, "error": "unexpected JSON payload", "model": model}, ensure_ascii=False))
        return 1
    except Exception as exc:
        print(
            json.dumps(
                {
                    "ok": False,
                    "model": model,
                    "error": redact_secrets(str(exc), api_key),
                },
                ensure_ascii=False,
            )
        )
        return 1


if __name__ == "__main__":
    raise SystemExit(main())
