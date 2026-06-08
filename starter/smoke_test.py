from __future__ import annotations

import json

import verifier


def main() -> int:
    result = verifier.verify(timeout=180, run=True)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["valid"]:
        print("PUBLIC_SMOKE_TEST_PASSED")
        return 0
    print("PUBLIC_SMOKE_TEST_FAILED")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
