"""Wholesale distractor-leak re-check of a frozen suite file.

Run: python -m agentflight.tasks.leak_check data/agentflight_qa_v1.json
"""

from __future__ import annotations

import argparse
import json
import sys

from agentflight.tasks.leak_check import check_suite
from agentflight.tasks.schema import AgentFlightSuite


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("suite")
    args = ap.parse_args()
    with open(args.suite, encoding="utf-8") as f:
        suite = AgentFlightSuite.model_validate(json.load(f))
    report = check_suite(suite)
    print(json.dumps({k: v for k, v in report.items() if k != "failures"}, indent=2))
    for fail in report["failures"][:20]:
        print("FAIL", fail)
    sys.exit(0 if report["pass"] else 1)


if __name__ == "__main__":
    main()
