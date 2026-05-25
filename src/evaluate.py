from __future__ import annotations

import json

from src.paths import EVALUATION_REPORT_PATH


def main() -> None:
    with open(EVALUATION_REPORT_PATH, "r", encoding="utf-8") as handle:
        metrics = json.load(handle)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
