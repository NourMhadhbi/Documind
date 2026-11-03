"""Écriture JSON atomique : un fichier de résultats n'est jamais laissé à moitié écrit,
même si le programme est interrompu pendant une évaluation de plusieurs dizaines de minutes."""

import json
import os
from pathlib import Path
from typing import Any


def write_json_atomic(path: str, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    tmp = target.with_suffix(target.suffix + ".tmp")
    with open(tmp, "w", encoding="utf-8") as f:
        json.dump(payload, f, ensure_ascii=False, indent=2)
    os.replace(tmp, target)


def read_json(path: str) -> Any:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)
