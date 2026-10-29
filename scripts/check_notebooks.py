"""Validate notebook structure and code, or clear outputs before a commit.

This uses only the Python standard library and does not execute training cells.
The supported line magics configure autoreload and Matplotlib in these labs.
"""

import argparse
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
MAGICS = {"%load_ext autoreload", "%autoreload 2", "%matplotlib inline"}


def check_notebook(path, clear_outputs=False):
    """Check a notebook's cell structure, clean state, and Python syntax."""
    notebook = json.loads(path.read_text(encoding="utf-8"))
    if notebook.get("nbformat") != 4:
        raise ValueError(f"{path}: expected notebook format 4")
    if not isinstance(notebook.get("metadata"), dict):
        raise ValueError(f"{path}: missing notebook metadata")
    cells = notebook.get("cells")
    if not isinstance(cells, list) or not cells:
        raise ValueError(f"{path}: expected a nonempty cell list")
    ids = set()
    for index, cell in enumerate(cells):
        location = f"{path.name}:cell {index}"
        if not cell.get("id") or cell["id"] in ids:
            raise ValueError(f"{location}: missing or duplicate cell ID")
        ids.add(cell["id"])
        if not isinstance(cell.get("metadata"), dict):
            raise ValueError(f"{location}: missing cell metadata")
        source = cell.get("source")
        if not isinstance(source, list) or not all(
            isinstance(line, str) for line in source
        ):
            raise ValueError(f"{location}: source must be a list of strings")
        if cell["cell_type"] == "code":
            if clear_outputs:
                cell["outputs"] = []
                cell["execution_count"] = None
                cell["metadata"].pop("execution", None)
            if (
                cell.get("outputs") != []
                or cell.get("execution_count", "missing") is not None
            ):
                raise ValueError(
                    f"{location}: clear outputs and execution counts before committing"
                )
            python_source = "\n".join(
                "" if line.strip() in MAGICS else line
                for line in "".join(source).splitlines()
            )
            compile(python_source, location, "exec")
        elif cell["cell_type"] not in ("markdown", "raw"):
            raise ValueError(f"{location}: unknown cell type")
    if clear_outputs:
        notebook["metadata"].pop("widgets", None)
        path.write_text(
            json.dumps(notebook, indent=1, ensure_ascii=False) + "\n", encoding="utf-8"
        )


def main():
    """Validate both lab notebooks and optionally remove generated outputs."""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--clear-outputs", action="store_true")
    args = parser.parse_args()
    paths = sorted((ROOT / "notebooks").glob("*.ipynb"))
    if not paths:
        raise SystemExit("No notebooks found")
    for path in paths:
        check_notebook(path, clear_outputs=args.clear_outputs)
        print(f"OK: {path.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
