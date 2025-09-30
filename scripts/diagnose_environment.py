"""Environment diagnostics script.
Run: python scripts/diagnose_environment.py
It will:
 - Show Python executable, version, sys.path
 - Try importing critical packages and report versions / errors
 - Optionally (with --freeze) print a trimmed pip freeze list.
"""
from __future__ import annotations
import importlib
import json
import os
import sys
import argparse
from typing import Dict, Any

CRITICAL_PACKAGES = [
    "fastapi",
    "uvicorn",
    "pydantic",
    "spacy",
    "nltk",
    "langdetect",
    "transformers",
    "torch",
    "networkx",
]

OPTIONAL_PACKAGES = [
    "datasets",
    "scikit_learn",  # alias; real module is sklearn
    "sklearn",
    "matplotlib",
    "seaborn",
    "plotly",
]


def try_import(pkg: str) -> Dict[str, Any]:
    result: Dict[str, Any] = {"package": pkg, "ok": False, "version": None, "error": None}
    module_name = pkg
    if pkg == "scikit_learn":  # skip, use sklearn instead
        result["skipped"] = True
        return result
    try:
        module = importlib.import_module(module_name)
        result["ok"] = True
        result["version"] = getattr(module, "__version__", "<no __version__>")
    except Exception as e:  # broad for diagnostics
        result["error"] = f"{type(e).__name__}: {e}"
    return result


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--freeze", action="store_true", help="Include pip freeze output")
    parser.add_argument("--json", action="store_true", help="Emit JSON only")
    args = parser.parse_args()

    report: Dict[str, Any] = {}
    report["python_executable"] = sys.executable
    report["python_version"] = sys.version
    report["cwd"] = os.getcwd()

    # Basic virtual environment heuristic
    report["in_venv"] = (
        hasattr(sys, 'real_prefix') or
        (hasattr(sys, 'base_prefix') and sys.base_prefix != sys.prefix)
    )

    report["sys_path_head"] = sys.path[:8]

    report["critical_imports"] = [try_import(p) for p in CRITICAL_PACKAGES]
    report["optional_imports"] = [try_import(p) for p in OPTIONAL_PACKAGES]

    if args.freeze:
        try:
            import subprocess
            freeze = subprocess.check_output([sys.executable, "-m", "pip", "freeze"], text=True)
            # keep only top relevant ones
            wanted = {pkg.lower() for pkg in CRITICAL_PACKAGES + OPTIONAL_PACKAGES}
            filtered = []
            for line in freeze.splitlines():
                name = line.split("==")[0].lower()
                if name in wanted:
                    filtered.append(line)
            report["pip_freeze_filtered"] = filtered
        except Exception as e:
            report["pip_freeze_error"] = str(e)

    if args.json:
        print(json.dumps(report, indent=2))
        return

    # Human readable output
    print("=== Python Environment ===")
    print("Executable:", report["python_executable"])
    print("Version:", report["python_version"].splitlines()[0])
    print("Working Dir:", report["cwd"])
    print("In VirtualEnv:", report["in_venv"])
    print()
    print("=== sys.path (first entries) ===")
    for p in report["sys_path_head"]:
        print(" ", p)
    print()

    def show(section: str):
        print(f"=== {section.replace('_', ' ').title()} ===")
        for item in report[section]:
            if item.get("skipped"):
                print(f" - {item['package']}: skipped")
                continue
            if item["ok"]:
                print(f" - {item['package']}: OK (version {item['version']})")
            else:
                print(f" - {item['package']}: MISSING ({item['error']})")
        print()

    show("critical_imports")
    show("optional_imports")

    if args.freeze and "pip_freeze_filtered" in report:
        print("=== Filtered pip freeze (core/optional packages) ===")
        for line in report["pip_freeze_filtered"]:
            print(" ", line)

    print("Done.")


if __name__ == "__main__":  # pragma: no cover
    main()
