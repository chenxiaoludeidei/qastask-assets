from __future__ import annotations

import argparse
import importlib.util
import json
import subprocess
import sys
from pathlib import Path
from typing import Any


TASKS = ("Binary", "Iris", "LiH")
REQUIRED_FUNCTIONS = (
    "build_search_space",
    "apply_noise_model",
    "run_na_qas",
    "select_best_architecture",
    "predict_with_architecture",
    "run_lih",
)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _check_number(errors: list[str], data: dict[str, Any], path: str, *, lo: float | None = None, hi: float | None = None) -> None:
    value = data.get(path.split(".")[-1])
    if not _is_number(value):
        errors.append(f"{path} should be a number")
        return
    if lo is not None and float(value) < lo:
        errors.append(f"{path} should be >= {lo}")
    if hi is not None and float(value) > hi:
        errors.append(f"{path} should be <= {hi}")


def _check_int(errors: list[str], data: dict[str, Any], key: str, path: str) -> None:
    if not _is_int(data.get(key)):
        errors.append(f"{path} should be an integer")


def _check_angles(errors: list[str], value: Any, path: str) -> None:
    if not isinstance(value, list):
        errors.append(f"{path} should be a list")
        return
    if not all(_is_number(item) for item in value):
        errors.append(f"{path} should contain only numbers")


def validate_schema(data: Any) -> list[str]:
    errors: list[str] = []
    if not isinstance(data, dict):
        return ["best_circuit.txt should contain a JSON object"]

    required_top = {
        "schema_version",
        "best_task",
        "Trade_off_solutions",
        "Search_space",
        "Objectives",
        "Supernet",
        "Binary",
        "Iris",
        "LiH",
        "Pareto_fronts",
    }
    missing = sorted(required_top - set(data))
    if missing:
        errors.append(f"missing top-level fields: {', '.join(missing)}")

    if data.get("best_task") not in TASKS:
        errors.append("best_task should be one of Binary, Iris, LiH")
    if not _is_int(data.get("Trade_off_solutions")):
        errors.append("Trade_off_solutions should be an integer")

    search = data.get("Search_space")
    if not isinstance(search, dict):
        errors.append("Search_space should be an object")
    else:
        for key in ("n_qubits", "min_depth", "max_depth"):
            _check_int(errors, search, key, f"Search_space.{key}")
        gates = search.get("gate_set")
        if not isinstance(gates, list) or set(gates) != {"Rx", "Ry", "Rz"}:
            errors.append("Search_space.gate_set should be exactly ['Rx', 'Ry', 'Rz'] in any order")
        if not isinstance(search.get("allow_directed_cnot"), bool):
            errors.append("Search_space.allow_directed_cnot should be a boolean")

    objectives = data.get("Objectives")
    if not isinstance(objectives, dict):
        errors.append("Objectives should be an object")
    else:
        hardware = objectives.get("hardware_cost")
        if objectives.get("loss") != "minimize":
            errors.append("Objectives.loss should be 'minimize'")
        if not isinstance(hardware, dict):
            errors.append("Objectives.hardware_cost should be an object")
        else:
            _check_number(errors, hardware, "Objectives.hardware_cost.alpha")
            _check_number(errors, hardware, "Objectives.hardware_cost.beta")

    supernet = data.get("Supernet")
    if not isinstance(supernet, dict):
        errors.append("Supernet should be an object")
    else:
        _check_int(errors, supernet, "num_experts", "Supernet.num_experts")
        _check_number(errors, supernet, "Supernet.epsilon", lo=0.0, hi=1.0)

    for task in ("Binary", "Iris"):
        item = data.get(task)
        if not isinstance(item, dict):
            errors.append(f"{task} should be an object")
            continue
        _check_number(errors, item, f"{task}.accuracy", lo=0.0, hi=1.0)
        _check_number(errors, item, f"{task}.loss", lo=0.0)
        _check_number(errors, item, f"{task}.hardware_cost", lo=0.0)
        _check_int(errors, item, "depth", f"{task}.depth")
        _check_int(errors, item, "cnot", f"{task}.cnot")
        _check_int(errors, item, "architecture_index", f"{task}.architecture_index")
        _check_angles(errors, item.get("rotation_angles"), f"{task}.rotation_angles")

    lih = data.get("LiH")
    if not isinstance(lih, dict):
        errors.append("LiH should be an object")
    else:
        _check_number(errors, lih, "LiH.ground_state_energy")
        _check_number(errors, lih, "LiH.reference_energy")
        _check_number(errors, lih, "LiH.energy_error", lo=0.0)
        _check_number(errors, lih, "LiH.hardware_cost", lo=0.0)
        _check_int(errors, lih, "depth", "LiH.depth")
        _check_int(errors, lih, "cnot", "LiH.cnot")
        _check_int(errors, lih, "architecture_index", "LiH.architecture_index")
        _check_angles(errors, lih.get("rotation_angles"), "LiH.rotation_angles")

    fronts = data.get("Pareto_fronts")
    if not isinstance(fronts, dict):
        errors.append("Pareto_fronts should be an object")
    else:
        for task in TASKS:
            if not isinstance(fronts.get(task), list):
                errors.append(f"Pareto_fronts.{task} should be a list")

    return errors


def run_main(timeout: int) -> tuple[bool, str]:
    try:
        proc = subprocess.run(
            [sys.executable, "main.py"],
            timeout=timeout,
            check=False,
            text=True,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        return False, f"python main.py timed out after {timeout}s"
    except Exception as exc:
        return False, f"python main.py failed to start: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1200:]
        return False, f"python main.py exited with code {proc.returncode}: {tail}"
    return True, "python main.py completed"


def load_result() -> tuple[Any | None, str | None]:
    path = Path("best_circuit.txt")
    if not path.is_file():
        return None, "best_circuit.txt was not generated"
    try:
        return json.loads(path.read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"best_circuit.txt is not valid JSON: {exc}"


def check_artifacts() -> list[str]:
    errors: list[str] = []
    if not Path("best_circuit.txt").is_file():
        errors.append("best_circuit.txt was not generated")
    if not Path("best_circuit.svg").is_file():
        errors.append("best_circuit.svg was not generated")
    elif Path("best_circuit.svg").stat().st_size < 50:
        errors.append("best_circuit.svg is unexpectedly small")
    return errors


def check_imports() -> list[str]:
    errors: list[str] = []
    spec = importlib.util.spec_from_file_location("qastask_main", "main.py")
    if spec is None or spec.loader is None:
        return ["main.py could not be imported"]
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return [f"main.py import failed: {type(exc).__name__}: {exc}"]
    for name in REQUIRED_FUNCTIONS:
        if not callable(getattr(module, name, None)):
            errors.append(f"{name} should be a callable function in main.py")
    return errors


def verify(timeout: int = 180, run: bool = True) -> dict[str, Any]:
    errors: list[str] = []
    messages: list[str] = []

    if run:
        ok, message = run_main(timeout)
        messages.append(message)
        if not ok:
            errors.append(message)
            return {"valid": False, "errors": errors, "messages": messages}

    errors.extend(check_artifacts())
    data, json_error = load_result()
    if json_error:
        errors.append(json_error)
    else:
        errors.extend(validate_schema(data))

    errors.extend(check_imports())
    return {"valid": not errors, "errors": errors, "messages": messages}


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--timeout", type=int, default=180)
    parser.add_argument("--no-run", action="store_true")
    args = parser.parse_args()

    result = verify(timeout=args.timeout, run=not args.no_run)
    print(json.dumps(result, ensure_ascii=False, indent=2))
    return 0 if result["valid"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
