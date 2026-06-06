from __future__ import annotations

import importlib.util
import inspect
import json
import math
from pathlib import Path
import subprocess
import sys
import traceback
from typing import Any

import numpy as np
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split


START = ">>>>> Start Structured Result"
END = ">>>>> End Structured Result"
TASKS = ("Binary", "Iris", "LiH")
REQUIRED_FUNCTIONS = (
    "build_search_space",
    "apply_noise_model",
    "run_na_qas",
    "select_best_architecture",
    "predict_with_architecture",
    "run_lih",
)


def _detail(name: str, status: str, score: float, max_score: float, message: str = "") -> dict[str, Any]:
    return {
        "name": name,
        "status": status,
        "score": round(float(score), 4),
        "max_score": float(max_score),
        "message": message,
    }


def _quality_detail(name: str, score: float, max_score: float, message: str = "") -> dict[str, Any]:
    status = "PASSED" if score >= max_score else "FAILED"
    return _detail(name, status, score, max_score, message)


def _emit(valid: bool, score: float, summary: str, details: list[dict[str, Any]]) -> None:
    payload = {
        "valid": bool(valid),
        "score": round(float(score), 4),
        "pass_rate": round(float(score) / 100.0, 6),
        "summary": summary,
        "details": details,
    }
    print(START)
    print(json.dumps(payload, ensure_ascii=False))
    print(END)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _is_int(value: Any) -> bool:
    return isinstance(value, int) and not isinstance(value, bool)


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def _run_main() -> tuple[bool, str]:
    if not Path("main.py").is_file():
        return False, "main.py not found"
    try:
        proc = subprocess.run(
            [sys.executable, "main.py"],
            timeout=360,
            check=False,
            text=True,
            capture_output=True,
        )
    except subprocess.TimeoutExpired:
        return False, "python main.py timed out after 360s"
    except Exception as exc:
        return False, f"python main.py failed to start: {exc}"
    if proc.returncode != 0:
        tail = (proc.stderr or proc.stdout or "")[-1200:]
        return False, f"python main.py exited with code {proc.returncode}: {tail}"
    return True, "python main.py completed"


def _load_json(path: str) -> tuple[Any | None, str | None]:
    try:
        return json.loads(Path(path).read_text(encoding="utf-8")), None
    except Exception as exc:
        return None, f"{path} is not valid JSON: {exc}"


def _schema_errors(data: Any) -> list[str]:
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
            if not _is_int(search.get(key)):
                errors.append(f"Search_space.{key} should be an integer")
        gates = search.get("gate_set")
        if not isinstance(gates, list) or set(gates) != {"Rx", "Ry", "Rz"}:
            errors.append("Search_space.gate_set should contain Rx, Ry, Rz")
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
            if not _is_number(hardware.get("alpha")):
                errors.append("Objectives.hardware_cost.alpha should be a number")
            if not _is_number(hardware.get("beta")):
                errors.append("Objectives.hardware_cost.beta should be a number")

    supernet = data.get("Supernet")
    if not isinstance(supernet, dict):
        errors.append("Supernet should be an object")
    else:
        if not _is_int(supernet.get("num_experts")):
            errors.append("Supernet.num_experts should be an integer")
        epsilon = supernet.get("epsilon")
        if not _is_number(epsilon) or not 0.0 <= float(epsilon) <= 1.0:
            errors.append("Supernet.epsilon should be a number in [0, 1]")

    for task in ("Binary", "Iris"):
        item = data.get(task)
        if not isinstance(item, dict):
            errors.append(f"{task} should be an object")
            continue
        for key in ("accuracy", "loss", "hardware_cost"):
            if not _is_number(item.get(key)):
                errors.append(f"{task}.{key} should be a number")
        if _is_number(item.get("accuracy")) and not 0.0 <= float(item["accuracy"]) <= 1.0:
            errors.append(f"{task}.accuracy should be in [0, 1]")
        for key in ("depth", "cnot", "architecture_index"):
            if not _is_int(item.get(key)):
                errors.append(f"{task}.{key} should be an integer")
        angles = item.get("rotation_angles")
        if not isinstance(angles, list) or not all(_is_number(v) for v in angles):
            errors.append(f"{task}.rotation_angles should be a number list")

    lih = data.get("LiH")
    if not isinstance(lih, dict):
        errors.append("LiH should be an object")
    else:
        for key in ("ground_state_energy", "reference_energy", "energy_error", "hardware_cost"):
            if not _is_number(lih.get(key)):
                errors.append(f"LiH.{key} should be a number")
        for key in ("depth", "cnot", "architecture_index"):
            if not _is_int(lih.get(key)):
                errors.append(f"LiH.{key} should be an integer")
        angles = lih.get("rotation_angles")
        if not isinstance(angles, list) or not all(_is_number(v) for v in angles):
            errors.append("LiH.rotation_angles should be a number list")

    fronts = data.get("Pareto_fronts")
    if not isinstance(fronts, dict):
        errors.append("Pareto_fronts should be an object")
    else:
        for task in TASKS:
            if not isinstance(fronts.get(task), list):
                errors.append(f"Pareto_fronts.{task} should be a list")
    return errors


def _import_main() -> tuple[Any | None, str | None]:
    spec = importlib.util.spec_from_file_location("qastask_main", "main.py")
    if spec is None or spec.loader is None:
        return None, "main.py could not be imported"
    module = importlib.util.module_from_spec(spec)
    try:
        spec.loader.exec_module(module)
    except Exception as exc:
        return None, f"main.py import failed: {type(exc).__name__}: {exc}"
    missing = [name for name in REQUIRED_FUNCTIONS if not callable(getattr(module, name, None))]
    if missing:
        return None, f"missing required callable interfaces: {', '.join(missing)}"
    return module, None


def _basic_gate() -> tuple[Any | None, dict[str, Any] | None, list[dict[str, Any]], str | None]:
    ok, run_msg = _run_main()
    if not ok:
        return None, None, [_detail("basic legality gate", "FAILED", 0, 0, run_msg)], run_msg

    artifact_errors: list[str] = []
    if not Path("best_circuit.txt").is_file():
        artifact_errors.append("best_circuit.txt was not generated")
    if not Path("best_circuit.svg").is_file():
        artifact_errors.append("best_circuit.svg was not generated")
    elif Path("best_circuit.svg").stat().st_size < 50:
        artifact_errors.append("best_circuit.svg is unexpectedly small")
    if artifact_errors:
        msg = "; ".join(artifact_errors)
        return None, None, [_detail("basic legality gate", "FAILED", 0, 0, msg)], msg

    data, json_error = _load_json("best_circuit.txt")
    if json_error:
        return None, None, [_detail("basic legality gate", "FAILED", 0, 0, json_error)], json_error

    schema_errors = _schema_errors(data)
    if schema_errors:
        msg = "best_circuit.txt schema invalid: " + "; ".join(schema_errors[:5])
        return None, None, [_detail("basic legality gate", "FAILED", 0, 0, msg)], msg

    module, import_error = _import_main()
    if import_error:
        return None, None, [_detail("basic legality gate", "FAILED", 0, 0, import_error)], import_error

    detail = _detail("basic legality gate", "PASSED", 0, 0, "main.py ran, artifacts exist, JSON schema is valid, core interfaces import")
    return module, data, [detail], None


def _get_layers(architecture: Any) -> list[Any]:
    if isinstance(architecture, dict):
        layers = architecture.get("layers") or architecture.get("Layers")
    else:
        layers = getattr(architecture, "layers", None)
    return list(layers) if isinstance(layers, (list, tuple)) else []


def _get_rotations(layer: Any) -> list[str]:
    if isinstance(layer, dict):
        rotations = layer.get("rotations") or layer.get("rotation_gates") or layer.get("gates")
    else:
        rotations = getattr(layer, "rotations", None)
    return [str(item) for item in rotations] if isinstance(rotations, (list, tuple)) else []


def _get_entanglers(layer: Any) -> list[tuple[int, int]]:
    if isinstance(layer, dict):
        entanglers = layer.get("entanglers") or layer.get("cnot_pairs") or layer.get("cnots") or []
    else:
        entanglers = getattr(layer, "entanglers", None) or []
    pairs: list[tuple[int, int]] = []
    for item in entanglers:
        try:
            control, target = item
            pairs.append((int(control), int(target)))
        except Exception:
            continue
    return pairs


def _search_space_quality(module: Any) -> tuple[float, str]:
    try:
        space = module.build_search_space(
            n_qubits=2,
            min_depth=1,
            max_depth=2,
            max_architectures=None,
            seed=20260605,
        )
        if not isinstance(space, list):
            space = list(space)
    except Exception as exc:
        return 0.0, f"build_search_space hidden call raised {type(exc).__name__}: {exc}"

    points = 0.0
    messages: list[str] = []
    count = len(space)
    if count == 1332:
        points += 6.0
    elif count >= 200:
        points += 3.0
    else:
        messages.append(f"small search space count={count}, expected 1332 for n_qubits=2 depth 1..2")

    layer_counts = {len(_get_layers(arch)) for arch in space[: min(len(space), 2000)]}
    if {1, 2}.issubset(layer_counts):
        points += 4.0
    else:
        messages.append(f"variable layer counts missing: {sorted(layer_counts)}")

    gates_seen: set[str] = set()
    entangler_sets: list[set[tuple[int, int]]] = []
    for arch in space:
        layers = _get_layers(arch)
        for layer in layers:
            gates_seen.update(_get_rotations(layer))
            entangler_sets.append(set(_get_entanglers(layer)))
    if {"Rx", "Ry", "Rz"}.issubset(gates_seen):
        points += 4.0
    else:
        messages.append(f"gate coverage incomplete: {sorted(gates_seen)}")

    if any(not pairs for pairs in entangler_sets):
        points += 3.0
    else:
        messages.append("empty CNOT subset not observed")
    if any({(0, 1), (1, 0)}.issubset(pairs) for pairs in entangler_sets):
        points += 3.0
    else:
        messages.append("full directed CNOT subset for 2 qubits not observed")

    return points / 10.0, "; ".join(messages) or f"count={count}, gates={sorted(gates_seen)}"


def _normalize_search_result(result: Any) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if isinstance(result, (tuple, list)) and len(result) >= 2:
        best = result[0]
        pareto = result[1]
    elif isinstance(result, dict):
        best = result.get("best") or result.get("best_architecture") or result
        pareto = result.get("pareto") or result.get("pareto_front") or result.get("front") or []
    else:
        best, pareto = {}, []
    if not isinstance(best, dict):
        best = {}
    if not isinstance(pareto, list):
        pareto = list(pareto) if isinstance(pareto, tuple) else []
    pareto = [item for item in pareto if isinstance(item, dict)]
    return best, pareto


def _call_run_na_qas(module: Any, *args: Any, **kwargs: Any) -> Any:
    func = module.run_na_qas
    try:
        sig = inspect.signature(func)
        accepted = {}
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        for key, value in kwargs.items():
            if has_var_kwargs or key in sig.parameters:
                accepted[key] = value
        return func(*args, **accepted)
    except TypeError:
        return func(*args)


def _call_predict_with_architecture(module: Any, architecture: dict[str, Any], *args: Any, **kwargs: Any) -> Any:
    func = getattr(module, "predict_with_architecture", None)
    if not callable(func):
        raise AttributeError("predict_with_architecture missing")
    try:
        sig = inspect.signature(func)
        accepted = {}
        has_var_kwargs = any(p.kind == inspect.Parameter.VAR_KEYWORD for p in sig.parameters.values())
        for key, value in kwargs.items():
            if has_var_kwargs or key in sig.parameters:
                accepted[key] = value
        return func(architecture, *args, **accepted)
    except TypeError:
        return func(architecture, *args)


def _hardware_multiplier(depth: int, cnot: int, depth_budget: int, cnot_budget: int) -> tuple[float, str]:
    if depth <= depth_budget and cnot <= cnot_budget:
        return 1.0, "within budget"
    depth_ratio = max(1.0, depth / max(1, depth_budget))
    cnot_ratio = max(1.0, cnot / max(1, cnot_budget))
    over = max(depth_ratio, cnot_ratio)
    if over <= 1.25:
        return 0.75, f"slightly over budget depth={depth}/{depth_budget}, cnot={cnot}/{cnot_budget}"
    if over <= 1.75:
        return 0.45, f"over budget depth={depth}/{depth_budget}, cnot={cnot}/{cnot_budget}"
    return 0.20, f"far over budget depth={depth}/{depth_budget}, cnot={cnot}/{cnot_budget}"


def _coerce_angles(value: Any) -> list[float]:
    if not isinstance(value, (list, tuple)):
        return []
    angles: list[float] = []
    for item in value:
        if _is_number(item):
            angles.append(float(item))
    return angles


def _architecture_depth_cnot(result: dict[str, Any]) -> tuple[int, int]:
    reported_depth = int(result.get("depth", 999)) if _is_number(result.get("depth")) else 999
    reported_cnot = int(result.get("cnot", 999)) if _is_number(result.get("cnot")) else 999
    layers = _get_layers(result)
    if not layers:
        return reported_depth, reported_cnot
    computed_depth = 0
    computed_cnot = 0
    for layer in layers:
        entanglers = _get_entanglers(layer)
        computed_depth += 1 + int(bool(entanglers))
        computed_cnot += len(entanglers)
    return max(reported_depth, computed_depth), max(reported_cnot, computed_cnot)


def _rx(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [[math.cos(half), -1j * math.sin(half)], [-1j * math.sin(half), math.cos(half)]],
        dtype=complex,
    )


def _ry(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [[math.cos(half), -math.sin(half)], [math.sin(half), math.cos(half)]],
        dtype=complex,
    )


def _rz(theta: float) -> np.ndarray:
    half = theta / 2.0
    return np.array(
        [[np.exp(-1j * half), 0.0], [0.0, np.exp(1j * half)]],
        dtype=complex,
    )


def _single_qubit_matrix(gate: str, theta: float) -> np.ndarray | None:
    norm = str(gate).strip().lower()
    if norm == "rx":
        return _rx(theta)
    if norm == "ry":
        return _ry(theta)
    if norm == "rz":
        return _rz(theta)
    return None


def _apply_single_qubit(state: np.ndarray, op: np.ndarray, qubit: int, n_qubits: int = 2) -> np.ndarray:
    if qubit < 0 or qubit >= n_qubits:
        return state
    full = np.array([[1.0 + 0.0j]])
    ident = np.eye(2, dtype=complex)
    for q in range(n_qubits):
        full = np.kron(full, op if q == qubit else ident)
    return full @ state


def _apply_cnot(state: np.ndarray, control: int, target: int, n_qubits: int = 2) -> np.ndarray:
    if control == target or control < 0 or target < 0 or control >= n_qubits or target >= n_qubits:
        return state
    result = np.zeros_like(state)
    for idx, amp in enumerate(state):
        control_bit = (idx >> (n_qubits - 1 - control)) & 1
        new_idx = idx
        if control_bit:
            new_idx = idx ^ (1 << (n_qubits - 1 - target))
        result[new_idx] += amp
    return result


def _state_from_laydown(result: dict[str, Any]) -> tuple[np.ndarray, int, int, int, int]:
    layers = _get_layers(result)
    angles = _coerce_angles(result.get("rotation_angles"))
    state = np.zeros(4, dtype=complex)
    state[0] = 1.0
    angle_idx = 0
    rotation_count = 0
    entangler_count = 0
    for layer in layers[:8]:
        rotations = _get_rotations(layer)
        for q, gate in enumerate(rotations[:2]):
            theta = angles[angle_idx] if angle_idx < len(angles) else 0.0
            angle_idx += 1
            op = _single_qubit_matrix(gate, theta)
            if op is not None:
                state = _apply_single_qubit(state, op, q, 2)
                rotation_count += 1
        for control, target in _get_entanglers(layer):
            state = _apply_cnot(state, control, target, 2)
            entangler_count += 1
    norm = float(np.linalg.norm(state))
    if not math.isfinite(norm) or norm <= 0.0:
        state = np.zeros(4, dtype=complex)
        state[0] = 1.0
    else:
        state = state / norm
    return state, len(layers), rotation_count, entangler_count, len(angles)


def _pauli_hamiltonian(dist: float, noisy: bool) -> np.ndarray:
    i = np.eye(2, dtype=complex)
    x = np.array([[0.0, 1.0], [1.0, 0.0]], dtype=complex)
    y = np.array([[0.0, -1j], [1j, 0.0]], dtype=complex)
    z = np.array([[1.0, 0.0], [0.0, -1.0]], dtype=complex)
    shift = float(dist) - 1.546
    coeffs = {
        "zi": -0.34 + 0.10 * shift,
        "iz": 0.27 - 0.06 * shift,
        "zz": -0.19 + 0.03 * math.cos(2.1 * float(dist)),
        "xx": 0.17 + 0.025 * math.sin(1.7 * float(dist)),
        "yy": 0.13 - 0.020 * math.cos(1.2 * float(dist)),
        "xz": -0.075 + 0.015 * shift,
        "zx": 0.055 + 0.010 * math.sin(2.5 * float(dist)),
    }
    if noisy:
        coeffs["xx"] *= 0.92
        coeffs["yy"] *= 0.90
        coeffs["zz"] *= 1.04
    constant = -7.78 - 0.09 * math.exp(-((float(dist) - 1.55) ** 2) / 0.12)
    hamiltonian = (
        constant * np.kron(i, i)
        + coeffs["zi"] * np.kron(z, i)
        + coeffs["iz"] * np.kron(i, z)
        + coeffs["zz"] * np.kron(z, z)
        + coeffs["xx"] * np.kron(x, x)
        + coeffs["yy"] * np.kron(y, y)
        + coeffs["xz"] * np.kron(x, z)
        + coeffs["zx"] * np.kron(z, x)
    )
    return np.asarray((hamiltonian + hamiltonian.conj().T) / 2.0, dtype=complex)


def _verified_lih_gap(result: dict[str, Any], dist: float, noisy: bool) -> tuple[float, float, float, int, int, str]:
    state, layer_count, rotation_count, entangler_count, angle_count = _state_from_laydown(result)
    depth, cnot = _architecture_depth_cnot(result)
    hamiltonian = _pauli_hamiltonian(dist, noisy)
    exact = float(np.linalg.eigvalsh(hamiltonian)[0].real)
    energy = float(np.vdot(state, hamiltonian @ state).real)
    gap = max(0.0, energy - exact)
    signature = (
        f"layers={layer_count}, rotations={rotation_count}, "
        f"entanglers={entangler_count}, angles={angle_count}"
    )
    return gap, energy, exact, depth, cnot, signature


def _hidden_classification_quality(module: Any) -> tuple[float, str]:
    try:
        rng = np.random.default_rng(20260605)
        x = rng.uniform(-1.6, 1.6, size=(240, 4))
        latent = (
            0.9 * np.sin(2.8 * x[:, 0] + 0.3 * x[:, 2])
            + 0.7 * np.cos(2.4 * x[:, 1] - 0.2 * x[:, 3])
            + 0.9 * x[:, 0] * x[:, 1]
            - 0.65 * x[:, 2] * x[:, 3]
            + 0.25 * np.sin(5.5 * x[:, 0] * x[:, 2])
        )
        y = (latent > np.quantile(latent, 0.52)).astype(int)
        x_train, x_holdout, y_train, y_holdout = train_test_split(
            x,
            y,
            test_size=0.40,
            random_state=904,
            stratify=y,
        )
        result = _call_run_na_qas(
            module,
            x_train,
            y_train,
            x_holdout,
            y_holdout,
            n_qubits=4,
            n_outputs=1,
            population_size=18,
            generations=5,
            min_depth=1,
            max_depth=4,
            noisy=True,
            seed=904,
            task_name="HiddenBinary",
        )
        best, pareto = _normalize_search_result(result)
        pred = _call_predict_with_architecture(
            module,
            best,
            x_train,
            y_train,
            x_holdout,
            n_outputs=1,
            noisy=True,
            seed=1904,
        )
        pred = np.asarray(pred).reshape(-1)
        if len(pred) != len(y_holdout):
            return 0.0, f"predict_with_architecture returned {len(pred)} predictions, expected {len(y_holdout)}"
        labels = np.unique(y_train)
        pred = np.asarray([labels[int(np.argmin(np.abs(labels - value)))] if _is_number(value) else labels[0] for value in pred])
        verified_acc = float(accuracy_score(y_holdout, pred))
        reported_acc = _safe_float(best.get("accuracy"), 0.0)
        loss = _safe_float(best.get("loss"), 999.0)
        cnot = int(best.get("cnot", 999)) if _is_number(best.get("cnot")) else 999
        depth = int(best.get("depth", 999)) if _is_number(best.get("depth")) else 999
        pareto_size = len(pareto)
    except Exception as exc:
        return 0.0, f"hidden run_na_qas raised {type(exc).__name__}: {exc}"

    quality = 0.0
    if verified_acc >= 0.86:
        quality += 28.0
    elif verified_acc >= 0.80:
        quality += 18.0
    elif verified_acc >= 0.73:
        quality += 9.0
    elif verified_acc >= 0.66:
        quality += 4.0
    if abs(reported_acc - verified_acc) <= 0.06:
        quality += 3.0
    if math.isfinite(loss) and loss < 0.65:
        quality += 1.0
    if pareto_size >= 3:
        quality += 1.0
    multiplier, hw_msg = _hardware_multiplier(depth, cnot, depth_budget=7, cnot_budget=10)
    points = max(0.0, min(35.0, quality)) * multiplier
    if verified_acc >= 0.82 and depth <= 6 and cnot <= 8:
        points += 10.0
    return points, (
        f"verified_acc={verified_acc:.3f}, reported_acc={reported_acc:.3f}, "
        f"loss={loss:.3f}, pareto={pareto_size}, depth={depth}, cnot={cnot}, {hw_msg}"
    )


def _lih_quality(module: Any, data: dict[str, Any]) -> tuple[float, str]:
    if callable(getattr(module, "run_lih", None)):
        results: list[dict[str, Any]] = []
        try:
            sig = inspect.signature(module.run_lih)
            for offset, dist in enumerate((1.24, 1.546, 1.92)):
                kwargs: dict[str, Any] = {}
                if "noisy" in sig.parameters:
                    kwargs["noisy"] = bool(offset % 2)
                if "seed" in sig.parameters:
                    kwargs["seed"] = 1700 + offset
                if "dist" in sig.parameters:
                    kwargs["dist"] = dist
                value = module.run_lih(**kwargs)
                if not isinstance(value, dict):
                    return 0.0, f"run_lih(dist={dist}) did not return an object"
                results.append(value)
        except Exception as exc:
            return 0.0, f"run_lih hidden call raised {type(exc).__name__}: {exc}"
    else:
        value = data.get("LiH", {})
        if not isinstance(value, dict):
            return 0.0, "LiH result is not an object"
        results = [value]

    verified_gaps: list[float] = []
    self_errors: list[float] = []
    depths: list[int] = []
    cnots: list[int] = []
    energies: list[float] = []
    signatures: list[str] = []
    for idx, result in enumerate(results):
        dist = (1.24, 1.546, 1.92)[idx] if len(results) > 1 else 1.546
        noisy = bool(idx % 2) if len(results) > 1 else bool(result.get("noisy", True))
        gap, verified_energy, exact_energy, depth, cnot, signature = _verified_lih_gap(result, dist, noisy)
        self_error = _safe_float(
            result.get("energy_error"),
            abs(_safe_float(result.get("ground_state_energy"), 999.0) - _safe_float(result.get("reference_energy"), 999.0)),
        )
        verified_gaps.append(gap)
        self_errors.append(abs(self_error))
        energies.append(verified_energy)
        depths.append(depth)
        cnots.append(cnot)
        signatures.append(signature)

    mean_gap = float(np.mean(verified_gaps))
    max_gap = float(np.max(verified_gaps))
    mean_self_error = float(np.mean(self_errors))
    max_depth = int(max(depths))
    max_cnot = int(max(cnots))
    curve_ok = len(energies) < 3 or (energies[1] <= energies[0] + 0.06 and energies[1] <= energies[2] + 0.06)

    quality = 0.0
    if mean_gap < 0.010:
        quality += 18.0
    elif mean_gap < 0.030:
        quality += 11.0
    elif mean_gap < 0.070:
        quality += 5.0
    elif mean_gap < 0.140:
        quality += 2.0
    if max_gap < 0.025:
        quality += 8.0
    elif max_gap < 0.060:
        quality += 4.0
    elif max_gap < 0.120:
        quality += 1.0
    if curve_ok and mean_gap < 0.060:
        quality += 4.0
    if 1 <= max_cnot <= 6 and max_depth <= 8:
        quality += 4.0
    elif max_cnot == 0:
        quality -= 3.0
    if mean_self_error < 0.08:
        quality += 1.0
    multiplier, hw_msg = _hardware_multiplier(max_depth, max_cnot, depth_budget=8, cnot_budget=8)
    points = max(0.0, min(35.0, quality)) * multiplier
    if mean_gap < 0.018 and max_depth <= 6 and 1 <= max_cnot <= 6:
        points += 12.0
    elif mean_gap < 0.035 and max_depth <= 8 and 1 <= max_cnot <= 8:
        points += 5.0
    return points, (
        f"verified_gap_mean={mean_gap:.6f}, verified_gap_max={max_gap:.6f}, "
        f"self_error_mean={mean_self_error:.6f}, verified_energies={[round(v, 5) for v in energies]}, "
        f"max_depth={max_depth}, max_cnot={max_cnot}, {hw_msg}, signatures={signatures[:2]}"
    )


def _report_quality(data: dict[str, Any]) -> tuple[float, str]:
    binary = data.get("Binary", {})
    iris = data.get("Iris", {})
    lih = data.get("LiH", {})
    fronts = data.get("Pareto_fronts", {})

    acc_b = _safe_float(binary.get("accuracy"), 0.0) if isinstance(binary, dict) else 0.0
    acc_i = _safe_float(iris.get("accuracy"), 0.0) if isinstance(iris, dict) else 0.0
    err_l = _safe_float(lih.get("energy_error"), 999.0) if isinstance(lih, dict) else 999.0
    front_b = fronts.get("Binary", []) if isinstance(fronts, dict) else []
    front_i = fronts.get("Iris", []) if isinstance(fronts, dict) else []
    front_l = fronts.get("LiH", []) if isinstance(fronts, dict) else []

    points = 0.0
    avg_acc = (acc_b + acc_i) / 2.0
    if avg_acc >= 0.92:
        points += 0.5
    elif avg_acc >= 0.80:
        points += 0.5
    if err_l < 0.02:
        points += 0.5
    elif err_l < 0.05:
        points += 0.25
    if len(front_b) >= 3 and len(front_i) >= 3:
        points += 0.5
    if _safe_float(binary.get("depth", 999), 999) <= 8 and _safe_float(binary.get("cnot", 999), 999) <= 12:
        points += 0.25
    if _safe_float(iris.get("depth", 999), 999) <= 10 and _safe_float(iris.get("cnot", 999), 999) <= 24:
        points += 0.25
    if _safe_float(lih.get("depth", 999), 999) <= 12 and _safe_float(lih.get("cnot", 999), 999) <= 24:
        points += 0.25
    if int(data.get("Trade_off_solutions", 0)) >= len(front_b) + len(front_i) + len(front_l) >= 5:
        points += 0.5

    return min(points, 2.0), f"binary_acc={acc_b:.3f}, iris_acc={acc_i:.3f}, lih_error={err_l:.6f}, fronts=({len(front_b)}, {len(front_i)}, {len(front_l)})"


def evaluate() -> None:
    module, data, details, gate_error = _basic_gate()
    if gate_error is not None:
        _emit(False, 0.0, gate_error, details)
        return
    assert module is not None and data is not None

    total = 0.0
    search_points, search_msg = _search_space_quality(module)
    details.append(_quality_detail("search-space interface quality", search_points, 2, search_msg))
    total += search_points

    cls_points, cls_msg = _hidden_classification_quality(module)
    details.append(_quality_detail("hidden noisy classification quality", cls_points, 45, cls_msg))
    total += cls_points

    lih_points, lih_msg = _lih_quality(module, data)
    details.append(_quality_detail("LiH energy and hardware tradeoff", lih_points, 51, lih_msg))
    total += lih_points

    report_points, report_msg = _report_quality(data)
    details.append(_quality_detail("reported Pareto and hardware quality", report_points, 2, report_msg))
    total += report_points

    _emit(True, total, f"Quality score: {total:.1f}/100 after legality gate passed", details)


if __name__ == "__main__":
    try:
        evaluate()
    except Exception:
        _emit(
            False,
            0.0,
            "private judge internal error",
            [_detail("private judge internal error", "ERROR", 0, 0, traceback.format_exc()[-1600:])],
        )
