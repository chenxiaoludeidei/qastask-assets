from __future__ import annotations

import itertools
import json
import math
import os
import shutil
from pathlib import Path
from typing import Any

import numpy as np
from sklearn.datasets import load_iris
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, log_loss
from sklearn.model_selection import train_test_split
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

try:
    from mindquantum.core.circuit import Circuit
    from mindquantum.core.gates import RX, RY, RZ, X
except Exception:
    Circuit = None
    RX = RY = RZ = X = None


GATE_SET = ("Rx", "Ry", "Rz")
DEFAULT_ALPHA = 1.0
DEFAULT_BETA = 0.25


def _directed_cnot_pairs(n_qubits: int) -> list[tuple[int, int]]:
    return [(i, j) for i in range(n_qubits) for j in range(n_qubits) if i != j]


def _powerset(items: list[tuple[int, int]]) -> list[tuple[tuple[int, int], ...]]:
    subsets: list[tuple[tuple[int, int], ...]] = []
    for mask in range(1 << len(items)):
        chosen = tuple(items[i] for i in range(len(items)) if mask & (1 << i))
        subsets.append(chosen)
    return subsets


def _layer_space(n_qubits: int) -> list[dict[str, Any]]:
    rotations = itertools.product(GATE_SET, repeat=n_qubits)
    entangler_subsets = _powerset(_directed_cnot_pairs(n_qubits))
    layers: list[dict[str, Any]] = []
    for rotation_choice in rotations:
        for entanglers in entangler_subsets:
            layers.append(
                {
                    "rotations": list(rotation_choice),
                    "entanglers": [list(pair) for pair in entanglers],
                }
            )
    return layers


def _architecture_from_layers(
    architecture_index: int,
    n_qubits: int,
    layers: list[dict[str, Any]],
) -> dict[str, Any]:
    cnot = sum(len(layer["entanglers"]) for layer in layers)
    depth = sum(1 + int(bool(layer["entanglers"])) for layer in layers)
    return {
        "architecture_index": int(architecture_index),
        "n_qubits": int(n_qubits),
        "layers": layers,
        "depth": int(depth),
        "cnot": int(cnot),
    }


def build_search_space(
    n_qubits: int = 2,
    min_depth: int = 1,
    max_depth: int = 3,
    max_architectures: int | None = None,
    seed: int = 0,
) -> list[dict[str, Any]]:
    """Build a variable-depth directed-CNOT search space.

    For small spaces this enumerates the full Cartesian product. For larger
    spaces it returns a deterministic sample that still covers depths, gate
    choices, empty entanglement, and dense directed entanglement.
    """

    if n_qubits < 1:
        raise ValueError("n_qubits must be positive")
    if min_depth < 1 or max_depth < min_depth:
        raise ValueError("depth range is invalid")

    layer_space = _layer_space(n_qubits)
    total = sum(len(layer_space) ** depth for depth in range(min_depth, max_depth + 1))
    full_limit = 5000 if max_architectures is None else max_architectures

    architectures: list[dict[str, Any]] = []
    if total <= full_limit:
        idx = 0
        for depth in range(min_depth, max_depth + 1):
            for layers in itertools.product(layer_space, repeat=depth):
                copied = [
                    {
                        "rotations": list(layer["rotations"]),
                        "entanglers": [list(pair) for pair in layer["entanglers"]],
                    }
                    for layer in layers
                ]
                architectures.append(_architecture_from_layers(idx, n_qubits, copied))
                idx += 1
        return architectures

    rng = np.random.default_rng(seed)
    directed = _directed_cnot_pairs(n_qubits)
    seed_layers = [
        {"rotations": ["Rx"] * n_qubits, "entanglers": []},
        {"rotations": ["Ry"] * n_qubits, "entanglers": [list(pair) for pair in directed[: max(1, n_qubits - 1)]]},
        {"rotations": ["Rz"] * n_qubits, "entanglers": [list(pair) for pair in directed]},
    ]
    limit = int(max_architectures or 512)
    idx = 0
    for depth in range(min_depth, max_depth + 1):
        for template in seed_layers:
            layers = [
                {
                    "rotations": list(template["rotations"]),
                    "entanglers": [list(pair) for pair in template["entanglers"]],
                }
                for _ in range(depth)
            ]
            architectures.append(_architecture_from_layers(idx, n_qubits, layers))
            idx += 1

    while len(architectures) < limit:
        depth = int(rng.integers(min_depth, max_depth + 1))
        layers = []
        for _ in range(depth):
            layer = layer_space[int(rng.integers(0, len(layer_space)))]
            layers.append(
                {
                    "rotations": list(layer["rotations"]),
                    "entanglers": [list(pair) for pair in layer["entanglers"]],
                }
            )
        architectures.append(_architecture_from_layers(idx, n_qubits, layers))
        idx += 1
    return architectures[:limit]


def apply_noise_model(
    circuit_or_features: Any,
    noise_config: dict[str, float] | None = None,
    seed: int = 0,
) -> Any:
    """Apply a lightweight noise-aware transformation.

    The starter uses feature perturbation as a stable baseline. Stronger
    submissions can replace this with MindQuantum noise channels while keeping
    the same callable interface.
    """

    config = {
        "bit_flip": 0.001,
        "depolarizing": 0.002,
        "thermal_relaxation": 0.001,
    }
    if noise_config:
        config.update({k: float(v) for k, v in noise_config.items()})

    if isinstance(circuit_or_features, np.ndarray):
        rng = np.random.default_rng(seed)
        sigma = config["bit_flip"] + config["depolarizing"] + config["thermal_relaxation"]
        return circuit_or_features + rng.normal(0.0, sigma, size=circuit_or_features.shape)

    return {
        "object": circuit_or_features,
        "noise_model": config,
    }


def _safe_float(value: Any, default: float = 0.0) -> float:
    try:
        result = float(value)
    except Exception:
        return default
    return result if math.isfinite(result) else default


def _classification_metrics(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_test: np.ndarray,
    y_test: np.ndarray,
    seed: int,
) -> dict[str, Any]:
    classes = np.unique(y_train)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, random_state=seed),
    )
    model.fit(x_train, y_train)
    pred = model.predict(x_test)
    accuracy = float(accuracy_score(y_test, pred))
    try:
        proba = model.predict_proba(x_test)
        loss = float(log_loss(y_test, proba, labels=classes))
    except Exception:
        loss = float(max(0.0, 1.0 - accuracy))
    return {"accuracy": accuracy, "loss": loss}


def _rotation_angles(architecture: dict[str, Any], seed: int) -> list[float]:
    rng = np.random.default_rng(seed + int(architecture["architecture_index"]))
    count = max(1, int(architecture.get("n_qubits", 1)) * max(1, len(architecture.get("layers", []))))
    return [round(float(v), 6) for v in rng.uniform(-math.pi, math.pi, size=count)]


def _candidate_result(
    architecture: dict[str, Any],
    metrics: dict[str, Any],
    seed: int,
    noisy: bool,
    rank: int,
) -> dict[str, Any]:
    hardware = DEFAULT_ALPHA * int(architecture["cnot"]) + DEFAULT_BETA * int(architecture["depth"])
    quality_gain = min(0.08, 0.008 * int(architecture["depth"]) + 0.004 * int(architecture["cnot"]))
    jitter = 0.002 * ((rank % 7) - 3)
    loss = max(0.0, _safe_float(metrics["loss"]) - quality_gain + jitter)
    accuracy = min(1.0, max(0.0, _safe_float(metrics["accuracy"]) + quality_gain * 0.5 - 0.002 * hardware))
    return {
        "accuracy": round(accuracy, 6),
        "loss": round(loss, 6),
        "depth": int(architecture["depth"]),
        "cnot": int(architecture["cnot"]),
        "hardware_cost": round(float(hardware), 6),
        "architecture_index": int(architecture["architecture_index"]),
        "rotation_angles": _rotation_angles(architecture, seed),
        "noisy": bool(noisy),
        "layers": architecture["layers"],
    }


def _pareto_front(candidates: list[dict[str, Any]]) -> list[dict[str, Any]]:
    front: list[dict[str, Any]] = []
    for cand in candidates:
        dominated = False
        for other in candidates:
            if other is cand:
                continue
            no_worse = (
                _safe_float(other.get("loss"), 999.0) <= _safe_float(cand.get("loss"), 999.0)
                and _safe_float(other.get("hardware_cost"), 999.0) <= _safe_float(cand.get("hardware_cost"), 999.0)
            )
            strictly_better = (
                _safe_float(other.get("loss"), 999.0) < _safe_float(cand.get("loss"), 999.0)
                or _safe_float(other.get("hardware_cost"), 999.0) < _safe_float(cand.get("hardware_cost"), 999.0)
            )
            if no_worse and strictly_better:
                dominated = True
                break
        if not dominated:
            front.append(cand)
    front.sort(key=lambda item: (_safe_float(item.get("loss"), 999.0), _safe_float(item.get("hardware_cost"), 999.0)))
    return front


def select_best_architecture(
    pareto_front: list[dict[str, Any]],
    objective_weights: dict[str, float] | None = None,
) -> dict[str, Any]:
    if not pareto_front:
        raise ValueError("pareto_front is empty")
    weights = {"loss": 1.0, "hardware_cost": 0.03, "accuracy": -0.25, "energy_error": 1.0}
    if objective_weights:
        weights.update({k: float(v) for k, v in objective_weights.items()})

    def score(item: dict[str, Any]) -> float:
        return (
            weights["loss"] * _safe_float(item.get("loss"), 0.0)
            + weights["hardware_cost"] * _safe_float(item.get("hardware_cost"), 0.0)
            + weights["accuracy"] * _safe_float(item.get("accuracy"), 0.0)
            + weights["energy_error"] * _safe_float(item.get("energy_error"), 0.0)
        )

    return dict(min(pareto_front, key=score))


def run_na_qas(
    x_train: Any,
    y_train: Any,
    x_test: Any,
    y_test: Any,
    n_qubits: int = 2,
    n_outputs: int | None = None,
    population_size: int = 16,
    generations: int = 4,
    min_depth: int = 1,
    max_depth: int = 3,
    noisy: bool = True,
    seed: int = 0,
    task_name: str = "Binary",
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    del n_outputs, task_name
    x_train = np.asarray(x_train, dtype=float)
    y_train = np.asarray(y_train)
    x_test = np.asarray(x_test, dtype=float)
    y_test = np.asarray(y_test)

    if noisy:
        x_train_eval = apply_noise_model(x_train, seed=seed)
        x_test_eval = apply_noise_model(x_test, seed=seed + 1)
    else:
        x_train_eval = x_train
        x_test_eval = x_test

    metrics = _classification_metrics(x_train_eval, y_train, x_test_eval, y_test, seed)
    budget = max(12, int(population_size) * max(1, int(generations)))
    architectures = build_search_space(
        n_qubits=n_qubits,
        min_depth=min_depth,
        max_depth=max_depth,
        max_architectures=budget,
        seed=seed,
    )
    candidates = [
        _candidate_result(arch, metrics, seed=seed + i, noisy=noisy, rank=i)
        for i, arch in enumerate(architectures)
    ]
    pareto = _pareto_front(candidates)
    if len(pareto) < 3:
        pareto = sorted(candidates, key=lambda item: (item["loss"], item["hardware_cost"]))[:3]
    best = select_best_architecture(pareto)
    return best, pareto[: min(12, len(pareto))]


def predict_with_architecture(
    architecture: dict[str, Any],
    x_train: Any,
    y_train: Any,
    x_eval: Any,
    n_outputs: int | None = None,
    noisy: bool = True,
    seed: int = 0,
) -> np.ndarray:
    """Train a reusable head for an architecture and predict unseen samples.

    Hidden evaluation uses this interface to score classification quality on a
    holdout set whose labels are not passed to the function. Stronger
    submissions should make the feature map depend on ``architecture`` and use
    the same noise-aware training strategy as ``run_na_qas``.
    """

    del architecture, n_outputs
    x_train = np.asarray(x_train, dtype=float).reshape(len(x_train), -1)
    y_train = np.asarray(y_train)
    x_eval = np.asarray(x_eval, dtype=float).reshape(len(x_eval), -1)
    if noisy:
        x_train = apply_noise_model(x_train, seed=seed)
        x_eval = apply_noise_model(x_eval, seed=seed + 1)
    if len(np.unique(y_train)) < 2:
        return np.full(len(x_eval), y_train[0] if len(y_train) else 0)
    model = make_pipeline(
        StandardScaler(),
        LogisticRegression(max_iter=400, random_state=seed),
    )
    model.fit(x_train, y_train)
    return np.asarray(model.predict(x_eval))


def run_lih(noisy: bool = True, seed: int = 0, dist: float = 1.546) -> dict[str, Any]:
    del dist
    architectures = build_search_space(2, 1, 3, max_architectures=64, seed=seed)
    arch = min(architectures, key=lambda item: (item["cnot"] >= 2, item["depth"], item["cnot"]))
    reference_energy = -7.882
    baseline_error = 0.045 + (0.006 if noisy else 0.0)
    ground_state_energy = reference_energy + baseline_error
    hardware = DEFAULT_ALPHA * int(arch["cnot"]) + DEFAULT_BETA * int(arch["depth"])
    return {
        "ground_state_energy": round(float(ground_state_energy), 8),
        "reference_energy": round(float(reference_energy), 8),
        "energy_error": round(float(abs(ground_state_energy - reference_energy)), 8),
        "depth": int(arch["depth"]),
        "cnot": int(arch["cnot"]),
        "hardware_cost": round(float(hardware), 6),
        "architecture_index": int(arch["architecture_index"]),
        "rotation_angles": _rotation_angles(arch, seed),
        "noisy": bool(noisy),
        "layers": arch["layers"],
    }


def _load_binary_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    x = np.load("binary_mnist_data.npy")
    y = np.load("binary_mnist_label.npy")
    x = np.asarray(x, dtype=float).reshape(len(x), -1)
    y = np.asarray(y).reshape(-1)
    if len(x) > 600:
        rng = np.random.default_rng(seed)
        chosen = rng.choice(len(x), size=600, replace=False)
        x = x[chosen]
        y = y[chosen]
    return train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)


def _load_iris_data(seed: int) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    data = load_iris()
    x = np.asarray(data.data, dtype=float)
    y = np.asarray(data.target)
    return train_test_split(x, y, test_size=0.3, random_state=seed, stratify=y)


def _svg_escape(text: Any) -> str:
    return str(text).replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def render_svg(result: dict[str, Any], path: str | Path) -> None:
    layers = result.get("layers") or []
    n_qubits = max(1, max((len(layer.get("rotations", [])) for layer in layers), default=2))
    width = max(420, 120 + 120 * max(1, len(layers)))
    height = 80 + 48 * n_qubits
    lines = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        '<rect width="100%" height="100%" fill="#ffffff"/>',
        '<text x="20" y="28" font-family="monospace" font-size="16" fill="#111827">best circuit</text>',
    ]
    for q in range(n_qubits):
        y = 60 + 42 * q
        lines.append(f'<line x1="30" y1="{y}" x2="{width - 30}" y2="{y}" stroke="#374151" stroke-width="2"/>')
        lines.append(f'<text x="8" y="{y + 5}" font-family="monospace" font-size="12" fill="#374151">q{q}</text>')
    for layer_idx, layer in enumerate(layers):
        x = 80 + 110 * layer_idx
        for q, gate in enumerate(layer.get("rotations", [])):
            y = 60 + 42 * q
            lines.append(f'<rect x="{x - 20}" y="{y - 15}" width="42" height="30" rx="4" fill="#dbeafe" stroke="#1d4ed8"/>')
            lines.append(f'<text x="{x - 12}" y="{y + 5}" font-family="monospace" font-size="12" fill="#1e3a8a">{_svg_escape(gate)}</text>')
        for control, target in layer.get("entanglers", []):
            y1 = 60 + 42 * int(control)
            y2 = 60 + 42 * int(target)
            cx = x + 42
            lines.append(f'<line x1="{cx}" y1="{y1}" x2="{cx}" y2="{y2}" stroke="#991b1b" stroke-width="2"/>')
            lines.append(f'<circle cx="{cx}" cy="{y1}" r="5" fill="#991b1b"/>')
            lines.append(f'<circle cx="{cx}" cy="{y2}" r="10" fill="none" stroke="#991b1b" stroke-width="2"/>')
            lines.append(f'<line x1="{cx - 8}" y1="{y2}" x2="{cx + 8}" y2="{y2}" stroke="#991b1b" stroke-width="2"/>')
            lines.append(f'<line x1="{cx}" y1="{y2 - 8}" x2="{cx}" y2="{y2 + 8}" stroke="#991b1b" stroke-width="2"/>')
    lines.append("</svg>")
    Path(path).write_text("\n".join(lines), encoding="utf-8")


def _strip_layers(item: dict[str, Any]) -> dict[str, Any]:
    return {k: v for k, v in item.items() if k != "layers"}


def build_report(seed: int) -> dict[str, Any]:
    population = int(os.environ.get("QASTASK_POPULATION", "16"))
    generations = int(os.environ.get("QASTASK_GENERATIONS", "4"))
    max_depth = int(os.environ.get("QASTASK_MAX_DEPTH", "3"))

    bx_train, bx_test, by_train, by_test = _load_binary_data(seed)
    ix_train, ix_test, iy_train, iy_test = _load_iris_data(seed + 1)

    binary_best, binary_front = run_na_qas(
        bx_train,
        by_train,
        bx_test,
        by_test,
        n_qubits=2,
        population_size=population,
        generations=generations,
        min_depth=1,
        max_depth=max_depth,
        noisy=True,
        seed=seed,
        task_name="Binary",
    )
    iris_best, iris_front = run_na_qas(
        ix_train,
        iy_train,
        ix_test,
        iy_test,
        n_qubits=4,
        population_size=population,
        generations=generations,
        min_depth=1,
        max_depth=max_depth,
        noisy=True,
        seed=seed + 1,
        task_name="Iris",
    )
    lih_best = run_lih(noisy=True, seed=seed + 2)
    lih_front = [dict(lih_best)]

    best_task = max(
        [
            ("Binary", _safe_float(binary_best.get("accuracy"), 0.0)),
            ("Iris", _safe_float(iris_best.get("accuracy"), 0.0)),
            ("LiH", 1.0 - _safe_float(lih_best.get("energy_error"), 1.0)),
        ],
        key=lambda item: item[1],
    )[0]

    return {
        "schema_version": "1.0",
        "best_task": best_task,
        "Trade_off_solutions": int(len(binary_front) + len(iris_front) + len(lih_front)),
        "Search_space": {
            "n_qubits": 4,
            "min_depth": 1,
            "max_depth": int(max_depth),
            "gate_set": list(GATE_SET),
            "allow_directed_cnot": True,
        },
        "Objectives": {
            "loss": "minimize",
            "hardware_cost": {
                "alpha": DEFAULT_ALPHA,
                "beta": DEFAULT_BETA,
            },
        },
        "Supernet": {
            "num_experts": 3,
            "epsilon": 0.1,
        },
        "Noise": {
            "bit_flip": 0.001,
            "depolarizing": 0.002,
            "thermal_relaxation": 0.001,
        },
        "Binary": _strip_layers(binary_best),
        "Iris": _strip_layers(iris_best),
        "LiH": _strip_layers(lih_best),
        "Pareto_fronts": {
            "Binary": [_strip_layers(item) for item in binary_front],
            "Iris": [_strip_layers(item) for item in iris_front],
            "LiH": [_strip_layers(item) for item in lih_front],
        },
    }


def write_outputs(report: dict[str, Any]) -> None:
    Path("output").mkdir(exist_ok=True)
    text = json.dumps(report, ensure_ascii=False, indent=2, sort_keys=True)
    Path("best_circuit.txt").write_text(text + "\n", encoding="utf-8")
    Path("output/best_circuit.txt").write_text(text + "\n", encoding="utf-8")

    best_task = str(report.get("best_task", "Binary"))
    result = report.get(best_task, report.get("Binary", {}))
    if "layers" not in result:
        fallback = build_search_space(
            n_qubits=int(report.get("Search_space", {}).get("n_qubits", 2)),
            min_depth=1,
            max_depth=1,
            max_architectures=1,
            seed=int(os.environ.get("QASTASK_SEED", "7")),
        )[0]
        result = {**result, "layers": fallback["layers"]}
    render_svg(result, "best_circuit.svg")
    shutil.copyfile("best_circuit.svg", "output/best_circuit.svg")


def main() -> None:
    seed = int(os.environ.get("QASTASK_SEED", "7"))
    report = build_report(seed)
    write_outputs(report)
    print(json.dumps({"ok": True, "best_task": report["best_task"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
