# 抗噪量子架构搜索任务

你需要在固定入口文件 `main.py` 中实现一套抗噪量子架构搜索（NA-QAS）方法。出题方已经提供了可运行的基础框架、公开格式校验脚本和公开 smoke test；你的主要工作是提升 `main.py` 中核心算法接口的质量，而不是重新设计工程目录或输出协议。

## 交付边界

- 只需要修改 `main.py`。
- 不需要新建工程目录，不需要改 `verifier.py` 或 `smoke_test.py`。
- 运行 `python main.py` 必须稳定结束。
- 运行结束后必须在当前目录生成 `best_circuit.txt` 和 `best_circuit.svg`。
- 同时建议镜像生成 `output/best_circuit.txt` 和 `output/best_circuit.svg`，starter 已经包含这部分逻辑。

最终提交时，评测会重新运行你的 `main.py`，不会把你本地生成的 `best_circuit.txt` 当作真实答案。

## 必须实现的核心接口

`main.py` 中已经预置以下函数。你可以重写内部逻辑，但请保持函数名、参数兼容和返回结构可 JSON 序列化。

```python
def build_search_space(n_qubits=2, min_depth=1, max_depth=3, max_architectures=None, seed=0):
    ...

def apply_noise_model(circuit_or_features, noise_config=None, seed=0):
    ...

def run_na_qas(
    x_train,
    y_train,
    x_test,
    y_test,
    n_qubits=2,
    n_outputs=None,
    population_size=16,
    generations=4,
    min_depth=1,
    max_depth=3,
    noisy=True,
    seed=0,
    task_name="Binary",
):
    ...

def select_best_architecture(pareto_front, objective_weights=None):
    ...

def predict_with_architecture(architecture, x_train, y_train, x_eval, n_outputs=None, noisy=True, seed=0):
    ...

def run_lih(noisy=True, seed=0, dist=1.546):
    ...
```

评测重点会放在这些接口是否能泛化到隐藏数据和隐藏随机种子。隐藏评测不会只相信 `run_na_qas` 返回的自报 `accuracy` 或 `loss`，而会用 `run_na_qas` 选出的架构调用 `predict_with_architecture`，在未暴露标签的 holdout 上独立计算分类质量。
LiH 会通过 `run_lih` 在多个 Li-H 键长、噪声设置和随机种子下独立调用，不会只读取 `best_circuit.txt` 中的固定数值。

## 算法要求

你需要围绕以下核心点改进 starter baseline：

1. 搜索空间：架构由可变数量的 layer 组成。每个 layer 包含每个量子比特上的旋转门，旋转门从 `{Rx, Ry, Rz}` 中选择；纠缠层包含 0 到多个有向 CNOT 门。
2. 噪声模型：训练和搜索阶段都要考虑噪声。噪声至少覆盖 bit-flip、depolarizing 和 thermal relaxation 的参数化建模。可以基于 MindQuantum 的线路/噪声能力实现，也可以在特征、期望值或损失估计阶段实现等价的噪声感知评估。
3. Hybrid Supernet：实现参数共享或可复用训练策略，并包含多个经典专家头。训练时使用 epsilon-greedy 选择专家：大多数时候选择 loss 最小专家，少数时候随机探索。
4. 多目标优化：使用 NSGA-II 或等价的非支配排序策略优化两个目标：任务 loss 和硬件开销。
5. 硬件开销：`hardware_cost = alpha * cnot + beta * depth`，其中 `alpha` 和 `beta` 应在输出中记录。
6. 质量和资源平衡：隐藏评分会同时看独立验证准确率/能量误差与线路深度、CNOT 数。不能为了提高准确率或降低能量误差无限增加硬件开销；超出资源预算会显著扣分或封顶。
7. LiH：`run_lih` 不应只返回固定常数。隐藏评测会在多个 Li-H 键长和噪声设置下调用该函数，并检查能量曲线、参考能量、误差和硬件开销的一致性。

## 输出文件 schema

`best_circuit.txt` 必须是合法 JSON object，并至少包含下面字段。字段名区分大小写。

```json
{
  "schema_version": "1.0",
  "best_task": "Binary",
  "Trade_off_solutions": 3,
  "Search_space": {
    "n_qubits": 2,
    "min_depth": 1,
    "max_depth": 3,
    "gate_set": ["Rx", "Ry", "Rz"],
    "allow_directed_cnot": true
  },
  "Objectives": {
    "loss": "minimize",
    "hardware_cost": {
      "alpha": 1.0,
      "beta": 0.25
    }
  },
  "Supernet": {
    "num_experts": 3,
    "epsilon": 0.1
  },
  "Binary": {
    "accuracy": 0.85,
    "loss": 0.35,
    "depth": 4,
    "cnot": 3,
    "hardware_cost": 4.0,
    "architecture_index": 12,
    "rotation_angles": [0.1, 0.2, 0.3]
  },
  "Iris": {
    "accuracy": 0.90,
    "loss": 0.25,
    "depth": 5,
    "cnot": 4,
    "hardware_cost": 5.25,
    "architecture_index": 18,
    "rotation_angles": [0.1, 0.2, 0.3]
  },
  "LiH": {
    "ground_state_energy": -7.88,
    "reference_energy": -7.882,
    "energy_error": 0.002,
    "depth": 6,
    "cnot": 8,
    "hardware_cost": 9.5,
    "architecture_index": 21,
    "rotation_angles": [0.1, 0.2, 0.3]
  },
  "Pareto_fronts": {
    "Binary": [],
    "Iris": [],
    "LiH": []
  }
}
```

字段类型要求：

- `Trade_off_solutions` 必须是 integer。
- `accuracy` 必须是 0 到 1 之间的 number。
- `loss`、`hardware_cost`、`ground_state_energy`、`reference_energy`、`energy_error` 必须是 number。
- `depth`、`cnot`、`architecture_index` 必须是 integer。
- `rotation_angles` 必须是 number list。
- `Pareto_fronts.Binary`、`Pareto_fronts.Iris`、`Pareto_fronts.LiH` 必须是 list。

## 公开检查

开发时可以运行：

```bash
python smoke_test.py
```

或直接运行：

```bash
python verifier.py
```

公开检查只验证基础合法性，包括：

- `python main.py` 是否能正常结束；
- 是否生成 `best_circuit.txt`；
- 是否生成 `best_circuit.svg`；
- `best_circuit.txt` 是否为合法 JSON；
- JSON 字段和类型是否符合上述 schema；
- `main.py` 中的基础接口是否能 import。

公开检查不包含隐藏质量评分。隐藏评测会先做同样的基础合法性 gate；只有通过 gate 后才计算算法质量分。

## 数据与运行时间

当前目录提供 `binary_mnist_data.npy` 和 `binary_mnist_label.npy`。Iris 数据可以通过 `sklearn.datasets.load_iris` 加载。LiH 分子设置如下：

```python
dist = 1.546
geometry = [("Li", (0.0, 0.0, 0.0)), ("H", (0.0, 0.0, dist))]
basis = "sto3g"
multiplicity = 1
charge = 0
```

`python main.py` 应在合理时间内结束。可以通过环境变量控制规模，例如 `QASTASK_SEED`、`QASTASK_POPULATION`、`QASTASK_GENERATIONS`、`QASTASK_MAX_DEPTH`。隐藏评测会使用自己的参数和随机种子调用你的核心接口。
