import os
import json
import subprocess
import glob
import re

def static_code_analysis():
    """扫描目录下的所有 py 文件，检查是否包含特定的代码约束特征"""
    py_files = glob.glob("**/*.py", recursive=True)
    combined_code = ""
    for f in py_files:
        if f == "eval_script.py":
            continue
        try:
            with open(f, 'r', encoding='utf-8') as file:
                combined_code += file.read() + "\n"
        except:
            pass

    # 1. 搜索空间约束：Rx, Ry, Rz, CNOT
    has_rx = bool(re.search(r'\b(RX|Rx)\b', combined_code))
    has_ry = bool(re.search(r'\b(RY|Ry)\b', combined_code))
    has_rz = bool(re.search(r'\b(RZ|Rz)\b', combined_code))
    has_cnot = bool(re.search(r'\b(X|CNOT)\b', combined_code))
    
    # 2. 噪声模型约束：三种噪声
    has_bitflip = bool(re.search(r'(BitFlip|bit_flip)', combined_code, re.IGNORECASE))
    has_depolar = bool(re.search(r'(Depolarizing|depolarize)', combined_code, re.IGNORECASE))
    has_thermal = bool(re.search(r'(ThermalRelaxation|thermal_relaxation|T1|T2)', combined_code, re.IGNORECASE))
    
    # 3. 笛卡尔积与可变深度特征
    has_variable_depth = bool(re.search(r'(depth|layer|min_depth|max_depth|cartesian|product|itertools\.product)', combined_code, re.IGNORECASE))

    return {
        "gates_valid": has_rx and has_ry and has_rz and has_cnot,
        "variable_depth": has_variable_depth,
        "noise_bitflip": has_bitflip,
        "noise_depolar": has_depolar,
        "noise_thermal": has_thermal
    }

def evaluate():
    result = {"valid": False, "score": 0.0, "pass_rate": 0.0, "summary": "", "details": []}
    score = 0.0
    
    # ==========================
    # 1. 编译与执行 (15分)
    # ==========================
    if not os.path.exists("main.py"):
        result["summary"] = "CRITICAL: 未找到主入口文件 main.py"
        print(">>>>> Start Structured Result\n" + json.dumps(result) + "\n>>>>> End Structured Result")
        return

    try:
        # Agent执行需跑完整个搜索验证流程，允许较长超时
        subprocess.run(["python", "main.py"], timeout=3000, check=True, capture_output=True)
        score += 15.0
        result["details"].append({"name": "编译与无报错执行", "status": "PASSED", "score": 15})
    except Exception as e:
        result["summary"] = f"执行异常或超时: {str(e)}"
        print(">>>>> Start Structured Result\n" + json.dumps(result) + "\n>>>>> End Structured Result")
        return

    # ==========================
    # 2. 搜索空间约束 (20分)
    # ==========================
    code_features = static_code_analysis()
    
    if code_features["gates_valid"]:
        score += 10.0
        result["details"].append({"name": "旋转门集与CNOT笛卡尔积限制", "status": "PASSED", "score": 10})
    if code_features["variable_depth"]:
        score += 10.0
        result["details"].append({"name": "可变深度特性验证", "status": "PASSED", "score": 10})
        
    # ==========================
    # 3. 噪声模拟器约束 (15分)
    # ==========================
    noise_score = 0.0
    if code_features["noise_bitflip"]: noise_score += 5.0
    if code_features["noise_depolar"]: noise_score += 5.0
    if code_features["noise_thermal"]: noise_score += 5.0
    score += noise_score
    result["details"].append({"name": "三种噪声信道检查", "status": "PASSED", "score": noise_score})

    # ==========================
    # 5. 输出规范性 (10分)
    # ==========================
    format_score = 0.0
    if os.path.exists("best_circuit.svg"):
        format_score += 5.0
    if os.path.exists("best_circuit.txt"):
        format_score += 5.0
    score += format_score
    result["details"].append({"name": "文件输出规范性 (SVG & TXT)", "status": "PASSED", "score": format_score})

    # 读取 best_circuit.txt 准备验证质量
    try:
        with open("best_circuit.txt", "r", encoding='utf-8') as f:
            data = json.load(f)
    except:
        data = None
        result["summary"] = f"无法按 JSON 格式解析 best_circuit.txt，质量分计0分。"

    # ==========================
    # 4. Pareto 质量与有效性 (40分)
    # ==========================
    if data:
        quality_score = 0.0
        
        # 4.1 Trade-off 表现 (5分)
        if data.get("Trade_off_solutions", 0) > 1:
            quality_score += 5.0
            
        # 4.2 分类任务精度表现 (10分)
        acc_b = data.get("Binary", {}).get("accuracy", 0)
        acc_i = data.get("Iris", {}).get("accuracy", 0)
        acc_avg = (acc_b + acc_i) / 2.0 if (acc_b and acc_i) else max(acc_b, acc_i)
        
        if acc_avg >= 0.95:
            quality_score += 10.0
        elif acc_avg > 0.80:
            quality_score += 7.0  # 10 - 3
        elif acc_avg > 0.65:
            quality_score += 5.0  # 10 - 5
        else:
            quality_score += 0.0  # 10 - 10

        # 4.3 分子基态误差表现 (10分)
        energy_error = data.get("LiH", {}).get("energy_error", 999)
        if energy_error < 0.007:
            quality_score += 10.0
        elif energy_error <= 0.01:
            quality_score += 5.0   # 10 - 5
        else:
            quality_score += 0.0   # 10 - 10

        # 4.4 硬件开销评分 (15分, 每项满分5分)
        hw_score = 0.0
        
        # 分子基态 LiH (深度<5, CNOT<5)
        l_d = data.get("LiH", {}).get("depth", 99)
        l_c = data.get("LiH", {}).get("cnot", 99)
        lih_hw = 5.0 - max(0, l_d - 4)*0.1 - max(0, l_c - 4)*0.2  # <5 Means <=4
        hw_score += max(0.0, lih_hw)
        
        # 鸢尾花 Iris (深度<10, CNOT<20)
        i_d = data.get("Iris", {}).get("depth", 99)
        i_c = data.get("Iris", {}).get("cnot", 99)
        iris_hw = 5.0 - max(0, i_d - 9)*0.1 - max(0, i_c - 19)*0.2 # <10 Means <=9
        hw_score += max(0.0, iris_hw)
        
        # 二分类 Binary (深度<8, CNOT<8)
        b_d = data.get("Binary", {}).get("depth", 99)
        b_c = data.get("Binary", {}).get("cnot", 99)
        bin_hw = 5.0 - max(0, b_d - 7)*0.1 - max(0, b_c - 7)*0.2   # <8 Means <=7
        hw_score += max(0.0, bin_hw)

        quality_score += hw_score
        score += quality_score
        result["details"].append({"name": "Pareto质量、精度梯队与开销扣分", "status": "PASSED", "score": quality_score})

    # ==========================
    # 汇总输出
    # ==========================
    result["valid"] = True
    result["score"] = score
    result["pass_rate"] = score / 100.0
    result["summary"] = f"评测完成，总得分: {score:.1f}/100"

    print(">>>>> Start Structured Result")
    print(json.dumps(result))
    print(">>>>> End Structured Result")

if __name__ == "__main__":
    evaluate()