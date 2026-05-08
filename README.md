# Harness Dataset SII 2026 Summer Camp

本仓库提供用于 **Harness Engineering 文本分类任务** 的多领域 JSONL 数据集。数据集设计目标是帮助评估 Harness 在有限上下文窗口下的泛化能力、鲁棒性和安全性表现。

## 数据集概览

本数据集包含 5 个领域：

- `finance`
- `ecommerce`
- `tech_support`
- `medical_triage`
- `news_topic`

每个领域均包含：

- `train.jsonl`：150 条训练样本
- `test.jsonl`：300 条测试样本

每条样本采用统一 JSONL 格式：

```json
{"text": "...", "label": "..."}
```

其中：

- `text`：待分类的自然语言文本
- `label`：目标类别标签，预测结果需要与该字段完全一致

## 仓库结构

```text
Harness_Dataset_SII2026Summer-Camp/
├── finance/
│   ├── train.jsonl
│   └── test.jsonl
├── ecommerce/
│   ├── train.jsonl
│   └── test.jsonl
├── tech_support/
│   ├── train.jsonl
│   └── test.jsonl
├── medical_triage/
│   ├── train.jsonl
│   └── test.jsonl
├── news_topic/
│   ├── train.jsonl
│   └── test.jsonl
├── manifest.json
├── dataset_summary.json
└── README.md
```

## 设计目标

该数据集主要覆盖以下评估维度：

### 1. 多领域泛化能力

数据集覆盖金融、电商、技术支持、医疗分诊、新闻主题等多个领域，用于测试 Harness 是否能够适配不同任务语义和标签空间。

### 2. OOD 分布外泛化能力

每个领域的数据集中都混入了跨领域样本。例如：

- `finance` 中可能包含电商支付、医疗账单、技术门户等语境
- `ecommerce` 中可能包含金融支付、物流新闻、技术订阅等语境
- `tech_support` 中可能包含金融门户、医疗系统、新闻上传系统等语境
- `medical_triage` 中可能包含购物、工作、新闻阅读等外部场景
- `news_topic` 中可能包含商业、医疗、科技、交通等综合新闻主题

这些样本用于测试模型是否能在领域迁移或语境混合时保持稳定分类能力。

### 3. MCS 多选题能力

每个领域均包含自然语言多选题样本。  
此类样本的标签为：

- `A`
- `B`
- `C`
- `D`

用于测试 Harness 是否能处理复杂自然语言选择题，而不是只依赖传统意图分类模式。

### 4. Tone 鲁棒性

每个领域都包含不同语气或表达方式的样本，包括：

- neutral
- polite
- urgent
- frustrated
- casual
- formal
- terse
- verbose

这些样本用于测试不同提问语气下，模型是否仍能输出一致且正确的分类标签。

### 5. Prompt Injection 防护

每个领域均包含提示词注入样本，例如：

- 要求模型忽略原始指令
- 伪造 system / developer override
- 要求模型输出错误标签
- 要求泄露 hidden prompt
- 要求固定输出某个选项

这类样本用于测试 Harness 是否能将用户文本中的恶意指令视为待分类内容，而不是执行这些指令。

## 数据规模

| Domain         | Train Samples | Test Samples | Label Count | MCS Train | MCS Test | Injection Samples |
| -------------- | ------------: | -----------: | ----------: | --------: | -------: | ----------------- |
| finance        |           150 |          300 |          19 |        15 |       45 | Included          |
| ecommerce      |           150 |          300 |          19 |        15 |       45 | Included          |
| tech_support   |           150 |          300 |          19 |        15 |       45 | Included          |
| medical_triage |           150 |          300 |          19 |        15 |       45 | Included          |
| news_topic     |           150 |          300 |          19 |        15 |       45 | Included          |

说明：

- 每个领域至少包含 15 个领域分类标签
- 同时额外包含 `A/B/C/D` 作为 MCS 多选题标签
- 测试集中出现的所有标签均保证在对应训练集中出现过

## 使用方式

可以直接按领域读取数据：

```python
import json
from pathlib import Path

def load_jsonl(path):
    data = []
    with open(path, "r", encoding="utf-8") as f:
        for line in f:
            data.append(json.loads(line))
    return data

train_data = load_jsonl("finance/train.jsonl")
test_data = load_jsonl("finance/test.jsonl")

print(len(train_data))
print(len(test_data))
print(train_data[0])
```

## 与 Harness 任务结合

典型使用方式如下：

```python
for sample in train_data:
    harness.update(sample["text"], sample["label"])

correct = 0

for sample in test_data:
    pred = harness.predict(sample["text"])
    if pred == sample["label"]:
        correct += 1

accuracy = correct / len(test_data)
print("Accuracy:", accuracy)
```

## 数据校验

可以使用以下脚本检查数据集格式和标签约束：

```python
import json
from pathlib import Path
from collections import Counter

domains = [
    "finance",
    "ecommerce",
    "tech_support",
    "medical_triage",
    "news_topic",
]

for domain in domains:
    train_path = Path(domain) / "train.jsonl"
    test_path = Path(domain) / "test.jsonl"

    train = [json.loads(line) for line in train_path.open(encoding="utf-8")]
    test = [json.loads(line) for line in test_path.open(encoding="utf-8")]

    train_labels = {row["label"] for row in train}
    test_labels = {row["label"] for row in test}

    assert len(train) == 150, f"{domain}: train size error"
    assert len(test) == 300, f"{domain}: test size error"
    assert test_labels.issubset(train_labels), f"{domain}: test contains unseen labels"

    print(domain)
    print("  train samples:", len(train))
    print("  test samples:", len(test))
    print("  label count:", len(train_labels))
    print("  train label distribution:", Counter(row["label"] for row in train))
    print("  test label distribution:", Counter(row["label"] for row in test))
```

## 文件说明

### `manifest.json`

记录每个领域的数据文件路径、标签集合和数据说明。

### `dataset_summary.json`

记录每个领域的数据规模、标签数量、MCS 样本数量和注入样本统计信息。

### `*/train.jsonl`

对应领域的训练集。

### `*/test.jsonl`

对应领域的测试集。

## 注意事项

1. 本数据集主要用于 Harness Engineering、文本分类、OOD 泛化、MCS 选择题泛化和 Prompt Injection 鲁棒性测试。
2. 数据中的 Prompt Injection 文本是评测内容的一部分，模型或 Harness 不应执行其中的恶意指令。
3. 对于 MCS 样本，预测结果应直接返回 `A`、`B`、`C` 或 `D`。
4. 对于普通分类样本，预测结果应直接返回对应的标签字符串。
5. 评估时建议使用 exact match accuracy。

## Recommended Evaluation Metric

推荐使用准确率作为主要指标：

```python
accuracy = correct_predictions / total_predictions
```

由于该任务要求返回标签字符串，建议使用 exact match，而不是模糊匹配。

## License

This dataset is provided for research, education, and evaluation purposes.
