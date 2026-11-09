# 当代人工智能 · 课程实验

> **Contemporary Artificial Intelligence — Course Labs**

本仓库收录《当代人工智能》课程的全部实验项目。每个实验独立成目录，按 `lab01`、`lab02` 编号，
目录内包含该实验的源码、数据、图表、实验报告与说明文档。

---

## 实验索引

| 目录 | 实验主题 | 关键内容 |
| --- | --- | --- |
| [`lab01`](./lab01) | 高性能新闻文本分类与泛化机理探究 | TF-IDF + 逻辑回归 / MLP、特征消融、Mini-batch 损失分析、Word2Vec 与 Mixup 对比 |
| [`lab02`](./lab02) | 搜索算法：A\* 与 Alpha-Beta 剪枝 | 加权 A\*、Minimax / Alpha-Beta、置换表缓存、规模与访问顺序对照 |
| `lab03` | 待更新 | — |

---

## 目录结构

```
.
├── lab01/                          实验一：文本分类
│   ├── figures/                    实验图表
│   ├── *.py                        各步骤脚本（逻辑回归 / MLP / 消融 / 词向量）
│   ├── train_data.csv              训练数据
│   ├── test_data_unlabeled.csv     测试数据
│   ├── prediction.csv              预测结果
│   ├── README.md                   实验说明与复现指南
│   └── TUNING.md                   调参与实验记录
│
└── lab02/                          实验二：搜索算法
    ├── search_core.py              公共搜索核心（A* / 单源最短路 / 双向 BFS）
    ├── t1.py ~ t4.py               四道题的实现
    ├── bench.py                    规模实验与图表生成
    ├── verify.py                   一键验证（用例 + 边界 + 不变量）
    ├── figs/                       报告用图表
    ├── results_*.csv               基准实验原始数据
    ├── 实验二实验报告_搜索算法.pdf  实验报告
    └── README.md                   完整设计说明与实验数据
```

---

## 快速开始

各实验的环境与运行方式见其目录下的 `README.md`。

| 实验 | 环境 | 入口 |
| --- | --- | --- |
| lab01 | Python 3.10+，scikit-learn / numpy / pandas / matplotlib | 见 [`lab01/README.md`](./lab01/README.md) 的复现步骤 |
| lab02 | Python 3.8+（题目仅用标准库），`bench.py` 需要 matplotlib | `python t1.py` … `python t4.py`，`python verify.py`，`python bench.py` |

---

## 约定

- 每个实验一个独立目录，互不依赖、可单独运行。
- 随机实验固定随机种子，结果可复现；节点数等确定性指标与机器无关。
- 提交内容包含源码、实验数据 / 图表，以及实验报告。
- 目录名统一为 `labNN`（两位数），便于后续实验继续追加。

---

## 作者

[vvangzt](https://github.com/vvangzt)
