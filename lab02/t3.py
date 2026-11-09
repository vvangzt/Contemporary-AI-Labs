import random
from typing import Optional, List, Dict, Any

class TreeNode:
    """题目指定的博弈树节点结构"""
    def __init__(self, value: Optional[int] = None, children: Optional[List['TreeNode']] = None):
        self.value = value  # 叶子节点的得分 (None 表示非叶子节点)
        self.children = children if children is not None else []  # 子节点列表

# ==================== 核心算法实现 ====================

def alpha_beta_search(
    node: TreeNode,
    alpha: float,
    beta: float,
    is_maximizing: bool,
    stats: Optional[Dict[str, int]] = None,
    order_mode: str = "natural",
    true_values: Optional[Dict[int, int]] = None
) -> int:
    """
    Alpha-Beta 剪枝递归核心函数
    :param order_mode: 子节点访问顺序
        - "natural": 按生成顺序（默认，与题目要求一致）
        - "reverse": 逆序（对照实验用）
        - "oracle":  按子树真实博弈值排序（MAX 降序 / MIN 升序，理论最优顺序，需提供 true_values）
    """
    if stats is not None:
        stats["visited_nodes"] += 1

    # 叶子节点直接返回分值
    if not node.children or node.value is not None:
        return node.value

    children = list(node.children)
    if order_mode == "reverse":
        children.reverse()
    elif order_mode == "oracle" and true_values is not None:
        children.sort(key=lambda c: true_values[id(c)], reverse=is_maximizing)

    if is_maximizing:
        max_eval = float('-inf')
        for i, child in enumerate(children):
            eval_score = alpha_beta_search(child, alpha, beta, False, stats, order_mode, true_values)
            max_eval = max(max_eval, eval_score)
            alpha = max(alpha, eval_score)
            # 剪枝判断: alpha >= beta
            if beta <= alpha:
                if stats is not None:
                    # 统计被剪掉的剩余兄弟分支数
                    stats["pruned_branches"] += (len(children) - 1 - i)
                break
        return max_eval
    else:
        min_eval = float('inf')
        for i, child in enumerate(children):
            eval_score = alpha_beta_search(child, alpha, beta, True, stats, order_mode, true_values)
            min_eval = min(min_eval, eval_score)
            beta = min(beta, eval_score)
            # 剪枝判断: beta <= alpha
            if beta <= alpha:
                if stats is not None:
                    stats["pruned_branches"] += (len(children) - 1 - i)
                break
        return min_eval

# 题目要求的标准输入接口：传入根节点，返回 MAX 玩家最佳得分
def get_best_score(root: TreeNode) -> int:
    return alpha_beta_search(root, float('-inf'), float('inf'), is_maximizing=True)

# ==================== 报告对照实验工具 ====================

def minimax_without_pruning(node: TreeNode, is_maximizing: bool, stats: Dict[str, int]) -> int:
    """无剪枝的纯 Minimax 算法（用于实验报告做开销对照）"""
    stats["visited_nodes"] += 1
    if not node.children or node.value is not None:
        return node.value

    if is_maximizing:
        max_eval = float('-inf')
        for child in node.children:
            max_eval = max(max_eval, minimax_without_pruning(child, False, stats))
        return max_eval
    else:
        min_eval = float('inf')
        for child in node.children:
            min_eval = min(min_eval, minimax_without_pruning(child, True, stats))
        return min_eval

def generate_random_tree(depth: int, branching: int, min_val: int = 0, max_val: int = 30,
                         seed: int = 42) -> TreeNode:
    """
    自设拓展实验：生成随机完整博弈树
    深度 depth 表示叶子在第 depth 层（根为第 0 层），每个内部节点有 branching 个子节点。
    固定 seed 保证实验可复现。
    """
    rng = random.Random(seed)

    def build(d: int) -> TreeNode:
        if d == 0:
            return TreeNode(value=rng.randint(min_val, max_val))
        return TreeNode(children=[build(d - 1) for _ in range(branching)])

    return build(depth)

def compute_true_values(root: TreeNode) -> Dict[int, int]:
    """先用一次完整 Minimax 求出所有节点的真实博弈值，供 oracle 最优排序使用（实验用，非题目接口）"""
    vals: Dict[int, int] = {}

    def mm(node: TreeNode, maximizing: bool) -> int:
        if not node.children or node.value is not None:
            vals[id(node)] = node.value
            return node.value
        child_vals = [mm(c, not maximizing) for c in node.children]
        v = max(child_vals) if maximizing else min(child_vals)
        vals[id(node)] = v
        return v

    mm(root, True)
    return vals

# JSON 格式解析构建函数（对应 PPT 第 10-11 页的输入样例格式）
def build_tree_from_json(node_data: Dict[str, Any]) -> TreeNode:
    if "value" in node_data:
        return TreeNode(value=node_data["value"])
    children = [build_tree_from_json(child) for child in node_data.get("children", [])]
    return TreeNode(children=children)

# 依 PPT 第 12 页构建示例树
def build_ppt_tree() -> TreeNode:
    leaf1 = TreeNode(value=3)
    leaf2 = TreeNode(value=5)
    leaf3 = TreeNode(value=6)
    leaf4 = TreeNode(value=9)
    leaf5 = TreeNode(value=1)
    leaf6 = TreeNode(value=2)

    max1 = TreeNode(children=[leaf1, leaf2])
    max2 = TreeNode(children=[leaf3, leaf4])
    max3 = TreeNode(children=[leaf5, leaf6])

    min1 = TreeNode(children=[max1, max2])
    min2 = TreeNode(children=[max2, max3])

    root = TreeNode(children=[min1, min2])
    return root

#测试代码
if __name__ == "__main__":
    root = build_ppt_tree()

    # 1. 运行 Alpha-Beta 剪枝（题目要求测试）
    ab_stats = {"visited_nodes": 0, "pruned_branches": 0}
    best_val = alpha_beta_search(root, float('-inf'), float('inf'), is_maximizing=True, stats=ab_stats)

    print("===== 题目三测试结果 =====")
    print(f"MAX 玩家的最佳得分 (预期 5): {best_val}")
    print(f"Alpha-Beta 访问节点数: {ab_stats['visited_nodes']}")
    print(f"Alpha-Beta 剪枝次数: {ab_stats['pruned_branches']}")

    # 2. 报告对照实验：Minimax 无剪枝对照
    mm_stats = {"visited_nodes": 0}
    mm_val = minimax_without_pruning(root, is_maximizing=True, stats=mm_stats)
    print("\n-- 对照实验 (用于报告对比开销与剪枝效率) --")
    print(f"纯 Minimax 访问节点数: {mm_stats['visited_nodes']}")
    print(f"搜索开销压缩率: {(1 - ab_stats['visited_nodes'] / mm_stats['visited_nodes']) * 100:.2f}%")

    # 3. 对照实验：同一棵 PPT 树上不同子节点访问顺序的开销
    print("\n-- 访问顺序对照 (PPT 树, 报告要求) --")
    for mode in ("natural", "reverse"):
        s = {"visited_nodes": 0, "pruned_branches": 0}
        v = alpha_beta_search(root, float('-inf'), float('inf'), True, s, order_mode=mode)
        print(f"[{mode:>7}] 最佳得分: {v}, 访问节点数: {s['visited_nodes']}, 剪枝分支数: {s['pruned_branches']}")

    # 4. 验证 JSON 解析构建功能
    json_sample = {
        "is_maximizing_player": True,
        "children": [
            {
                "is_maximizing_player": False,
                "children": [
                    {"children": [{"value": 3}, {"value": 5}]},
                    {"children": [{"value": 6}, {"value": 9}]}
                ]
            },
            {
                "is_maximizing_player": False,
                "children": [
                    {"children": [{"value": 6}, {"value": 9}]},
                    {"children": [{"value": 1}, {"value": 2}]}
                ]
            }
        ]
    }
    json_root = build_tree_from_json(json_sample)
    print(f"\n从 JSON 字典结构求解结果: {get_best_score(json_root)}")

    # 5. 自设拓展实验：不同规模的随机博弈树 x 访问顺序，观察剪枝效率变化
    #    oracle 顺序需要预先知道各子树真实值（用一次纯 Minimax 预计算），代表理论最优访问顺序
    print("\n===== 自设拓展实验：随机博弈树规模与访问顺序 (seed=42, 叶值 0~30) =====")
    print(f"{'配置':<16}{'纯Minimax':>10}{'AB自然序':>10}{'AB逆序':>10}{'AB最优序':>10}{'根值校验':>10}")
    for depth, branching in [(6, 3), (4, 5), (8, 2)]:
        tree = generate_random_tree(depth, branching, seed=42)
        true_vals = compute_true_values(tree)
        leaves = branching ** depth

        row = []
        root_val = None
        for mode in ("minimax", "natural", "reverse", "oracle"):
            s = {"visited_nodes": 0, "pruned_branches": 0}
            if mode == "minimax":
                v = minimax_without_pruning(tree, True, s)
            else:
                v = alpha_beta_search(tree, float('-inf'), float('inf'), True, s,
                                      order_mode=mode, true_values=true_vals)
            if root_val is None:
                root_val = v
            elif v != root_val:
                print("警告: 各方法结果不一致!")
            row.append(s["visited_nodes"])

        config = f"D={depth}, b={branching}"
        print(f"{config:<16}{row[0]:>10}{row[1]:>10}{row[2]:>10}{row[3]:>10}{root_val:>10}   (叶子数 {leaves})")
