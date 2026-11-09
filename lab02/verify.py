"""一键验证：四道题的核心输出与边界用例。

用法：
    python verify.py

全部断言通过时打印 "ALL CHECKS PASSED" 并以退出码 0 结束；
任一断言失败会抛出 AssertionError 并以非零退出码结束。
"""

import sys

import t4
from t1 import a_star_maze, solve_maze
from t2 import solve_weighted_cost_only, solve_weighted_maze
from t3 import build_ppt_tree, get_best_score

# ---------------- 课程给定测试用例 ----------------

GRID_1 = [
    ['S', '.', '.', 'X', '.'],
    ['.', 'X', '.', '.', '.'],
    ['.', 'X', '.', 'X', '.'],
    ['.', '.', '.', 'X', 'E'],
    ['X', 'X', '.', '.', '.'],
]
EXPECTED_PATH_1 = [(0, 0), (0, 1), (0, 2), (1, 2), (1, 3), (1, 4), (2, 4), (3, 4)]

GRID_2 = [
    ['S', 1, 1, 8, 1],
    [1, 8, 3, 1, 1],
    [1, 8, 1, 8, 1],
    [1, 1, 1, 8, 1],
    [8, 8, 1, 1, 'E'],
]


def check_cases():
    assert solve_maze((0, 0), (3, 4), GRID_1) == EXPECTED_PATH_1, "题目一路径不符"
    assert solve_weighted_cost_only((0, 0), (4, 4), GRID_2) == 7, "题目二代价不符"
    assert get_best_score(build_ppt_tree()) == 5, "题目三最佳得分不符"

    state = t4.GameState("A", "B", ("T1", "T2", "T3", "T4"))
    stats2 = {"visited_nodes": 0, "pruned_branches": 0}
    stats4 = {"visited_nodes": 0, "pruned_branches": 0}
    v2, a2 = t4.alpha_beta(state, 0, 2, float("-inf"), float("inf"), t4.DIST, stats2)
    v4, a4 = t4.alpha_beta(state, 0, 4, float("-inf"), float("inf"), t4.DIST, stats4)
    assert (a2, v2) == ("T2", -1), f"题目四 H=2 不符: {a2}, {v2}"
    assert (a4, v4) == ("T3", 3), f"题目四 H=4 不符: {a4}, {v4}"
    assert stats2["visited_nodes"] == 13, "题目四 H=2 节点数不符"
    assert stats4["visited_nodes"] == 47, "题目四 H=4 节点数不符"
    return stats2, stats4


def check_edge_cases():
    # 起点等于终点
    assert a_star_maze((0, 0), (0, 0), GRID_1) == ([(0, 0)], 1)
    assert solve_weighted_maze((0, 0), (0, 0), GRID_2)[:2] == (0, [(0, 0)])

    # 起点被障碍完全围死（终点可通行）：搜索耗尽
    sealed_1 = [['S', 'X', '.'], ['X', 'X', '.'], ['.', '.', 'E']]
    assert a_star_maze((0, 0), (2, 2), sealed_1) == ([], 1)

    sealed_2 = [[1, 'X', 1], ['X', 'X', 1], [1, 1, 1]]
    assert solve_weighted_maze((0, 0), (2, 1), sealed_2)[:2] == (-1, [])

    # 终点落在障碍上：参数检查直接返回
    assert a_star_maze((0, 0), (0, 1), sealed_1) == ([], 0)
    assert solve_weighted_maze((0, 0), (0, 1), sealed_2)[:2] == (-1, [])


def check_invariants():
    # 距离矩阵：单源调用 6 次；本图所有关键点入格代价为 1，故对称
    assert t4.MATRIX_CALLS == 6, "距离矩阵调用次数应为 6"
    pts = ["A", "B", "T1", "T2", "T3", "T4"]
    matrix, _ = t4.compute_distance_matrix()
    assert all(matrix[u][v] == t4.DIST[u][v] for u in pts for v in pts)
    assert all(t4.DIST[u][v] == t4.DIST[v][u] for u in pts for v in pts)

    # 缓存版与纯 Minimax 的博弈值一致
    state = t4.GameState("A", "B", ("T1", "T2", "T3", "T4"))
    mm = {"visited_nodes": 0}
    v_mm, _ = t4.minimax(state, 0, 4, t4.DIST, mm)
    ca = {"visited_nodes": 0, "cache_hits": 0}
    v_ca = t4.minimax_cached(state, 0, 4, t4.DIST, ca, {})
    assert v_mm == v_ca == 3, f"缓存值不一致: {v_mm} vs {v_ca}"

    # 剪枝不改变最优值：正序 / 逆序同值
    for reverse in (False, True):
        v, _ = t4.alpha_beta(state, 0, 4, float("-inf"), float("inf"),
                             t4.DIST, {"visited_nodes": 0, "pruned_branches": 0},
                             reverse_order=reverse)
        assert v == 3


def main():
    stats2, stats4 = check_cases()
    check_edge_cases()
    check_invariants()

    print("题目一：路径与 7 步最短路径一致")
    print("题目二：最小代价 7")
    print("题目三：最佳得分 5")
    print(f"题目四：H=2 -> T2/-1（{stats2['visited_nodes']} 节点），"
          f"H=4 -> T3/3（{stats4['visited_nodes']} 节点）")
    print("边界用例：起终点相同 / 起点围死 / 终点为障碍 均通过")
    print("不变量：矩阵 6 次调用且对称；缓存与 Minimax 同值；剪枝不改变最优值")
    print("\nALL CHECKS PASSED")
    return 0


if __name__ == "__main__":
    sys.exit(main())
