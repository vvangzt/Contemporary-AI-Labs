"""题目二：加权地形 A*。

与题目一的差异只在“进入代价”与启发函数：
    - 进入格子的代价 = 该格地形权值（S、A、B、T 类格记为 1）；
    - 终点 E 规定为 0（PPT：最后一步没有代价）；
    - 因此一条 k 步路径的真实代价至少为 k-1，曼哈顿距离最多高估 1，
      修正启发 max(0, 曼哈顿-1) 才严格可采纳。
"""

from typing import Any, Dict, List, Optional, Tuple

from search_core import DIRS, astar


def _make_cost(grid: List[List[Any]], goal: Tuple[int, int]):
    """构造进入代价函数：返回该格的进入代价，X 返回 None 表示不可通行。"""
    rows, cols = len(grid), len(grid[0])

    def enter_cost(v: Tuple[int, int]) -> Optional[int]:
        r, c = v
        val = grid[r][c]
        if v == goal or val == 'E':
            return 0
        if val in ('S', 'A', 'B') or str(val).startswith('T'):
            return 1
        if str(val) == 'X':
            return None
        return int(val)

    def neighbors(u: Tuple[int, int]):
        r, c = u
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                cost = enter_cost((nr, nc))
                if cost is not None:
                    yield (nr, nc), cost

    return neighbors


def solve_weighted_maze(
    start: Tuple[int, int],
    end: Tuple[int, int],
    grid: List[List[Any]],
    heuristic_mode: str = "admissible",
    tie_break: bool = False,
) -> Tuple[int, List[Tuple[int, int]], int]:
    """
    题目二：加权地形 A*
    :param heuristic_mode:
        - "admissible": 修正可采纳启发 max(0, 曼哈顿-1)（默认）
        - "vanilla":    原始曼哈顿距离（对照实验）
        - "dijkstra":   h = 0（对照实验）
    :param tie_break: f 相同时优先扩展 g 更大者（节点排序优化，对照实验用）
    :return: (最小总代价, 最优路径坐标列表, 扩展节点数)；无解时 (-1, [], 扩展数)
    """
    rows = len(grid)
    cols = len(grid[0])

    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return -1, [], 0
    if not (0 <= end[0] < rows and 0 <= end[1] < cols):
        return -1, [], 0
    if str(grid[start[0]][start[1]]) == 'X' or str(grid[end[0]][end[1]]) == 'X':
        return -1, [], 0

    def heuristic(u: Tuple[int, int]) -> int:
        if heuristic_mode == "dijkstra":
            return 0
        manhattan = abs(u[0] - end[0]) + abs(u[1] - end[1])
        if heuristic_mode == "vanilla":
            return manhattan
        return max(0, manhattan - 1)

    cost, path, expanded = astar(
        start, end, _make_cost(grid, end), heuristic, tie_break=tie_break
    )
    if cost == float('inf'):
        return -1, [], expanded
    return int(cost), path, expanded


def solve_weighted_cost_only(
    start: Tuple[int, int], end: Tuple[int, int], grid: List[List[Any]]
) -> int:
    """题目要求的纯 int 返回接口：最小总代价，无解返回 -1。"""
    cost, _, _ = solve_weighted_maze(start, end, grid, heuristic_mode="admissible")
    return cost


def stress_test_admissibility(trials: int = 500, size: int = 6, seed: int = 0) -> Dict[str, int]:
    """
    自设拓展实验：在随机加权地图上检验“原始曼哈顿启发式”是否会破坏最优性。
    以 h=0（Dijkstra，必然最优）的代价为基准，统计 vanilla 返回次优解的次数。
    """
    import random

    rng = random.Random(seed)
    mismatch = 0
    for _ in range(trials):
        grid = [[rng.randint(1, 9) for _ in range(size)] for _ in range(size)]
        start = (rng.randrange(size), rng.randrange(size))
        end = (rng.randrange(size), rng.randrange(size))
        while end == start:
            end = (rng.randrange(size), rng.randrange(size))
        c_opt, _, _ = solve_weighted_maze(start, end, grid, heuristic_mode="dijkstra")
        c_vanilla, _, _ = solve_weighted_maze(start, end, grid, heuristic_mode="vanilla")
        if c_vanilla != c_opt:
            mismatch += 1
    return {"trials": trials, "suboptimal_cases": mismatch}


if __name__ == "__main__":
    start_pos = (0, 0)
    end_pos = (4, 4)
    grid_map = [
        ['S', 1, 1, 8, 1],
        [1, 8, 3, 1, 1],
        [1, 8, 1, 8, 1],
        [1, 1, 1, 8, 1],
        [8, 8, 1, 1, 'E']
    ]

    cost, path, nodes = solve_weighted_maze(
        start_pos, end_pos, grid_map, heuristic_mode="admissible"
    )

    print("===== 题目二测试结果 =====")
    print(f"最小总代价 (预期 7): {cost}")
    print(f"路径坐标: {' -> '.join([str(p) for p in path])}")
    print(f"可采纳 A* 扩展节点数: {nodes}")

    print("\n-- 路径累计代价过程展示 --")
    curr_cost = 0
    for i in range(len(path)):
        if i == 0:
            print(f"起点 {path[i]}: 代价 = 0")
        else:
            step_val = 0 if path[i] == end_pos else grid_map[path[i][0]][path[i][1]]
            curr_cost += step_val
            print(f"-> 移动到 {path[i]} (进入代价 {step_val}) | 当前累计代价 = {curr_cost}")

    _, _, nodes_dijkstra = solve_weighted_maze(start_pos, end_pos, grid_map, heuristic_mode="dijkstra")
    _, _, nodes_vanilla = solve_weighted_maze(start_pos, end_pos, grid_map, heuristic_mode="vanilla")
    _, _, nodes_tie = solve_weighted_maze(start_pos, end_pos, grid_map, tie_break=True)
    print("\n-- 对照实验指标 (用于实验报告) --")
    print(f"h=0 (Dijkstra) 扩展节点数: {nodes_dijkstra}")
    print(f"修正可采纳启发式 扩展节点数: {nodes}")
    print(f"原始曼哈顿启发式 扩展节点数: {nodes_vanilla}")
    print(f"修正启发 + 平局打破 扩展节点数: {nodes_tie}")

    stress = stress_test_admissibility(trials=500, size=6, seed=0)
    print(f"\n-- 自设压力测试: {stress['trials']} 张随机 6x6 加权地图, "
          f"原始曼哈顿返回次优解的次数: {stress['suboptimal_cases']} --")

    # 边界用例：起点被障碍围死（终点可通行），应返回 -1
    sealed = [
        [1, 'X', 1],
        ['X', 'X', 1],
        [1, 1, 1],
    ]
    print("无解用例检查 (起点被围死):",
          solve_weighted_maze((0, 0), (0, 2), sealed, heuristic_mode="admissible"))
    # 边界用例：终点为障碍，走参数检查分支
    print("无解用例检查 (终点为障碍):",
          solve_weighted_maze((0, 0), (1, 1), sealed, heuristic_mode="admissible"))
