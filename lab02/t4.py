"""题目四：竞争环境下的宝藏分配（综合题）。

流程：
    1. 用 A* / 单源最短路预计算关键点之间的最小移动代价，得到距离矩阵；
    2. 以 (双方位置, 剩余宝藏, 累计得分, 当前玩家) 描述博弈状态；
    3. 在 H=2、H=4 下分别用 Minimax 与 Alpha-Beta 求 A 的最优首步。

相对初版的三处工程/算法改进（报告中给出对照数据）：
    1. 距离矩阵：36 次成对 A*  →  6 次单源最短路，调用次数降至 1/6；
    2. 新增置换表版 Minimax，用于分析“缓存”随状态规模变化的收益；
    3. 新增深度对抗实验：不同搜索深度的两名玩家完整对弈，统计胜率。
"""

import json
import random
from typing import Any, Dict, List, Optional, Tuple

from search_core import dijkstra

# ---------------- 地图与规则 ----------------

GRID = [
    ['A', '1', '1', 'X', '2', '1', 'T1'],
    ['1', 'X', '6', 'X', '1', 'X', '1'],
    ['1', '1', 'T2', '1', '1', '1', '1'],
    ['X', '1', 'X', '8', 'X', '1', 'X'],
    ['1', '1', '1', '1', 'T3', '1', '1'],
    ['1', 'X', '9', 'X', '1', 'X', '1'],
    ['T4', '1', '1', 'X', '1', '1', 'B']
]

LOCATIONS: Dict[str, Tuple[int, int]] = {
    "A": (0, 0),
    "B": (6, 6),
    "T1": (0, 6),
    "T2": (2, 2),
    "T3": (4, 4),
    "T4": (6, 0),
}

TREASURE_VALUES: Dict[str, int] = {"T1": 9, "T2": 6, "T3": 7, "T4": 7}

# 拓展实验用的额外宝藏（均落在原图代价为 1 的空地上，不改变原题地图）
EXTRA_TREASURES = [
    ("T5", (0, 2), 8),
    ("T6", (2, 6), 5),
    ("T7", (4, 6), 6),
    ("T8", (6, 4), 4),
]

DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))
INF = float('inf')


def _make_neighbors(grid: List[List[Any]], special_cells):
    """进入代价：X 不可通行；A/B/T 所在格为 1；数字格取数值。"""
    rows, cols = len(grid), len(grid[0])
    special = set(special_cells)

    def neighbors(u: Tuple[int, int]):
        r, c = u
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols:
                val = grid[nr][nc]
                if val == 'X':
                    continue
                v = (nr, nc)
                yield v, 1 if v in special else int(val)

    return neighbors


def build_extended_game(n_treasures: int) -> Tuple[Dict[str, Tuple[int, int]], Dict[str, int]]:
    """构造含 n_treasures 个宝藏的拓展局（n=4 时为原题）。"""
    locations = dict(LOCATIONS)
    values = dict(TREASURE_VALUES)
    for name, pos, val in EXTRA_TREASURES[:max(0, n_treasures - 4)]:
        locations[name] = pos
        values[name] = val
    return locations, values


# ---------------- 任务一：距离矩阵 ----------------

def compute_distance_matrix(
    locations: Optional[Dict[str, Tuple[int, int]]] = None,
    grid: Optional[List[List[Any]]] = None,
) -> Tuple[Dict[str, Dict[str, int]], int]:
    """用单源最短路计算全部关键点两两距离。

    原实现逐对调用 A* 共 6x6=36 次；单源搜索每个起点只需一次，
    调用次数降至 6 次，矩阵结果完全一致（最短路的正确性保证）。
    返回 (距离矩阵, 单源搜索调用次数)。
    """
    locations = dict(locations or LOCATIONS)
    grid = grid or GRID
    points = list(locations)
    neighbors = _make_neighbors(grid, locations.values())

    matrix: Dict[str, Dict[str, int]] = {}
    for p in points:
        dist = dijkstra(locations[p], neighbors)
        matrix[p] = {q: int(dist[locations[q]]) for q in points}
    return matrix, len(points)


def a_star_cost(start: Tuple[int, int], end: Tuple[int, int],
                grid: Optional[List[List[Any]]] = None) -> int:
    """单对最小移动代价（保留供单点查询，内部用单源最短路实现）。"""
    if start == end:
        return 0
    grid = grid or GRID
    neighbors = _make_neighbors(grid, LOCATIONS.values())
    return int(dijkstra(start, neighbors)[end])


# ---------------- 任务二：博弈状态 ----------------

class GameState:
    """
    博弈状态：
    - pos_A / pos_B: 双方当前位置（'A'/'B'/'T1'~'T8'）
    - remaining_treasures: 剩余宝藏元组
    - score_A / score_B: 累计净收益
    - current_player: 当前行动方
    - values: 宝藏价值表（拓展实验可替换）
    """

    def __init__(self, pos_A: str, pos_B: str, remaining_treasures: Tuple[str, ...],
                 score_A: int = 0, score_B: int = 0, current_player: str = "A",
                 values: Optional[Dict[str, int]] = None):
        self.pos_A = pos_A
        self.pos_B = pos_B
        self.remaining_treasures = remaining_treasures
        self.score_A = score_A
        self.score_B = score_B
        self.current_player = current_player
        self.values = values or TREASURE_VALUES

    def get_utility(self) -> int:
        """博弈评价值 U = Score_A - Score_B。"""
        return self.score_A - self.score_B

    def generate_next_state(self, target_treasure: str,
                            dist_matrix: Dict[str, Dict[str, int]]) -> 'GameState':
        """生成选取目标宝藏后的后继状态。"""
        rem = tuple(t for t in self.remaining_treasures if t != target_treasure)
        v = self.values[target_treasure]

        if self.current_player == "A":
            dist = dist_matrix[self.pos_A][target_treasure]
            return GameState(
                pos_A=target_treasure, pos_B=self.pos_B, remaining_treasures=rem,
                score_A=self.score_A + v - dist, score_B=self.score_B,
                current_player="B", values=self.values,
            )
        dist = dist_matrix[self.pos_B][target_treasure]
        return GameState(
            pos_A=self.pos_A, pos_B=target_treasure, remaining_treasures=rem,
            score_A=self.score_A, score_B=self.score_B + v - dist,
            current_player="A", values=self.values,
        )


# ---------------- 任务三：Minimax 与 Alpha-Beta ----------------

def minimax(state: GameState, depth: int, max_depth: int,
            dist_matrix: Dict[str, Dict[str, int]],
            stats: Dict[str, int]) -> Tuple[int, Optional[str]]:
    """纯 Minimax（无剪枝、无缓存）。"""
    stats["visited_nodes"] += 1
    if depth == max_depth or not state.remaining_treasures:
        return state.get_utility(), None

    best_action = None
    if state.current_player == "A":
        best_val = float('-inf')
        for t in state.remaining_treasures:
            val, _ = minimax(state.generate_next_state(t, dist_matrix),
                             depth + 1, max_depth, dist_matrix, stats)
            if val > best_val:
                best_val, best_action = val, t
        return best_val, best_action

    best_val = float('inf')
    for t in state.remaining_treasures:
        val, _ = minimax(state.generate_next_state(t, dist_matrix),
                         depth + 1, max_depth, dist_matrix, stats)
        if val < best_val:
            best_val, best_action = val, t
    return best_val, best_action


def alpha_beta(state: GameState, depth: int, max_depth: int, alpha: float, beta: float,
               dist_matrix: Dict[str, Dict[str, int]], stats: Dict[str, int],
               reverse_order: bool = False) -> Tuple[int, Optional[str]]:
    """Alpha-Beta 剪枝。reverse_order 用于对照不同子节点访问顺序。"""
    stats["visited_nodes"] += 1
    if depth == max_depth or not state.remaining_treasures:
        return state.get_utility(), None

    actions = list(state.remaining_treasures)
    if reverse_order:
        actions.reverse()

    best_action = None
    if state.current_player == "A":
        best_val = float('-inf')
        for i, t in enumerate(actions):
            val, _ = alpha_beta(state.generate_next_state(t, dist_matrix),
                                depth + 1, max_depth, alpha, beta, dist_matrix,
                                stats, reverse_order)
            if val > best_val:
                best_val, best_action = val, t
            alpha = max(alpha, best_val)
            if beta <= alpha:
                stats["pruned_branches"] += (len(actions) - 1 - i)
                break
        return best_val, best_action

    best_val = float('inf')
    for i, t in enumerate(actions):
        val, _ = alpha_beta(state.generate_next_state(t, dist_matrix),
                            depth + 1, max_depth, alpha, beta, dist_matrix,
                            stats, reverse_order)
        if val < best_val:
            best_val, best_action = val, t
        beta = min(beta, best_val)
        if beta <= alpha:
            stats["pruned_branches"] += (len(actions) - 1 - i)
            break
    return best_val, best_action


def minimax_cached(state: GameState, depth: int, max_depth: int,
                   dist_matrix: Dict[str, Dict[str, int]],
                   stats: Dict[str, int],
                   memo: Dict[Any, int]) -> int:
    """带置换表的 Minimax，返回“自当前局面起”的未来收益差。

    累计得分不进入缓存键：未来收益只取决于 (双方位置, 剩余宝藏, 行动方)，
    历史累计得分对该局面之后的决策没有影响，因此可以安全复用。
    """
    stats["visited_nodes"] += 1
    if depth == max_depth or not state.remaining_treasures:
        return 0

    key = (state.pos_A, state.pos_B, state.remaining_treasures, state.current_player)
    if key in memo:
        stats["cache_hits"] += 1
        return memo[key]

    best = None
    for t in state.remaining_treasures:
        nxt = state.generate_next_state(t, dist_matrix)
        delta = nxt.get_utility() - state.get_utility()
        val = delta + minimax_cached(nxt, depth + 1, max_depth, dist_matrix, stats, memo)
        if best is None:
            best = val
        elif state.current_player == "A":
            best = max(best, val)
        else:
            best = min(best, val)

    memo[key] = best
    return best


def _potential(state: GameState, dist_matrix: Dict[str, Dict[str, int]]) -> int:
    """浅层截断时用的启发式评估：按“谁更近”贪心分配剩余宝藏的潜在净收益。"""
    pot = 0
    for t in state.remaining_treasures:
        v = state.values[t]
        dA = dist_matrix[state.pos_A][t]
        dB = dist_matrix[state.pos_B][t]
        if dA <= dB:
            pot += max(0, v - dA)
        else:
            pot -= max(0, v - dB)
    return pot


def alpha_beta_eval(state: GameState, depth: int, max_depth: int, alpha: float, beta: float,
                    dist_matrix: Dict[str, Dict[str, int]], stats: Dict[str, int],
                    use_eval: bool = False) -> Tuple[int, Optional[str]]:
    """Alpha-Beta + 可选截断评估函数，用于验证“评估函数能否弥补视野不足”。"""
    stats["visited_nodes"] += 1
    if depth == max_depth or not state.remaining_treasures:
        base = state.get_utility()
        if use_eval and state.remaining_treasures:
            base += _potential(state, dist_matrix)
        return base, None

    best_action = None
    if state.current_player == "A":
        best_val = float('-inf')
        for t in state.remaining_treasures:
            val, _ = alpha_beta_eval(state.generate_next_state(t, dist_matrix),
                                     depth + 1, max_depth, alpha, beta, dist_matrix,
                                     stats, use_eval)
            if val > best_val:
                best_val, best_action = val, t
            alpha = max(alpha, best_val)
            if beta <= alpha:
                break
        return best_val, best_action

    best_val = float('inf')
    for t in state.remaining_treasures:
        val, _ = alpha_beta_eval(state.generate_next_state(t, dist_matrix),
                                 depth + 1, max_depth, alpha, beta, dist_matrix,
                                 stats, use_eval)
        if val < best_val:
            best_val, best_action = val, t
        beta = min(beta, best_val)
        if beta <= alpha:
            break
    return best_val, best_action


def get_principal_variation(state: GameState, max_depth: int,
                            dist_matrix: Dict[str, Dict[str, int]]) -> List[str]:
    """提取最优行动序列（主变例），用于分析双方决策过程。"""
    pv = []
    s, depth = state, 0
    while depth < max_depth and s.remaining_treasures:
        mover = s.current_player
        _, act = minimax(s, depth, max_depth, dist_matrix, {"visited_nodes": 0})
        if act is None:
            break
        s = s.generate_next_state(act, dist_matrix)
        pv.append(f"{mover} 取 {act} (U={s.get_utility()})")
        depth += 1
    return pv


def play_match(values: Dict[str, int], depth_A: int, depth_B: int,
               dist_matrix: Dict[str, Dict[str, int]]) -> int:
    """A、B 分别使用固定搜索深度完成一次完整对局，返回最终收益差 U。"""
    state = GameState("A", "B", tuple(sorted(values)), values=values)
    while state.remaining_treasures:
        limit = depth_A if state.current_player == "A" else depth_B
        _, act = alpha_beta(state, 0, limit, float('-inf'), float('inf'),
                            dist_matrix, {"visited_nodes": 0, "pruned_branches": 0})
        state = state.generate_next_state(act, dist_matrix)
    return state.get_utility()


def depth_match_experiment(trials: int = 200, seed: int = 7) -> Dict[str, Any]:
    """深度对抗：随机生成 trials 组宝藏价值，统计不同深度组合的平均收益差与胜率。"""
    rng = random.Random(seed)
    names = list(TREASURE_VALUES)
    # 2x2 对照：分别改变 A 的深度与 B 的深度，才能把两者的影响分离
    pairs = [(2, 2), (2, 4), (4, 2), (4, 4)]
    results = {f"A{dA}-B{dB}": [] for dA, dB in pairs}

    for _ in range(trials):
        values = {t: rng.randint(1, 9) for t in names}
        for dA, dB in pairs:
            u = play_match(values, dA, dB, DIST)
            results[f"A{dA}-B{dB}"].append(u)

    summary = {}
    for key, us in results.items():
        summary[key] = {
            "mean": sum(us) / len(us),
            "win_rate": sum(1 for u in us if u > 0) / len(us),
            "loss_rate": sum(1 for u in us if u < 0) / len(us),
            "draw_rate": sum(1 for u in us if u == 0) / len(us),
        }
    return summary


DIST, MATRIX_CALLS = compute_distance_matrix()


if __name__ == "__main__":
    print("=" * 60)
    print("题目四：竞争环境下的宝藏分配（综合题）运行输出")
    print("=" * 60)

    print(f"\n[1] 6x6 两两节点间最小移动代价矩阵 (单源搜索调用 {MATRIX_CALLS} 次，"
          f"初版逐对 A* 需 36 次):")
    headers = ["A", "B", "T1", "T2", "T3", "T4"]
    print(f"{'':>6}" + "".join([f"{h:>6}" for h in headers]))
    for u in headers:
        print(f"{u:>6}" + "".join([f"{DIST[u][v]:>6}" for v in headers]))

    initial_state = GameState("A", "B", ("T1", "T2", "T3", "T4"))

    for H in [2, 4]:
        print("\n" + "-" * 50)
        print(f"搜索深度 H = {H}")
        print("-" * 50)

        stats_mm = {"visited_nodes": 0}
        val_mm, act_mm = minimax(initial_state, 0, H, DIST, stats_mm)
        print(f"[Minimax]    => 最优首步: {act_mm}, 最优收益差: {val_mm}, "
              f"访问节点数: {stats_mm['visited_nodes']}")

        stats_ab = {"visited_nodes": 0, "pruned_branches": 0}
        val_ab, act_ab = alpha_beta(initial_state, 0, H, float('-inf'), float('inf'),
                                    DIST, stats_ab, reverse_order=False)
        print(f"[Alpha-Beta] => 最优首步: {act_ab}, 最优收益差: {val_ab}, "
              f"访问节点数: {stats_ab['visited_nodes']}, 剪枝次数: {stats_ab['pruned_branches']}")
        print(f"测试用例格式输出: {json.dumps({'best_target': act_ab, 'best_value': val_ab})}")

        pv = get_principal_variation(initial_state, H, DIST)
        print(f"最优行动序列: {' -> '.join(pv)}")

        stats_rev = {"visited_nodes": 0, "pruned_branches": 0}
        alpha_beta(initial_state, 0, H, float('-inf'), float('inf'), DIST,
                   stats_rev, reverse_order=True)
        print(f"[Alpha-Beta(逆序)] 访问节点数: {stats_rev['visited_nodes']}, "
              f"剪枝次数: {stats_rev['pruned_branches']}")

        reduction = (1 - stats_ab['visited_nodes'] / stats_mm['visited_nodes']) * 100
        print(f">> Alpha-Beta 搜索开销压缩率: {reduction:.2f}%")

    # ---------------- 拓展实验 1：置换表缓存随宝藏数量的收益 ----------------
    print("\n" + "=" * 60)
    print("[拓展 1] 置换表缓存随宝藏数量的收益（完整对局，深度=宝藏数）")
    print("=" * 60)
    print(f"{'宝藏数':<8}{'纯Minimax':>12}{'Minimax+缓存':>14}{'缓存命中':>10}{'节点降幅':>10}")
    for n in (4, 5, 6, 7):
        locs, vals = build_extended_game(n)
        d, _ = compute_distance_matrix(locs)
        st = GameState("A", "B", tuple(sorted(vals)), values=vals)

        s1 = {"visited_nodes": 0}
        v1, _ = minimax(st, 0, n, d, s1)

        s2 = {"visited_nodes": 0, "cache_hits": 0}
        v2 = minimax_cached(st, 0, n, d, s2, {})

        ok = (v1 == v2)
        drop = (1 - s2["visited_nodes"] / s1["visited_nodes"]) * 100
        print(f"{n:<8}{s1['visited_nodes']:>12}{s2['visited_nodes']:>14}"
              f"{s2['cache_hits']:>10}{drop:>9.1f}%   值一致={ok}")

    # ---------------- 拓展实验 2：搜索深度对抗 ----------------
    print("\n" + "=" * 60)
    print("[拓展 2] 深度对抗：不同搜索深度玩家的完整对局（200 组随机宝藏价值）")
    print("=" * 60)
    summary = depth_match_experiment(trials=200, seed=7)
    for key, s in summary.items():
        print(f"{key}: 平均 U = {s['mean']:+.3f}, A 胜率 = {s['win_rate']*100:.1f}%, "
              f"负率 = {s['loss_rate']*100:.1f}%, 平局 = {s['draw_rate']*100:.1f}%")

    # ---------------- 拓展实验 3：评估函数能否弥补视野不足 ----------------
    print("\n" + "=" * 60)
    print("[拓展 3] H=2 截断评估函数：数值估计是否改善、首步是否改变")
    print("=" * 60)
    exact_val, exact_act = alpha_beta(initial_state, 0, 4, float('-inf'), float('inf'),
                                      DIST, {"visited_nodes": 0, "pruned_branches": 0})
    v1, a1 = alpha_beta_eval(initial_state, 0, 2, float('-inf'), float('inf'), DIST,
                             {"visited_nodes": 0}, use_eval=False)
    v2, a2 = alpha_beta_eval(initial_state, 0, 2, float('-inf'), float('inf'), DIST,
                             {"visited_nodes": 0}, use_eval=True)
    print(f"完整博弈（H=4）真实值: 首步 {exact_act}, 值 {exact_val}")
    print(f"H=2 裸截断:            首步 {a1}, 值 {v1}")
    print(f"H=2 + 潜在收益评估:    首步 {a2}, 值 {v2}   "
          f"(评估把数值从 {v1} 修正到 {v2}，首步是否改变: {a1 != a2})")
