"""基准测试：为实验报告生成规模化对照数据与图表。

运行：python bench.py
输出：
    figs/fig_astar_scaling.png   A* 扩展节点数随地图规模的变化
    figs/fig_game_scaling.png    博弈搜索节点数随宝藏数量的变化
    results_astar.csv            题目一/二规模实验原始数据
    results_game.csv             题目四规模实验原始数据

所有随机数据固定种子，重复运行结果一致。
"""

import csv
import os
import random
import statistics
import time
from typing import Dict, List, Tuple

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

from search_core import DIRS, astar, bidirectional_bfs
import t4

# 图表使用的中文字体（Windows 自带，缺失时回退到默认西文字体）
plt.rcParams["font.sans-serif"] = ["Microsoft YaHei", "SimHei", "DejaVu Sans"]
plt.rcParams["axes.unicode_minus"] = False

OUT_DIR = os.path.dirname(os.path.abspath(__file__))
FIG_DIR = os.path.join(OUT_DIR, "figs")

# 配色：低饱和、打印友好
C_BASE = "#7C8FA6"
C_TIE = "#C08552"
C_DIJK = "#A8B5A2"
C_BI = "#B5838D"


# ---------------- 通用工具 ----------------

def make_neighbors(grid: List[List[str]]):
    rows, cols = len(grid), len(grid[0])

    def neighbors(u: Tuple[int, int]):
        r, c = u
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 'X':
                yield (nr, nc), 1

    return neighbors


def random_grid(n: int, obstacle: float, seed: int) -> List[List[str]]:
    """生成 n x n 随机障碍图，保证起点与终点连通。"""
    rng = random.Random(seed)
    while True:
        grid = [['X' if rng.random() < obstacle else '.' for _ in range(n)] for _ in range(n)]
        grid[0][0], grid[n - 1][n - 1] = 'S', 'E'
        seen, stack = {(0, 0)}, [(0, 0)]
        while stack:
            r, c = stack.pop()
            for dr, dc in DIRS:
                nr, nc = r + dr, c + dc
                if 0 <= nr < n and 0 <= nc < n and grid[nr][nc] != 'X' and (nr, nc) not in seen:
                    seen.add((nr, nc))
                    stack.append((nr, nc))
        if (n - 1, n - 1) in seen:
            return grid


def run_timed(fn, repeat: int = 3) -> Tuple[float, float]:
    """返回 (结果, 最快一次耗时毫秒)。节点数确定，取最快时间以减小噪声。"""
    best = float("inf")
    result = None
    for _ in range(repeat):
        t0 = time.perf_counter()
        result = fn()
        best = min(best, (time.perf_counter() - t0) * 1000)
    return result, best


# ---------------- 实验一：A* 规模曲线 ----------------

def experiment_astar(sizes=(11, 21, 31, 41, 51), seeds=(0, 1, 2), obstacle=0.25):
    rows = []
    for n in sizes:
        for seed in seeds:
            grid = random_grid(n, obstacle, seed)
            start, goal = (0, 0), (n - 1, n - 1)
            nb = make_neighbors(grid)

            def h(u, goal=goal):
                return abs(u[0] - goal[0]) + abs(u[1] - goal[1])

            (cost_base, _, n_base), t_base = run_timed(
                lambda: astar(start, goal, nb, h, tie_break=False))
            (cost_tie, _, n_tie), t_tie = run_timed(
                lambda: astar(start, goal, nb, h, tie_break=True))
            (cost_dij, _, n_dij), t_dij = run_timed(
                lambda: astar(start, goal, nb, lambda u: 0))
            (cost_bi, n_bi), t_bi = run_timed(
                lambda: bidirectional_bfs(start, goal, nb))

            assert cost_base == cost_tie == cost_dij == cost_bi, "各变体代价必须一致"
            rows.append({
                "size": n, "seed": seed, "cost": int(cost_base),
                "nodes_baseline": n_base, "nodes_tiebreak": n_tie,
                "nodes_dijkstra": n_dij, "nodes_bidir": n_bi,
                "ms_baseline": t_base, "ms_tiebreak": t_tie,
                "ms_dijkstra": t_dij, "ms_bidir": t_bi,
            })
    return rows


# ---------------- 实验二：博弈搜索规模曲线 ----------------

def experiment_game(n_treasures=(4, 5, 6, 7)):
    rows = []
    for n in n_treasures:
        locs, vals = t4.build_extended_game(n)
        dist, calls = t4.compute_distance_matrix(locs)
        state = t4.GameState("A", "B", tuple(sorted(vals)), values=vals)

        s_mm = {"visited_nodes": 0}
        t0 = time.perf_counter()
        v_mm, _ = t4.minimax(state, 0, n, dist, s_mm)
        ms_mm = (time.perf_counter() - t0) * 1000

        s_ca = {"visited_nodes": 0, "cache_hits": 0}
        t0 = time.perf_counter()
        v_ca = t4.minimax_cached(state, 0, n, dist, s_ca, {})
        ms_ca = (time.perf_counter() - t0) * 1000

        s_ab = {"visited_nodes": 0, "pruned_branches": 0}
        t0 = time.perf_counter()
        v_ab, _ = t4.alpha_beta(state, 0, n, float("-inf"), float("inf"),
                                dist, s_ab)
        ms_ab = (time.perf_counter() - t0) * 1000

        assert v_mm == v_ab == v_ca, "各算法博弈值必须一致"
        rows.append({
            "treasures": n, "value": v_mm, "matrix_calls": calls,
            "nodes_minimax": s_mm["visited_nodes"],
            "nodes_cached": s_ca["visited_nodes"],
            "cache_hits": s_ca["cache_hits"],
            "nodes_alphabeta": s_ab["visited_nodes"],
            "pruned": s_ab["pruned_branches"],
            "ms_minimax": ms_mm, "ms_cached": ms_ca, "ms_alphabeta": ms_ab,
        })
    return rows


# ---------------- 图表 ----------------

def plot_astar(rows):
    sizes = sorted({r["size"] for r in rows})
    series = {
        "Dijkstra (h=0)": ("nodes_dijkstra", C_DIJK, "o"),
        "A* (曼哈顿)": ("nodes_baseline", C_BASE, "s"),
        "A* + 平局打破": ("nodes_tiebreak", C_TIE, "^"),
        "双向 BFS": ("nodes_bidir", C_BI, "d"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for label, (key, color, marker) in series.items():
        ys = [statistics.mean(r[key] for r in rows if r["size"] == s) for s in sizes]
        ax.plot(sizes, ys, marker=marker, color=color, linewidth=1.6,
                markersize=4.5, label=label)

    ax.set_yscale("log")
    ax.set_xlabel("地图边长 n（n×n，25% 障碍）", fontsize=9)
    ax.set_ylabel("扩展节点数（对数刻度）", fontsize=9)
    ax.set_title("")
    ax.grid(True, which="both", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=7.5, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_astar_scaling.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


def plot_game(rows):
    ns = [r["treasures"] for r in rows]
    width = 0.26
    series = [
        ("纯 Minimax", "nodes_minimax", C_BASE),
        ("Minimax + 置换表", "nodes_cached", C_TIE),
        ("Alpha-Beta", "nodes_alphabeta", C_DIJK),
    ]
    fig, ax = plt.subplots(figsize=(6.4, 3.5))
    for i, (label, key, color) in enumerate(series):
        xs = [n + (i - 1) * width for n in ns]
        ys = [r[key] for r in rows]
        bars = ax.bar(xs, ys, width=width, label=label, color=color,
                      edgecolor="white", linewidth=0.6)
        for b, y in zip(bars, ys):
            ax.annotate(f"{y}", (b.get_x() + b.get_width() / 2, y),
                        textcoords="offset points", xytext=(0, 2),
                        ha="center", fontsize=6.5)

    ax.set_yscale("log")
    ax.set_xticks(ns)
    ax.set_xlabel("宝藏数量", fontsize=9)
    ax.set_ylabel("访问节点数（对数刻度）", fontsize=9)
    ax.set_title("")
    ax.grid(True, axis="y", linestyle="--", linewidth=0.4, alpha=0.4)
    ax.legend(fontsize=7.5, frameon=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.tick_params(labelsize=8)
    fig.tight_layout()
    path = os.path.join(FIG_DIR, "fig_game_scaling.png")
    fig.savefig(path, dpi=170)
    plt.close(fig)
    return path


# ---------------- 主流程 ----------------

def main():
    os.makedirs(FIG_DIR, exist_ok=True)

    print("=" * 66)
    print("基准一：A* 随地图规模的变化（每档 3 张随机图取均值）")
    print("=" * 66)
    astar_rows = experiment_astar()
    sizes = sorted({r["size"] for r in astar_rows})
    print(f"{'边长':>5}{'代价':>6}{'Dijkstra':>10}{'A*':>8}{'A*+平局':>10}{'双向BFS':>9}{'A*耗时ms':>10}")
    for s in sizes:
        sub = [r for r in astar_rows if r["size"] == s]
        print(f"{s:>5}{sub[0]['cost']:>6}"
              f"{statistics.mean(r['nodes_dijkstra'] for r in sub):>10.0f}"
              f"{statistics.mean(r['nodes_baseline'] for r in sub):>8.0f}"
              f"{statistics.mean(r['nodes_tiebreak'] for r in sub):>10.0f}"
              f"{statistics.mean(r['nodes_bidir'] for r in sub):>9.0f}"
              f"{statistics.mean(r['ms_baseline'] for r in sub):>10.2f}")

    base_51 = statistics.mean(r['nodes_baseline'] for r in astar_rows if r['size'] == 51)
    tie_51 = statistics.mean(r['nodes_tiebreak'] for r in astar_rows if r['size'] == 51)
    bi_51 = statistics.mean(r['nodes_bidir'] for r in astar_rows if r['size'] == 51)
    print(f"\n51x51: 平局打破较基线下降 {(1 - tie_51 / base_51) * 100:.1f}%；"
          f"双向 BFS 为基线的 {bi_51 / base_51:.2f} 倍（反而更差）")

    print("\n" + "=" * 66)
    print("基准二：博弈搜索随宝藏数量的变化（完整对局，深度=宝藏数）")
    print("=" * 66)
    game_rows = experiment_game()
    print(f"{'宝藏数':>6}{'博弈值':>8}{'Minimax':>10}{'+置换表':>9}{'缓存命中':>9}{'Alpha-Beta':>12}{'剪枝':>7}")
    for r in game_rows:
        print(f"{r['treasures']:>6}{r['value']:>8}{r['nodes_minimax']:>10}"
              f"{r['nodes_cached']:>9}{r['cache_hits']:>9}"
              f"{r['nodes_alphabeta']:>12}{r['pruned']:>7}")

    # CSV
    for name, rows in (("results_astar.csv", astar_rows), ("results_game.csv", game_rows)):
        with open(os.path.join(OUT_DIR, name), "w", newline="", encoding="utf-8-sig") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"已写出 {name}")

    p1 = plot_astar(astar_rows)
    p2 = plot_game(game_rows)
    print(f"已生成图表：\n  {p1}\n  {p2}")

    # 深度对抗（2x2）与评估函数实验，直接复用 t4 的实现
    print("\n" + "=" * 66)
    print("基准三：搜索深度 2x2 对抗（200 组随机宝藏价值）")
    print("=" * 66)
    summary = t4.depth_match_experiment(trials=200, seed=7)
    for key, s in summary.items():
        print(f"{key}: 平均 U = {s['mean']:+.3f}, A 胜率 = {s['win_rate'] * 100:.1f}%")


if __name__ == "__main__":
    main()
