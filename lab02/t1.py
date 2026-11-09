"""题目一：A* 算法迷宫寻路。

实现采用公共核心 search_core.astar：
    - 状态：格点坐标 (row, col)
    - g：起点到当前格的步数（每步代价 1）
    - h：到终点的曼哈顿距离（use_heuristic=False 时 h=0，退化为 Dijkstra）
    - f = g + h，最小堆按 f 取优，闭集去重，parent 回溯路径
"""

from typing import List, Tuple

from search_core import DIRS, astar


def _make_neighbors(grid: List[List[str]]):
    """构造邻接函数：四方向移动，每步进入代价为 1，X 不可通行。"""
    rows, cols = len(grid), len(grid[0])

    def neighbors(u: Tuple[int, int]):
        r, c = u
        for dr, dc in DIRS:
            nr, nc = r + dr, c + dc
            if 0 <= nr < rows and 0 <= nc < cols and grid[nr][nc] != 'X':
                yield (nr, nc), 1

    return neighbors


def a_star_maze(
    start: Tuple[int, int],
    end: Tuple[int, int],
    grid: List[List[str]],
    use_heuristic: bool = True,
    tie_break: bool = False,
) -> Tuple[List[Tuple[int, int]], int]:
    """
    题目一：A* 算法迷宫寻路
    :param start: 起点坐标 (row, col)
    :param end: 终点坐标 (row, col)
    :param grid: 地图网格
    :param use_heuristic: True 用曼哈顿距离；False 用 h=0（Dijkstra，对照实验用）
    :param tie_break: True 时 f 相同优先扩展 g 更大者（节点排序优化，对照实验用）
    :return: (最短路径坐标列表, 扩展节点总数)；无解时返回 ([], 扩展数)
    """
    rows = len(grid)
    cols = len(grid[0])

    if not (0 <= start[0] < rows and 0 <= start[1] < cols):
        return [], 0
    if not (0 <= end[0] < rows and 0 <= end[1] < cols):
        return [], 0
    if grid[start[0]][start[1]] == 'X' or grid[end[0]][end[1]] == 'X':
        return [], 0

    def heuristic(u: Tuple[int, int]) -> int:
        if not use_heuristic:
            return 0
        return abs(u[0] - end[0]) + abs(u[1] - end[1])

    cost, path, expanded = astar(
        start, end, _make_neighbors(grid), heuristic, tie_break=tie_break
    )
    if cost == float('inf'):
        return [], expanded
    return path, expanded


def solve_maze(
    start: Tuple[int, int], end: Tuple[int, int], grid: List[List[str]]
) -> List[Tuple[int, int]]:
    """题目要求的标准输入接口：返回最短路径坐标列表。"""
    path, _ = a_star_maze(start, end, grid, use_heuristic=True)
    return path


if __name__ == "__main__":
    start_pos = (0, 0)
    end_pos = (3, 4)
    grid_map = [
        ['S', '.', '.', 'X', '.'],
        ['.', 'X', '.', '.', '.'],
        ['.', 'X', '.', 'X', '.'],
        ['.', '.', '.', 'X', 'E'],
        ['X', 'X', '.', '.', '.']
    ]

    path, nodes_a_star = a_star_maze(start_pos, end_pos, grid_map, use_heuristic=True)
    print("A* 寻路结果:")
    print("路径:", path)
    print("路径长度(步数):", len(path) - 1 if path else 0)
    print("A* 扩展节点数:", nodes_a_star)

    # 报告对照实验：h=0 等价于 Dijkstra
    _, nodes_dijkstra = a_star_maze(start_pos, end_pos, grid_map, use_heuristic=False)
    print("h=0 (Dijkstra) 扩展节点数:", nodes_dijkstra)

    # 报告对照实验：平局打破（f 相同优先 g 大者）在小图上的效果
    _, nodes_tie = a_star_maze(start_pos, end_pos, grid_map, tie_break=True)
    print("平局打破 扩展节点数:", nodes_tie)

    # 边界用例：起点被障碍完全封死（终点本身可通行），走的是"搜索耗尽"分支
    sealed = [
        ['S', 'X', '.'],
        ['X', 'X', '.'],
        ['.', '.', 'E'],
    ]
    print("无解用例检查 (起点被围死):", a_star_maze((0, 0), (2, 2), sealed))
    # 边界用例：终点落在障碍上，走的是参数检查分支
    print("无解用例检查 (终点为障碍):", a_star_maze((0, 0), (4, 0), grid_map))
