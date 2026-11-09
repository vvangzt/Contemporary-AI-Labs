"""实验二搜索算法公共核心。

题目一、二、四原本各有一份 A* 实现，逻辑几乎相同。这里收敛为一份实现：
调用方只提供 neighbors(u)（可通行邻接点及其进入代价）与 heuristic(u)，
即可复用同一套优先队列、闭集与路径回溯逻辑，减少重复代码与维护成本。

提供三个入口：
    astar()              通用 A*，可选“平局打破”节点排序优化
    dijkstra()           单源最短路，一次求出到多个目标的最优代价
    bidirectional_bfs()  双向搜索基线，仅用于等权图上的对照分析

统一约定：
    - 节点用 (row, col) 表示；
    - 代价为“进入该格”的代价值（与题目二、四的规定一致）；
    - 不可达时代价为 INF、路径为空列表。
"""

from collections import deque
from typing import Callable, Dict, Iterable, List, Tuple
import heapq

Node = Tuple[int, int]
Neighbors = Callable[[Node], Iterable[Tuple[Node, int]]]
Heuristic = Callable[[Node], float]

INF = float("inf")
DIRS = ((-1, 0), (1, 0), (0, -1), (0, 1))


def astar(
    start: Node,
    goal: Node,
    neighbors: Neighbors,
    heuristic: Heuristic,
    tie_break: bool = False,
) -> Tuple[float, List[Node], int]:
    """通用 A*，返回 (最小代价, 路径, 扩展节点数)。

    tie_break=True 时，f 值相同的节点优先扩展 g 更大者（即更接近目标的一侧）。
    该排序不改变最优性（h 可采纳且一致时），只改变同层节点的展开次序，
    典型均匀栅格上可显著减少扩展节点数。
    """
    if start == goal:
        return 0, [start], 1

    counter = 0
    open_heap: List[Tuple[float, int, int, Node]] = []
    heapq.heappush(open_heap, (heuristic(start), 0, counter, start))

    g_score: Dict[Node, float] = {start: 0}
    parent: Dict[Node, Node] = {}
    closed = set()

    while open_heap:
        _, _, _, u = heapq.heappop(open_heap)
        if u in closed:
            continue
        closed.add(u)

        if u == goal:
            path = [u]
            while path[-1] != start:
                path.append(parent[path[-1]])
            path.reverse()
            return g_score[u], path, len(closed)

        g_u = g_score[u]
        for v, step in neighbors(u):
            ng = g_u + step
            if ng < g_score.get(v, INF):
                g_score[v] = ng
                parent[v] = u
                counter += 1
                tie = -ng if tie_break else 0
                heapq.heappush(open_heap, (ng + heuristic(v), tie, counter, v))

    return INF, [], len(closed)


def dijkstra(start: Node, neighbors: Neighbors) -> Dict[Node, float]:
    """单源最短路（无启发）。

    题目四需要 6 个关键点两两之间的距离：逐对调用 A* 共 36 次，
    而单源搜索 6 次即可填满整个矩阵，调用次数降为原来的 1/6。
    """
    dist: Dict[Node, float] = {start: 0}
    pq: List[Tuple[float, Node]] = [(0, start)]

    while pq:
        d, u = heapq.heappop(pq)
        if d > dist.get(u, INF):
            continue
        for v, w in neighbors(u):
            nd = d + w
            if nd < dist.get(v, INF):
                dist[v] = nd
                heapq.heappush(pq, (nd, v))

    return dist


def bidirectional_bfs(
    start: Node, goal: Node, neighbors: Neighbors
) -> Tuple[float, int]:
    """双向分层 BFS（仅适用于每步代价相同的图）。

    作为对照基线使用：它不使用任何启发信息，两端各自向外扩展一层。
    返回 (最短步数, 扩展节点数)；不可达时返回 (INF, 扩展数)。
    """
    if start == goal:
        return 0, 1

    dist_f: Dict[Node, int] = {start: 0}
    dist_b: Dict[Node, int] = {goal: 0}
    layer_f: deque = deque([start])
    layer_b: deque = deque([goal])
    expanded = 0

    while layer_f and layer_b:
        if len(layer_f) <= len(layer_b):
            layer, dist_self, dist_other, is_forward = layer_f, dist_f, dist_b, True
        else:
            layer, dist_self, dist_other, is_forward = layer_b, dist_b, dist_f, False

        nxt: deque = deque()
        while layer:
            u = layer.popleft()
            expanded += 1
            for v, _ in neighbors(u):
                if v not in dist_self:
                    dist_self[v] = dist_self[u] + 1
                    nxt.append(v)
                if v in dist_other:
                    return dist_self[u] + 1 + dist_other[v], expanded

        if is_forward:
            layer_f = nxt
        else:
            layer_b = nxt

    return INF, expanded
