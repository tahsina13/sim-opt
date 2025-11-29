from typing import NamedTuple

from .dsu import DSU

__all__ = ["Edge", "MST"]


class Edge(NamedTuple):
    u: int
    v: int
    w: float


class MST:
    nodes: int
    edges: list[Edge]

    def __init__(self, nodes: int, edges: list[Edge]):
        self.nodes = nodes
        self.edges = sorted(edges, key=lambda e: e.w)

    def __call__(self) -> list[Edge]:
        dsu = DSU(self.nodes)
        mst_edges = []
        for e in self.edges:
            if dsu.find_set(e.u) != dsu.find_set(e.v):
                dsu.union_sets(e.u, e.v)
                mst_edges.append(e)
        return mst_edges
