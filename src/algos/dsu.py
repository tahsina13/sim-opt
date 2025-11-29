__all__ = ["DSU"]


class DSU:
    parent: list[int]
    size: list[int]

    def __init__(self, nodes: int):
        self.parent = list(range(nodes))
        self.size = [1] * nodes

    def find_set(self, v: int) -> int:
        if v == self.parent[v]:
            return v
        self.parent[v] = self.find_set(self.parent[v])
        return self.parent[v]

    def union_sets(self, a: int, b: int):
        a, b = self.find_set(a), self.find_set(b)
        if a != b:
            if self.size[a] < self.size[b]:
                a, b = b, a
            self.parent[b] = a
            self.size[a] += self.size[b]
