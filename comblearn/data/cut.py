import networkx as nx
import torch
import torch.nn as nn


class Graph:
    def __init__(self, n, p):
        self.graph = nx.erdos_renyi_graph(n, p)

    def get_cut_size(self, S: torch.tensor):
        assert len(S.size()) == 2
        output = []
        for _, x in enumerate(S):
            s = x.nonzero().squeeze().tolist()
            if isinstance(s, int):
                s = [s]
            output.append(nx.cut_size(self.graph, set(s)) + sum([self.graph.degree(a) for a in s]))
        return torch.tensor(output)