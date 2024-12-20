from tqdm import tqdm
from multiprocessing import Pool
import numpy as np
import os
import itertools
import networkx as nx
from .cascade import Cascade, worker_init


def _run_single_simulation(start_node):
    # This function is executed in each worker.
    # Cascade now uses the global data set by worker_init.
    worker = Cascade(
        # stop_nodes, max_hops, etc. are taken from globals inside Cascade __init__
        hit_hist=None,
        start_node_persistence=1,
    )
    hit_hist = np.zeros((worker.n_verts, worker.max_hops), dtype=int)
    worker.start(start_node)
    traversal = worker.traversal_
    for level, nodes in enumerate(traversal):
        hit_hist[nodes, level] += 1
    return hit_hist


def path_to_visits(paths, n_verts, from_order=True, out_inds=[]):
    visit_orders = {i: [] for i in range(n_verts)}
    for path in paths:
        for i, n in enumerate(path):
            if from_order:
                visit_orders[n].append(i + 1)
            else:
                visit_orders[n].append(len(path) - i)
    return visit_orders


def to_path_graph(paths):
    path_graph = nx.MultiDiGraph()

    all_nodes = list(itertools.chain.from_iterable(paths))
    all_nodes = np.unique(all_nodes)
    path_graph.add_nodes_from(all_nodes)

    for path in paths:
        path_graph.add_edges_from(nx.utils.pairwise(path))

    path_graph = collapse_multigraph(path_graph)
    return path_graph


def collapse_multigraph(multigraph):
    """REF : https://stackoverflow.com/questions/15590812/networkx-convert-multigraph-...
        into-simple-graph-with-weighted-edges

    Parameters
    ----------
    multigraph : [type]
        [description]
    """
    G = nx.DiGraph()
    for u, v, data in multigraph.edges(data=True):
        w = data["weight"] if "weight" in data else 1.0
        if G.has_edge(u, v):
            G[u][v]["weight"] += w
        else:
            G.add_edge(u, v, weight=w)
    return G


# Maybe this class is unnecessary?
class TraverseDispatcher:
    def __init__(
        self, worker, *args, n_init=10, verbose=False, simultaneous=True, **kwargs
    ):
        # Store arguments for use later
        self._worker_class = worker
        self.n_init = n_init
        self.simultaneous = simultaneous
        self.verbose = verbose
        # Create a worker instance just to derive dimensions, etc.
        self._worker = worker(*args, **kwargs)
        # Store them for passing to workers:
        self.transition_probs = self._worker.transition_probs
        self.neg_inds = (
            self._worker.neg_inds if hasattr(self._worker, "neg_inds") else None
        )
        self.stop_nodes = self._worker.stop_nodes
        self.max_hops = self._worker.max_hops
        self.record_traversal = self._worker.record_traversal
        self.allow_loops = self._worker.allow_loops
        self.start_node_persistence = self._worker.start_node_persistence

    def start(self, start_node, disable):
        worker = self._worker
        n_verts = worker.n_verts
        max_hops = worker.max_hops

        if not isinstance(self.transition_probs, np.memmap):
            raise ValueError("transition_probs must be a memmap for sharing.")

        transition_probs_file = self.transition_probs.filename
        shape = self.transition_probs.shape

        # Prepare arguments for each simulation
        args_iter = [(node,) for node in [start_node] * self.n_init]

        # Initialize pool
        with Pool(
            processes=os.cpu_count(),
            initializer=worker_init,
            initargs=(
                transition_probs_file,
                shape,
                self.neg_inds,
                self.stop_nodes,
                self.max_hops,
                self.record_traversal,
                self.allow_loops,
                self.start_node_persistence
            ),
        ) as pool:
            hit_hist = np.zeros((n_verts, max_hops), dtype=int)
            for res in tqdm(
                pool.imap_unordered(_run_single_simulation, args_iter),
                total=self.n_init,
                disable=disable,
            ):
                hit_hist += res

        self.hit_hist_ = hit_hist
        return hit_hist

    def multistart(self, start_nodes, disable):
        if self.simultaneous:
            hop_hist = self.start(start_nodes, disable=disable)
        else:
            n_verts = len(self._worker.transition_probs)
            hop_hist = np.zeros((n_verts, self._worker.max_hops))
            for s in start_nodes:
                hop_hist += self.start(s, disable=disable)
        return hop_hist
