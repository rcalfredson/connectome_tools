# contools/traverse/base_traverse.py
import numpy as np


class BaseTraverse:
    def __init__(
        self,
        transition_probs,
        stop_nodes=[],
        max_hops=10,
        hit_hist=None,
        record_traversal=True,
        allow_loops=True,
    ):
        self.transition_probs = transition_probs
        self.hit_hist = hit_hist
        self.record_traversal = record_traversal
        self.max_hops = max_hops
        self.stop_nodes = stop_nodes
        self.allow_loops = allow_loops
        self.n_verts = len(transition_probs)
        if record_traversal:
            self.traversal_ = None
        if not allow_loops:
            self._visited = None

    def _check_max_hops(self):
        return not self._hop >= self.max_hops  # do not continue if greater than

    def _check_stop_nodes(self):
        return self._active not in self.stop_nodes

    def _check_visited(self):
        if not self.allow_loops:
            return self._active not in self._visited
        else:
            return True

    def _check_stop_conditions(self):
        check_items = [self._check_max_hops(), self._check_visited()]
        return all(check_items)

    def _reset(self):
        self._hop = 0
        self._active = None
        self._visited = np.array([])
        if self.record_traversal:
            self.traversal_ = []

    def _update_state(self, nxt):
        if nxt is not None:
            self._active = nxt
            self._hop += 1
            self.traversal_.append(nxt)
            if not self.allow_loops:
                self._visited = np.union1d(self._visited, nxt)
            return True
        else:
            return False

    def _step(self):
        if self._check_stop_conditions():
            self._update_state(self._active)
            if self._check_stop_nodes():
                nxt = self._choose_next()
                nxt = self._post_process_active_nodes(nxt)
                self._active = nxt
            else:
                nxt = None
            return nxt

    def _post_process_active_nodes(self, nxt):
        return nxt

    def start(self, start_node):
        self._reset()
        self._active = start_node
        nxt = start_node
        while nxt is not None:
            nxt = self._step()
