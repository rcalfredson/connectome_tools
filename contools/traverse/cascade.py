import numpy as np
from anytree import Node
from anytree import LevelOrderGroupIter
from scipy.sparse import csr_matrix
from .traverse import BaseTraverse

# Global references to avoid copying large arrays in each process
GLOBAL_TRANSITION_PROBS = None
GLOBAL_NEG_INDS = None
GLOBAL_STOP_NODES = None
GLOBAL_MAX_HOPS = None
GLOBAL_RECORD_TRAVERSAL = None
GLOBAL_ALLOW_LOOPS = None


def worker_init(
    transition_probs_file,
    shape,
    neg_inds,
    stop_nodes,
    max_hops,
    record_traversal,
    allow_loops,
):
    """
    Initializer for worker processes.
    This sets global variables so that each worker references the same shared arrays.
    """
    global GLOBAL_TRANSITION_PROBS, GLOBAL_NEG_INDS
    global GLOBAL_STOP_NODES, GLOBAL_MAX_HOPS, GLOBAL_RECORD_TRAVERSAL, GLOBAL_ALLOW_LOOPS

    # Memory-map the transition probabilities file (read-only)
    GLOBAL_TRANSITION_PROBS = np.memmap(
        transition_probs_file, dtype=np.float32, mode="r", shape=shape
    )
    GLOBAL_NEG_INDS = neg_inds
    GLOBAL_STOP_NODES = stop_nodes
    GLOBAL_MAX_HOPS = max_hops
    GLOBAL_RECORD_TRAVERSAL = record_traversal
    GLOBAL_ALLOW_LOOPS = allow_loops


def to_transmission_matrix(
    sparse_matrix,
    p,
    method="uniform",
    memmap_file="transition_probs.dat",
    print_frequency=10000,
):
    """
    Convert a large sparse adjacency matrix into a transmission probability matrix incrementally,
    storing the result in a memory-mapped file. This is a scalable approach to accommodate very
    large matrices.

    Parameters
    ----------
    sparse_matrix : scipy.sparse.csr_matrix
        The sparse adjacency matrix of shape (N, N).
    p : float
        The probability parameter used to compute transmission probabilities.
    method : str, default="uniform"
        Currently only "uniform" is supported. If another method is passed, NotImplementedError is raised.
    memmap_file : str, default="transition_probs.dat"
        The path of the memory-mapped file to write the transmission probability matrix to.
    print_frequency : int, default=10000
        Frequency of status updates on processed rows.

    Returns
    -------
    np.memmap
        A memory-mapped array of shape (N, N) containing the transmission probabilities.
        Inhibitory rows will have negative probabilities as per the original logic.
    """
    if method != "uniform":
        raise NotImplementedError(
            "Currently only the 'uniform' method is supported in this sparse version."
        )

    # Ensure CSR format
    if not isinstance(sparse_matrix, csr_matrix):
        sparse_matrix = sparse_matrix.tocsr()

    shape = sparse_matrix.shape
    N = shape[0]

    # Determine which nodes are inhibitory by summing each row
    # Negative row sum indicates inhibitory node.
    print("Determining inhibitory nodes...")
    row_sums = np.zeros(N, dtype=np.float64)
    for i in range(N):
        row_start = sparse_matrix.indptr[i]
        row_end = sparse_matrix.indptr[i + 1]
        if row_end > row_start:
            row_sums[i] = np.sum(sparse_matrix.data[row_start:row_end])
        # If no entries, sum is 0 by default
    neg_inds = np.where(row_sums < 0)[0]
    neg_set = set(neg_inds)

    # Create a memmap for probabilities
    print(f"Creating memmap for probabilities at {memmap_file}, shape={shape}...")
    probs_memmap = np.memmap(memmap_file, dtype=np.float32, mode="w+", shape=shape)

    # Initialize the entire memmap to 0.0 (optional if desired)
    # This may be costly for very large arrays; you can skip this initialization.
    # probs_memmap[:] = 0.0
    # probs_memmap.flush()

    print("Computing transmission probabilities row by row...")
    for i in range(N):
        row_start = sparse_matrix.indptr[i]
        row_end = sparse_matrix.indptr[i + 1]

        # Set the row to zero first (if you didn't do a global initialization)
        probs_memmap[i, :] = 0.0

        if row_end > row_start:
            cols = sparse_matrix.indices[row_start:row_end]
            values = sparse_matrix.data[row_start:row_end]

            # Take absolute values for the probability calculation
            abs_values = np.abs(values)
            # Compute the probability that no synapse fires: (1 - p)^abs_values
            not_probs_values = np.power((1 - p), abs_values, dtype=np.float32)
            # Probability that at least one synapse fires: 1 - not_probs
            probs_values = 1.0 - not_probs_values

            # If node is inhibitory, negate the probabilities
            if i in neg_set:
                probs_values = -probs_values
                # Fix any -0.0 values to 0.0, as original code does
                probs_values[probs_values == -0.0] = 0.0

            # Store computed probabilities in the memmap
            probs_memmap[i, cols] = probs_values

        if (i % print_frequency) == 0 and i > 0:
            print(f"Processed {i} out of {N} rows")

    probs_memmap.flush()
    print("Transmission probability memmap created successfully.")
    return probs_memmap, neg_inds


class Cascade(BaseTraverse):
    def __init__(
        self,
        transition_probs=None,
        neg_inds=None,
        stop_nodes=None,
        max_hops=10,
        hit_hist=None,
        record_traversal=True,
        allow_loops=True,
        start_node_persistence=1,
    ):
        tp = (
            GLOBAL_TRANSITION_PROBS
            if GLOBAL_TRANSITION_PROBS is not None
            else transition_probs
        )
        st = GLOBAL_STOP_NODES if GLOBAL_STOP_NODES is not None else stop_nodes
        mh = GLOBAL_MAX_HOPS if GLOBAL_MAX_HOPS is not None else max_hops
        rt = (
            GLOBAL_RECORD_TRAVERSAL
            if GLOBAL_RECORD_TRAVERSAL is not None
            else record_traversal
        )
        al = GLOBAL_ALLOW_LOOPS if GLOBAL_ALLOW_LOOPS is not None else allow_loops
        ni = GLOBAL_NEG_INDS if GLOBAL_NEG_INDS is not None else neg_inds
        super().__init__(
            transition_probs=tp,
            stop_nodes=st if st is not None else [],
            max_hops=mh,
            hit_hist=hit_hist,
            record_traversal=rt,
            allow_loops=al,
        )
        self.neg_inds = ni
        self.start_node_persistence = start_node_persistence
        self._initial_active = None
        self._initial_active_duration = 0

    def _choose_next(self):
        # Identify active inhibitory nodes
        all_neg_inds = self.neg_inds
        neg_inds_active = np.intersect1d(self._active, all_neg_inds)
        # Identify active excitatory nodes
        active_excitatory = np.setdiff1d(self._active, all_neg_inds)

        # If no active excitatory nodes, nothing can propagate.
        if len(active_excitatory) == 0:
            return None

        # We'll build node_transition_probs only for the active excitatory subset.
        # This avoids loading the entire NxN array.
        n_targets = self.transition_probs.shape[1]
        node_transition_probs = np.zeros(
            (len(active_excitatory), n_targets), dtype=self.transition_probs.dtype
        )

        # Load excitatory rows from memmap
        # This is a subset of rows, hopefully small
        for i, node_idx in enumerate(active_excitatory):
            node_transition_probs[i, :] = self.transition_probs[node_idx, :]

        # If inhibitory nodes are active, sum up their rows to get the inhibitory influence.
        if len(neg_inds_active) > 0:
            summed_neg = np.zeros(n_targets, dtype=self.transition_probs.dtype)
            for neg_node in neg_inds_active:
                # Add this inhibitory node's row to summed_neg
                summed_neg += self.transition_probs[neg_node, :]

            # Apply the inhibitory influence to all active excitatory rows
            node_transition_probs += summed_neg

        # Clip negative probabilities to zero
        # Now node_transition_probs corresponds only to excitatory rows
        node_transition_probs[node_transition_probs < 0] = 0

        # Probabilistic transmission sampling
        # Generate a binomial random draw for each edge from the active excitatory nodes
        transmission_indicator = np.random.binomial(n=1, p=node_transition_probs)

        # Find which nodes received a signal
        # transmission_indicator is shape (len(active_excitatory), n_targets)
        # We want columns where there's a '1' from any row
        nxt = np.unique(np.nonzero(transmission_indicator)[1])

        if len(nxt) > 0:
            return nxt
        else:
            return None

    def start(self, start_node):
        if isinstance(start_node, int):
            start_node = np.array([start_node])
        else:
            start_node = np.array(start_node)
        super().start(start_node)
        self._initial_active = start_node
        self._initial_active_duration = self.start_node_persistence

    def _post_process_active_nodes(self, nxt):
        # If we still have duration left for the initial active nodes,
        # keep them active
        if self._initial_active_duration > 1:
            if nxt is None:
                nxt = self._initial_active
            else:
                nxt = np.unique(np.concatenate((nxt, self._initial_active)))

        # Decrement the duration after applying this step
        if self._initial_active_duration > 0:
            self._initial_active_duration -= 1

        return nxt

    def _check_visited(self):
        self._active = np.setdiff1d(self._active, self._visited)
        return len(self._active) > 0

    def _check_stop_nodes(self):
        self._active = np.setdiff1d(self._active, self.stop_nodes)
        return len(self._active) > 0


def generate_cascade_paths(
    start_ind,
    all_start_inds,
    probs,
    depth,
    stop_inds=[],
    visited=[],
    max_depth=10,  # added all_start_inds
):
    visited = visited.copy()
    visited.append(start_ind)

    probs = probs.copy()
    neg_inds = np.where(probs.sum(axis=1) < 0)[0]  # identify nodes with negative edges

    if (
        (depth < max_depth)
        and (start_ind not in stop_inds)
        and (start_ind not in neg_inds)
    ):  # transmission not allowed through negative edges

        neg_inds_active = np.intersect1d(
            all_start_inds, neg_inds
        )  # identify active inhibitory nodes
        if len(neg_inds_active) > 0:

            # sum all activate negative edges if multiple activate negative nodes
            if len(np.shape(probs[neg_inds])) > 1:
                summed_neg = probs[neg_inds_active].sum(axis=0)
                probs[start_ind] = (
                    probs[start_ind] + summed_neg
                )  # reduce probability of activating positive edges by magnitude of sum of negative edges

            # if only one activate negative node
            else:
                probs[start_ind] = (
                    probs[start_ind] + probs[neg_inds_active]
                )  # reduce probability of activating positive edges by magnitude of negative edges

        # where the probabilistic signal transmission occurs
        # probs must be positive, so all negative values are converted to zero
        probs[probs < 0] = 0

        transmission_indicator = np.random.binomial(
            np.ones(len(probs), dtype=int), probs[start_ind]
        )
        next_inds = np.where(transmission_indicator == 1)[0]
        paths = []
        for i in next_inds:
            if i not in visited:
                next_paths = generate_cascade_paths(
                    i,
                    next_inds,
                    probs,
                    depth + 1,
                    stop_inds=stop_inds,
                    visited=visited,
                    max_depth=max_depth,
                )
                for p in next_paths:
                    paths.append(p)
        return paths
    else:
        return [visited]


def generate_cascade_tree(
    node, probs, depth, stop_inds=[], visited=[], max_depth=10, loops=False
):
    start_ind = node.name
    if (depth < max_depth) and (start_ind not in stop_inds):
        transmission_indicator = np.random.binomial(
            np.ones(len(probs), dtype=int), probs[start_ind]
        )
        next_inds = np.where(transmission_indicator == 1)[0]
        for n in next_inds:
            if n not in visited:
                next_node = Node(n, parent=node)
                visited.append(n)
                generate_cascade_tree(
                    next_node,
                    probs,
                    depth + 1,
                    stop_inds=stop_inds,
                    visited=visited,
                    max_depth=max_depth,
                )
    return node


def cascades_from_node(
    start_ind,
    probs,
    stop_inds=[],
    max_depth=10,
    n_sims=1000,
    seed=None,
    n_bins=None,
    method="tree",
):
    if n_bins is None:
        n_bins = max_depth
    np.random.seed(seed)
    n_verts = len(probs)
    node_hist_mat = np.zeros((n_verts, n_bins), dtype=int)
    for n in range(n_sims):
        if method == "tree":
            _cascade_tree_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat)
        elif method == "path":
            _cascade_path_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat)
    return node_hist_mat


def _cascade_tree_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat):
    root = Node(start_ind)
    root = generate_cascade_tree(
        root, probs, 1, stop_inds=stop_inds, visited=[], max_depth=max_depth
    )
    for level, children in enumerate(LevelOrderGroupIter(root)):
        for node in children:
            node_hist_mat[node.name, level] += 1


def _cascade_path_helper(start_ind, probs, stop_inds, max_depth, node_hist_mat):
    paths = generate_cascade_paths(
        start_ind, probs, 1, stop_inds=stop_inds, max_depth=max_depth
    )
    for path in paths:
        for i, node in enumerate(path):
            node_hist_mat[node, i] += 1
    return paths
