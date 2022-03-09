# object for analysing hit_histograms from cascades run using TraverseDispatcher
import numpy as np
import pandas as pd
from contools.process_matrix import Promat
from contools.traverse import Cascade, to_transmission_matrix, TraverseDispatcher

#from contools.traverse import Cascade, to_transmission_matrix
#from contools.traverse import TraverseDispatcher

from joblib import Parallel, delayed
from tqdm import tqdm

class Cascade_Analyzer:
    def __init__(self, name, hit_hist, n_init, skids_in_hit_hist=True, adj_index=None): # changed mg to adj_index for custom/modified adj matrices
        self.hit_hist = hit_hist
        self.name = name
        self.n_init = n_init
        if(skids_in_hit_hist):
            self.adj_index = hit_hist.index
            self.skid_hit_hist = hit_hist
        if(skids_in_hit_hist==False):
            self.adj_index = adj_index
            self.skid_hit_hist = pd.DataFrame(hit_hist, index = self.adj_index) # convert indices to skids

    def get_hit_hist(self):
        return(self.hit_hist)

    def get_skid_hit_hist(self):
        return(self.skid_hit_hist)

    def get_name(self):
        return(self.name)

    def index_to_skid(self, index):
        return(self.adj_index[index].name)

    def skid_to_index(self, skid):
        index_match = np.where(self.adj_index == skid)[0]
        if(len(index_match)==1):
            return(int(index_match[0]))
        if(len(index_match)!=1):
            print(f'Not one match for skid {skid}!')
            return(False)

    def pairwise_threshold_detail(self, threshold, hops, pairs_path, excluded_skids=False, include_source=False):

        if(include_source):
            neurons = np.where((self.skid_hit_hist.iloc[:, 0:(hops+1)]).sum(axis=1)>threshold)[0]
        if(include_source==False):
            neurons = np.where((self.skid_hit_hist.iloc[:, 1:(hops+1)]).sum(axis=1)>threshold)[0]

        neurons = self.skid_hit_hist.index[neurons]

        # remove particular skids if included
        if(excluded_skids!=False): 
            neurons = np.delete(neurons, excluded_skids)

        neurons_pairs, neurons_unpaired, neurons_nonpaired = Promat.extract_pairs_from_list(neurons, Promat.get_pairs(pairs_path))
        return(neurons_pairs, neurons_unpaired, neurons_nonpaired)

    def pairwise_threshold(self, threshold, hops, pairs_path, excluded_skids=False, include_source=False):
        neurons_pairs, neurons_unpaired, neurons_nonpaired = Cascade_Analyzer.pairwise_threshold_detail(self, threshold, hops, pairs_path, excluded_skids=excluded_skids, include_source=include_source)
        skids = np.concatenate([neurons_pairs.leftid, neurons_pairs.rightid, neurons_nonpaired.nonpaired])
        return(skids)

    def cascades_in_celltypes(self, cta, hops, start_hop=1, normalize='visits', pre_counts = None):
        skid_hit_hist = self.skid_hit_hist
        n_init = self.n_init
        hits = []
        for celltype in cta.Celltypes:
            total = skid_hit_hist.loc[np.intersect1d(celltype.get_skids(), skid_hit_hist.index), :].sum(axis=0).iloc[start_hop:hops+1].sum()
            if(normalize=='visits'): total = total/(len(celltype.get_skids())*n_init)
            if(normalize=='skids'): total = total/(len(celltype.get_skids()))
            if(normalize=='pre-skids'): total = total/pre_counts
            hits.append([celltype.get_name(), total])

        data = pd.DataFrame(hits, columns=['neuropil', 'visits_norm'])
        return(data)

    def cascades_in_celltypes_hops(self, cta, hops=None, start_hop=0, normalize='visits', pre_counts = None):

        if(hops==None): hops = len(self.skid_hit_hist.columns)

        skid_hit_hist = self.skid_hit_hist
        n_init = self.n_init
        hits = []
        for celltype in cta.Celltypes:
            total = skid_hit_hist.loc[np.intersect1d(celltype.get_skids(), skid_hit_hist.index), :].sum(axis=0).iloc[start_hop:hops]
            if(normalize=='visits'): total = total/(len(celltype.get_skids())*n_init)
            if(normalize=='skids'): total = total/(len(celltype.get_skids()))
            if(normalize=='pre-skids'): total = total/pre_counts
            hits.append(total)

        data = pd.concat(hits, axis=1)
        return(data)

    @staticmethod
    def run_cascade(i, cdispatch, indicator=None):

        # used in run_cascades_parallel() to give a bit more feedback
        if(indicator!=None):
            print(indicator)
        return(cdispatch.multistart(start_nodes = i))

    @staticmethod
    def run_cascades_parallel(source_skids_list, source_names, stop_skids, adj, p, max_hops, n_init, simultaneous):
        # adj format must be pd.DataFrame with skids for index/columns

        source_indices_list = []
        for skids in source_skids_list:
            indices = np.where([x in skids for x in adj.index])[0]
            source_indices_list.append(indices)

        stop_indices = np.where([x in stop_skids for x in adj.index])[0]

        transition_probs = to_transmission_matrix(adj.values, p)

        cdispatch = TraverseDispatcher(
            Cascade,
            transition_probs,
            stop_nodes = stop_indices,
            max_hops=max_hops+1, # +1 because max_hops includes hop 0
            allow_loops = False,
            n_init=n_init,
            simultaneous=simultaneous,
        )

        job = Parallel(n_jobs=-1)(delayed(Cascade_Analyzer.run_cascade)(source_indices_list[i], cdispatch, indicator=f'Cascade {i}/{len(source_indices_list)} started...') for i in tqdm(range(0, len(source_indices_list))))
        data = [Cascade_Analyzer(name=source_names[i], hit_hist=hit_hist, n_init=n_init, skids_in_hit_hist=False, adj_index=adj.index) for i, hit_hist in enumerate(job)]
        return(data)

    @staticmethod
    def run_single_cascade(name, source_skids, stop_skids, adj, p, max_hops, n_init, simultaneous):

        source_indices = np.where([x in source_skids for x in adj.index])[0]
        stop_indices = np.where([x in stop_skids for x in adj.index])[0]
        transition_probs = to_transmission_matrix(adj.values, p)

        cdispatch = TraverseDispatcher(
            Cascade,
            transition_probs,
            stop_nodes = stop_indices,
            max_hops=max_hops+1, # +1 because max_hops includes hop 0
            allow_loops = False,
            n_init=n_init,
            simultaneous=simultaneous,
        )

        cascade = Cascade_Analyzer.run_cascade(i = source_indices, cdispatch = cdispatch)
        data = Cascade_Analyzer(name=name, hit_hist=cascade, n_init=n_init, skids_in_hit_hist=False, adj_index=adj.index)
        return(data)

