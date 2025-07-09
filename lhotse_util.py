from lhotse import CutSet
from lhotse.recipes import prepare_librispeech
from lhotse.cut import append_cuts


def from_strategy_to_cuts(source_cuts, strategy: list):
    """
    source_cuts: the cuts contains audio segments
    strategy: a list of list of cut_ids
    :return
        target_cuts: the cuts after applying the combination strategy
    """
    target_cuts_list = []
    for cluster_ids in strategy:
        cutlist = [source_cuts[cut_id] for cut_id in cluster_ids if cut_id in source_cuts]
        new_cut = append_cuts(cutlist)
        target_cuts_list.append(new_cut)
    return CutSet(target_cuts_list)


