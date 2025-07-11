from lhotse import CutSet
from lhotse.recipes import prepare_librispeech
from lhotse.cut import append_cuts
import json

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

def jsonl_head(jsonl_path, n=10):
    """
    Read the first n lines of a jsonl file.
    :param jsonl_path: path to the jsonl file
    :param n: number of lines to read
    :return: list of dictionaries
    """
    assert jsonl_path.endswith('.jsonl')
    output_path = jsonl_path[: -6] + "_head.jsonl"
    with open(jsonl_path, 'r', encoding='utf-8') as f:
        lines = [json.loads(line) for line in f.readlines()[:n]]
    with open(output_path, 'w', encoding='utf-8') as f:
        for line in lines:
            f.write(json.dumps(line, ensure_ascii=False) + '\n')
    return output_path

if __name__ == '__main__':
    json_path = "../datasets/LongSpeech/raw_cuts.jsonl"
    print(jsonl_head(json_path, 10))

