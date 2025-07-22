import multiprocessing as mp
from pathlib import Path
import os
import json
import pandas as pd
from tqdm import tqdm
from lhotse import CutSet
from lhotse.recipes import prepare_commonvoice
import logging
from util import get_sentence_embeddings
import numpy as np
import faiss, gc
from lhotse_util import from_strategy_to_cuts
from lhotse.cut import append_cuts
import librosa
from bitarray import bitarray
from concurrent.futures import ProcessPoolExecutor, as_completed
from worker import save_one_worker

def build_feature(cuts: CutSet, batch_size: int = 100, dim: int = 384):
    cut_list = cuts.to_eager()
    n = len(cut_list)

    vec_mm = np.memmap(f"{OUT_DIR}/vecs.f32", dtype="float32", mode="w+", shape=(n, dim))
    dur_mm = np.memmap(f"{OUT_DIR}/durs.f32", dtype="float32", mode="w+", shape=(n,))

    string_ids = []

    ptr = 0
    for i in tqdm(range(0, n, batch_size), desc="Get Embedding"):
        cut_batch = cut_list[i:i+batch_size]

        texts = [c.supervisions[0].text if c.supervisions else "" for c in cut_batch]
        durations = [c.duration for c in cut_batch]
        string_ids.extend([c.id for c in cut_batch])

        vec_np = get_sentence_embeddings(texts).astype("float32")
        B = len(cut_batch)

        vec_mm[ptr:ptr+B] = vec_np
        dur_mm[ptr:ptr+B] = durations
        ptr += B

    vec_mm.flush(); dur_mm.flush()

    return vec_mm, dur_mm, string_ids

def build_hnsw_index(vec_mm: np.memmap,
                     dim: int = 384,
                     m: int = 32,
                     ef_c: int = 200,
                     n_threads: int = mp.cpu_count(),
                     out_path: str = "cache_hnsw.faiss"):

    faiss.omp_set_num_threads(n_threads)
    faiss.normalize_L2(vec_mm)

    index = faiss.IndexHNSWFlat(dim, m)
    index.hnsw.efConstruction = ef_c
    index.metric_type = faiss.METRIC_INNER_PRODUCT

    index.add(vec_mm)
    faiss.write_index(index, os.path.join(OUT_DIR,out_path))
    return os.path.join(OUT_DIR,out_path)


def greedy_cluster(index_path: str,
                   vec_mm: np.memmap,
                   dur_mm: np.memmap,
                   ids,
                   cuts,
                   bucket_min: int = 480,
                   bucket_avg: int = 600,
                   k_neigh: int = 256,
                   ef_s: int = 96):
    index = faiss.read_index(index_path)

    params = faiss.SearchParametersHNSW()
    params.efSearch = ef_s

    N = len(vec_mm)
    assigned = bitarray(N)
    assigned.setall(False)

    order = np.argsort(-dur_mm)
    buckets = []

    for seed in tqdm(order, desc="Clustering (Optimized)"):
        if assigned[seed]:
            continue

        cluster = []
        total_dur = 0

        unassigned_indices_list = assigned.search(bitarray('0'))
        unassigned_indices = np.fromiter(unassigned_indices_list, dtype=np.int64)


        if len(unassigned_indices) > 0:
            selector = faiss.IDSelectorArray(unassigned_indices)
            params.sel = selector

            _, neighs = index.search(vec_mm[seed : seed + 1], k_neigh, params=params)


            for idx in neighs[0]:
                if idx == -1:
                    break
                if assigned[idx]:
                    print("Warning: Already assigned index", idx)
                    continue

                cluster.append(int(idx))
                assigned[idx] = True
                total_dur += dur_mm[idx]
                if total_dur >= bucket_avg:
                    break

            if total_dur < bucket_min:
                for i in cluster:
                    assigned[i] = False
            else:
                total_dur = dur_mm[cluster].sum()
                buckets.append((cluster, total_dur))

    final_buckets = [b for b in buckets if b[1] >= bucket_min]
    final_clusters = [c for c, _ in final_buckets]
    final_duration = sum(sec for _, sec in final_buckets)

    loss = 1 - final_duration / dur_mm.sum()
    print(f"桶数 {len(final_clusters)}, 最终时长 {final_duration:.2f}s, 总时长 {dur_mm.sum():.2f}s, 丢弃比例 {loss:.2%}")

    strategy = []
    for cluster in final_clusters:
        strategy.append([ids[i] for i in cluster])

    return strategy


def map_newid_cutset(cutset: CutSet, start_id: int = 0) -> CutSet:
    """
    Map the ids of a CutSet to a new id starting from start_id.
    """
    new_cuts = []
    for i, cut in enumerate(cutset):
        new_cut = cut.with_id(f"{start_id + i:06d}")
        new_cuts.append(new_cut)
    return CutSet.from_cuts(new_cuts), start_id + len(new_cuts)

def build_grouped_cuts(
        source_cuts: CutSet,
        strategy,
        start_id: int = 0
    ):

    src = {c.id: c for c in source_cuts}

    grouped = []
    next_id = start_id
    for cluster_ids in strategy:
        cuts = [src[cid].resample(SAMPLE_RATE) for cid in cluster_ids]
        merged = append_cuts(cuts).with_id(f"{next_id:06d}")
        grouped.append(merged)
        next_id += 1

    return CutSet.from_cuts(grouped), next_id

def json_from_commonvoice_to_allaudios(one_cut, lang = "en"):
    """
    Convert a single Commonvoice json record to a list of LongSpeech metadata.
    """
    sources = []
    speakers = set()
    total_dur = 0
    transcripts = []
    slices = []

    for subcut in one_cut["tracks"]:
        total_dur += subcut["cut"]["duration"]
        full_pth = subcut["cut"]["recording"]["sources"][0]["source"]
        slices.append([subcut["cut"]["start"], subcut["cut"]["duration"]])
        sources.append(full_pth.split("clips")[-1])
        [speakers.add(s["speaker"]) for s in subcut["cut"]["supervisions"] if s["speaker"]]
        transcript_param = " ".join([s["text"] for s in subcut["cut"]["supervisions"] if s["text"]])
        if transcript_param != "":
            transcripts.append(transcript_param)
        else:
            print(subcut)

    return {
        "id": one_cut["id"],
        "source_ds": "CommonVoice",
        "duration_sec": total_dur,
        "audio_auto": False,
        "text_auto": False,
        "language": lang,
        "num_speakers": len(speakers),
        "num_switches": len(transcripts),
        "slice": slices,
        "transcribe": " ".join(transcripts),
        "components": sources,
    }


def convert_record(source_jsonl_path: str, target_jsonl_path: str, map_fn, lang: str):
    with open(source_jsonl_path, "r", encoding="utf-8") as src_f, \
         open(target_jsonl_path, "a", encoding="utf-8") as tgt_f:
        for line in src_f:
            item = json.loads(line)
            new_item = map_fn(item, lang)
            tgt_f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

def save_audios_from_cutset(cutset, out_dir, num_jobs=None):
    if num_jobs is None:
        num_jobs = os.cpu_count()

    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    context = mp.get_context("spawn")

    with ProcessPoolExecutor(max_workers=num_jobs, mp_context=context) as pool:
        futures = [
            pool.submit(save_one_worker, cut, out_dir) 
            for cut in tqdm(cutset, desc="1. 提交任务中")
        ]
        for _ in tqdm(
        as_completed(futures),
        total=len(futures),
        desc=f"Saving WAVs ({num_jobs} workers)"
        ):
            pass


COMMONVOICE_LANGS = "tr  de  es  fr  id  it  th  zh-CN".split()
IN_DIR = "../datasets/LongSpeechSource/cv-corpus-22.0-2025-06-20"
OUT_DIR = '/home/yangrenyi.yry/LongSpeech_p3'

if __name__ == '__main__':

    for lang in COMMONVOICE_LANGS:
        config = json.load(open(os.path.join(OUT_DIR, 'metadata.json')))
        AVG_DURATION = config['avg_duration']
        SAMPLE_RATE = config['sample_rate']
        OUT_FILE_NAME = config['source']
        prev_amount = config['amount']
        print(f"processing language {lang}, prev amount {prev_amount}")


        manifests = prepare_commonvoice(corpus_dir=IN_DIR, output_dir=OUT_DIR, languages = lang , splits=['validated'], num_jobs=10)
        cuts = CutSet()
        for part in manifests.keys():
            rs = manifests[part]['validated']['recordings']
            ss = manifests[part]['validated']['supervisions']
            cut = CutSet.from_manifests(recordings=rs, supervisions=ss)
            cuts += cut

        resampled_cuts = cuts.to_eager()
        resampled_cuts.to_jsonl(os.path.join(OUT_DIR, f"{lang}_commonvoice_raw_cuts.jsonl"))

        vec_mm, dur_mm, string_ids = build_feature(resampled_cuts)
        index_path = build_hnsw_index(vec_mm)
        real_strategy = greedy_cluster(index_path, vec_mm, dur_mm, string_ids, resampled_cuts)

        grouped_cuts, new_amount = build_grouped_cuts(resampled_cuts, real_strategy, start_id=prev_amount)
        grouped_cuts.to_jsonl(os.path.join(OUT_DIR, f"{lang}_commonvoice_grouped_cuts.jsonl"))
        print(f"new amount: {new_amount}")

        convert_record(os.path.join(OUT_DIR, f"{lang}_commonvoice_grouped_cuts.jsonl"),
                    os.path.join(OUT_DIR, OUT_FILE_NAME),
                    json_from_commonvoice_to_allaudios, lang)

        mp.set_start_method('spawn', force=True)
        save_audios_from_cutset(grouped_cuts, os.path.join(OUT_DIR, 'wavs'))

        config['amount'] = new_amount
        with open(os.path.join(OUT_DIR, 'metadata.json'), 'w', encoding='utf-8') as f:
            json.dump(config, f, indent=4, ensure_ascii=False)



    
