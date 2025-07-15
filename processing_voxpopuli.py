import multiprocessing as mp
from pathlib import Path
import os
import json
import pandas as pd
from tqdm import tqdm
from lhotse import CutSet, RecordingSet, SupervisionSet, MonoCut
from lhotse.cut import append_cuts
from mylhotse.voxpopuli import prepare_voxpopuli, ASR_LANGUAGES
import logging
from functools import reduce
from datasets import load_dataset, DownloadConfig

# directory paths to save audio and transcript files
IN_DIR = "../datasets/LongSpeechSource/voxpopuli"
# directory paths to save metadata and processed aduio files
OUT_DIR = '../datasets/LongSpeech'


def json_from_voxpopuli_to_allaudios(one_cut, lang = "en"):
    """
    Convert a single LibriSpeech json record to a list of LongSpeech metadata.
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
        sources.append(full_pth.split("voxpopuli")[-1])
        [speakers.add(s["speaker"]) for s in subcut["cut"]["supervisions"] if s["speaker"]]
        transcript_param = " ".join([s["text"] for s in subcut["cut"]["supervisions"] if s["text"]])
        if transcript_param != "":
            transcripts.append(transcript_param)
        else:
            print(subcut)

    return {
        "id": one_cut["id"],
        "source_ds": "voxpopuli",
        "duration_sec": total_dur,
        "audio_auto": False,
        "text_auto": False,
        "language": lang,
        "num_speakers": len(speakers),
        "num_switches": len(one_cut["tracks"]),
        "slice": slices,
        "transcribe": " ".join(transcripts),
        "components": sources,
    }

def pack_cuts_to_long_audio(
    cuts: CutSet,
    target_duration: float = 600.0,
    starting_id: int = 0,
) -> (CutSet, int):
    final_long_cuts = []
    buffer_cut = None

    for cut in cuts:
        buffer_cut = cut if buffer_cut is None else buffer_cut.append(cut)
        if buffer_cut.duration >= target_duration:
            final_long_cuts.append(buffer_cut.with_id(f"{starting_id:06d}"))
            starting_id += 1
            buffer_cut = None

    return CutSet.from_cuts(final_long_cuts), starting_id

def convert_record(source_jsonl_path: str, target_jsonl_path: str, map_fn, lang: str):
    with open(source_jsonl_path, "r", encoding="utf-8") as src_f, \
         open(target_jsonl_path, "a", encoding="utf-8") as tgt_f:
        for line in src_f:
            item = json.loads(line)
            new_item = map_fn(item, lang)
            tgt_f.write(json.dumps(new_item, ensure_ascii=False) + "\n")

def save_audios_from_cutset(cutset, out_dir, num_jobs=1):
    """
    Save audios from a CutSet to the specified directory.
    """
    for cut in tqdm(cutset):
        cut.save_audio(os.path.join(out_dir, f"{cut.id}.wav"))


if __name__ == '__main__':
    config = json.load(open(os.path.join(OUT_DIR, 'metadata.json')))
    AVG_DURATION = config['avg_duration']
    SAMPLE_RATE = config['sample_rate']
    OUT_FILE_NAME = config['source']
    prev_amount = config['amount']
    print(prev_amount)
    task = "asr"
    for lang in ASR_LANGUAGES:
        if lang == "en":
            continue
        print("Processing {lang}")
        manifests = prepare_voxpopuli(corpus_dir=IN_DIR, output_dir=OUT_DIR, task=task, lang=lang, num_jobs=15)
        cuts = CutSet()
        for part in manifests.keys():
            rs = manifests[part]['recordings']
            ss = manifests[part]['supervisions']
            cut = CutSet.from_manifests(recordings=rs, supervisions=ss)
            cuts += cut
            spilted_cuts = cuts.trim_to_supervision_groups(max_pause=5).filter(lambda cut: cut.duration > 10).sort_by_recording_id().merge_supervisions(merge_policy="keep_first")
            grouped_cuts, new_amount = pack_cuts_to_long_audio(spilted_cuts, target_duration=600.0, starting_id = prev_amount)
            grouped_cuts.to_jsonl(OUT_DIR + "/grouped_cuts.jsonl")
            convert_record(os.path.join(OUT_DIR, "grouped_cuts.jsonl"),
                        os.path.join(OUT_DIR, OUT_FILE_NAME),
                        json_from_voxpopuli_to_allaudios, lang)
            save_audios_from_cutset(grouped_cuts, os.path.join(OUT_DIR, 'wavs'))
            print(f"Total amount: {new_amount}")
            prev_amount = new_amount




