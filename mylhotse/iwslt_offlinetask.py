import logging
import re
import string
from concurrent.futures.thread import ThreadPoolExecutor
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Union
from datasets import load_dataset
import tqdm
from itertools import chain
from lhotse import (
    AudioSource,
    Recording,
    RecordingSet,
    SupervisionSegment,
    SupervisionSet,
    validate_recordings_and_supervisions,
)
from lhotse.qa import fix_manifests
from lhotse.utils import Pathlike
from huggingface_hub import snapshot_download


prefix   = "iwslt_offline"
pair_key = "de-en"

def download_iwslt_offlinetask(
    target_dir: Pathlike = ".",
) -> None:
    """
    Download the IWSLT.OfflineTask dataset from Hugging Face.

    :param target_dir: Directory where the dataset should be downloaded.
    """


    snapshot_download(
        repo_id="IWSLT/IWSLT.OfflineTask",
        repo_type="dataset",
        local_dir=target_dir
    )



def clean_token(w: str) -> str:
    NOISE_PREFIXES = ('$(', '<', '[')
    w = re.sub(r'\(\d+\)$', '', w)     # 去掉尾部(1)
    if w.startswith(NOISE_PREFIXES):   # 过滤噪声占位
        return ''
    return w

def parse_ctm_to_supervisions(ctm_path: Path, recording_id: str, channel: int = 0):
    sups = []
    seg_start = None
    seg_id = None
    words = []
    last_word_end = 0.0

    def flush_segment():
        nonlocal sups, seg_id, seg_start, words, last_word_end
        if seg_id is None:
            return
        end_time = last_word_end
        if end_time is None or seg_start is None or end_time <= seg_start:
            return
        text = " ".join(words)
        if not text == "":
            sups.append(
                SupervisionSegment(
                    id=seg_id,
                    recording_id=recording_id,
                    start=seg_start,
                    duration=end_time - seg_start,
                    channel=channel,
                    text=text,
                    language="en",
                    speaker = recording_id,
                )
            )
        # reset
        seg_id = None
        seg_start = None
        words = []

    with ctm_path.open() as f:
        for ln in f:
            ln = ln.strip()
            if not ln:
                continue
            if ln.startswith('#'):
                flush_segment()
                parts = ln[1:].split()
                new_seg_id = parts[0]
                new_start = float(parts[1])
                seg_id = new_seg_id
                seg_start = new_start
                continue

            parts = ln.split()
            if len(parts) < 5:
                continue
            # utt = parts[0]; ch = int(parts[1])
            st, du, wd = float(parts[2]), float(parts[3]), parts[4]
            wd = clean_token(wd)
            if not wd:
                continue
            words.append(wd)
            last_word_end = st + du

    flush_segment()
    return SupervisionSet.from_segments(sups)

def fname(kind: str) -> str:
    return f"{prefix + '_' if prefix else ''}{pair_key}_{kind}.jsonl.gz"


def prepare_iwslt_offlinetask(
    corpus_dir: Pathlike,
    output_dir: Pathlike,
):
    audio_dir = corpus_dir / "data/en-de"
    all_recordings = []
    all_supervisions = []

    for split_dir in audio_dir.iterdir():

        file_order = split_dir / "FILE_ORDER"

        wav_dir = split_dir / "wavs"
        ctm_dir = split_dir / "ctms"

        if not wav_dir.is_dir() or not ctm_dir.is_dir():
            print(f"[WARN] skip {split_dir}: wav/ or ctms/ missing.")
            continue

        ids = [l.strip() for l in file_order.read_text().splitlines() if l.strip()]
        # ---- Supervisions ----
        sup_sets_this_split = []
        for rid in ids:
            ctm = ctm_dir / f"{rid}.ctm"
            if not ctm.is_file():
                print(f"[WARN] missing ctm: {ctm}")
                continue
            sup_sets_this_split.append(parse_ctm_to_supervisions(ctm, rid))
        sups_split = SupervisionSet.from_segments(seg for s in sup_sets_this_split for seg in s)
        all_supervisions.append(sups_split)

        # ---- Recordings ----
        recs_split = RecordingSet.from_recordings(
            Recording.from_file(wav_dir / f"{rid}.wav", recording_id=rid)
            for rid in ids
            if (wav_dir / f"{rid}.wav").is_file()
        )
        all_recordings.append(recs_split)

    recordings = RecordingSet.from_recordings(chain.from_iterable(r for r in all_recordings))
    supervisions = SupervisionSet.from_segments(chain.from_iterable(s for s in all_supervisions))

    recordings.to_file(output_dir / fname("recordings"))
    supervisions.to_file(output_dir / fname("supervisions"))
    manifests = {
            "en-de": {
            "recordings": recordings,
            "supervisions": supervisions
        }
    }
    return manifests


if __name__ == '__main__':
    snapshot_download(
        repo_id="IWSLT/IWSLT.OfflineTask",
        repo_type="dataset",
        local_dir="../../IWSLT_OfflineTask"
    )