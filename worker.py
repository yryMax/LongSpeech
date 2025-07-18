import os
from pathlib import Path


def save_one_worker(cut, out_dir):
    os.environ["LHOTSE_AUDIO_DURATION_MISMATCH_TOLERANCE"] = "1.5"
    dst = Path(out_dir) / f"{cut.id}.wav"
    
    # 假设 cut 对象有一个 save_audio 方法
    cut.save_audio(dst, format="wav")
    
    return cut.id