from pathlib import Path
from typing import List, Union
import json
import numpy as np
import librosa
import soundfile as sf


class LongSpeechEntity:
    """
    不同数据集的若干片段拼接的一条长音频。
    """

    """
    dataset metadata
    """
    AVG_DURATION = 600 # 10min = 600s
    SAMPLE_RATE = 16000
    OUT_DIR = '../datasets/LongSpeech'




    def __init__(self, id, ds):
        """
        Args:
            id: accending sequence_number
            ds: source_ds
        """
        # appendable numpy array
        self.audio_list = []
        self.duration_sec = 0
        self.components = []
        self.transcribe = ""
        self.id = id
        self.source_ds = ds

    def append(self, other_list, transcribe):
        """
        根据不同的策略追加，认为已经重采样到了目标采样率
        """




    def append_from_dataset_row(
        self,
        row: dict,
        file_key: str = "audio",
        to_path: bool = True,
    ):
        """
        从 Hugging Face / 自建数据集的一行记录追加片段。

        Args:
            row: 单行样本（dict）
            file_key: 行中指向音频文件或嵌入对象的键
            to_path: 若 row[file_key] 已是类文件对象，设为 False
        """
        if to_path:
            # row[file_key] 是路径或 datasets.Audio 对象
            path = Path(row[file_key]).expanduser().resolve()
            self.append_wav(path)
        else:
            # row[file_key] 已经是 ndarray / list / bytes
            audio = np.asarray(row[file_key], dtype=np.float32)
            if len(audio.shape) > 1:  # 立体声转单声道
                audio = librosa.to_mono(audio.T)
            if row.get("sampling_rate") and row["sampling_rate"] != self.target_sr:
                audio = librosa.resample(audio, orig_sr=row["sampling_rate"], target_sr=self.target_sr)
            self._segments.append(audio)
            self._segment_files.append(f"<inline>{row.get('id', len(self._segment_files))}")

    # -------- 导出 --------
    def export_wav(self, out_path: Union[str, Path]):
        """
        将所有片段拼接后保存成 WAV。
        """
        out_path = Path(out_path).expanduser().resolve()
        audio_concat = np.concatenate(self._segments) if self._segments else np.array([], dtype=np.float32)
        sf.write(out_path, audio_concat, self.target_sr)
        return out_path

    def dump_metadata(self, jsonl_path: Union[str, Path]):
        """
        导出元数据（仅自身信息 + file_name 列表）到 JSONL：
            第 1 行：整体信息
            后续行：{"file_name": "..."}
        """
        jsonl_path = Path(jsonl_path).expanduser().resolve()
        meta = {
            "target_sr": self.target_sr,
            "num_segments": len(self._segments),
            "total_samples": int(sum(len(a) for a in self._segments)),
        }
        with jsonl_path.open("w", encoding="utf-8") as f:
            f.write(json.dumps(meta, ensure_ascii=False) + "\n")
            for file_name in self._segment_files:
                f.write(json.dumps({"file_name": file_name}, ensure_ascii=False) + "\n")
        return jsonl_path

    # -------- 便捷属性 --------
    @property
    def duration_sec(self) -> float:
        """返回当前拼接后总时长（秒）。"""
        return sum(len(a) for a in self._segments) / self.target_sr

    @property
    def segment_files(self) -> List[str]:
        """只读：已追加的片段文件名列表。"""
        return self._segment_files.copy()

    @duration_sec.setter
    def duration_sec(self, value):
        self._duration_sec = value


if __name__ == "__main__":
    # 简单示例：将两个本地 wav 拼接
    lse = LongSpeechEntity(target_sr=16_000)
    lse.append_wav("clip1.wav")
    lse.append_wav("clip2.wav")
    wav_out = lse.export_wav("combined.wav")
    meta_out = lse.dump_metadata("combined_meta.jsonl")
    print(f"拼接完成，WAV -> {wav_out}, 元数据 -> {meta_out}, 总时长 {lse.duration_sec:.1f}s")
