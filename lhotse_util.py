import numpy as np
from lhotse import CutSet
from lhotse.cut import append_cuts
import json
import os
import tqdm
from speechbrain.pretrained import EncoderClassifier
from lhotse.features import FeatureExtractor
from lhotse.utils import Seconds
import torchaudio
import torch
from dataclasses import dataclass
from typing import Optional



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



def save_audios_from_cutset(cutset, out_dir, num_jobs=1):
    """
    Save audios from a CutSet to the specified directory.
    """
    for cut in tqdm(cutset):
        cut.save_audio(os.path.join(out_dir, f"{cut.id}.wav"))





@dataclass
class SpeakerEmbeddingConfig:
    """
    Configuration for the SpeakerEmbeddingExtractor.
    """
    device: str = "cpu"
    feature_dim: int = 192
    model_source: str = "speechbrain/spkrec-ecapa-voxceleb"


class SpeakerEmbeddingExtractor(FeatureExtractor):
    """
    一个使用 SpeechBrain 预训练模型提取 Speaker Embedding 的特征提取器。
    """
    name = "speaker_embedding"
    config_type = SpeakerEmbeddingConfig

    def __init__(self, config: Optional[SpeakerEmbeddingConfig] = None):
        super().__init__(config)
        self._feat_dim = self.config.feature_dim
        self.model = None

    def _initialize_model_if_needed(self):
        if self.model is not None:
            return

        self.model = EncoderClassifier.from_hparams(
            source=self.config.model_source,
            run_opts={"device": self.config.device}
        )

        self.model.eval()


    @property
    def frame_shift(self) -> Seconds:
        # I mean what the fuck????????????????????????
        return Seconds(100000.0)

    @property
    def feature_dim(self) -> int:
        return self._feat_dim

    def extract(self, samples: np.ndarray, sampling_rate: int) -> np.ndarray:
        """
        输入是原始音频波形，输出是 speaker embedding。

        :param samples: numpy ndarray，形状为 (1, num_samples) 或 (num_samples,)
        :param sampling_rate: 音频采样率.
        :return: numpy.ndarray，形状为 (1, feature_dim)，代表这个 cut 的 embedding
        """

        self._initialize_model_if_needed()
        samples = torch.tensor(samples, dtype=torch.float32)
        if sampling_rate != 16000:
            resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
            samples = resampler(samples)

        # 确保输入是 2D tensor (batch, samples)
        if samples.ndim == 1:
            samples = samples.unsqueeze(0)

        # 使用模型提取 embedding
        with torch.no_grad():
            # rel_length 参数用于处理不同长度的输入
            embedding = self.model.encode_batch(samples, wav_lens=torch.tensor([1.0], device=self.device))

        # embedding 的形状是 (1, 1, 192)，我们需要 (1, 192)
        return embedding.squeeze(1).cpu().numpy()




if __name__ == '__main__':
    #json_path = "../datasets/LongSpeech/raw_cuts.jsonl"
    #print(jsonl_head(json_path, 10))
    # test the custom feature extractor

    # Create with default config
    extractor1 = SpeakerEmbeddingExtractor()


    audio_path = "/mnt/d/voicedata/CommenVoice/delta/en/clips/common_voice_en_42696072.mp3"
    audio, sr = torchaudio.load(audio_path)
    print(extractor1.extract(audio.numpy(), sr))




