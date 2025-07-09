from abc import ABC, abstractmethod
import json
import numpy as np
import soundfile as sf
import random


class LongSpeechEntity:
    """
    不同数据集的若干片段拼接的一条长音频。
    """

    """
    dataset metadata
    """
    AVG_DURATION = 600 # 10min = 600s
    SAMPLE_RATE = 16000  # 16kHz (16000 samples per second), not seconds
    OUT_DIR = '../datasets/LongSpeech'


    @property
    @abstractmethod
    def AUDIO_AUTO(self) -> str:
        raise NotImplementedError


    @property
    @abstractmethod
    def TEST_AUTO(self) -> str:
        raise NotImplementedError



    def __init__(self, id=None, ds=None):
        """
        Args:
            id: accending sequence_number
            ds: source_ds
        """
        # appendable numpy array
        self.audio_list = []
        self.duration_sec = 0
        self.components = [] # 原始文件的filename
        self.transcribe = ""
        self.id = id # accending sequence_number
        self.source_ds = ds
        self.finished = False


    def get_metadata(self):
        """
        Returns metadata in jsonl format.
        """
        metadata = {
            "id": self.id,
            "source_ds": self.source_ds,
            "duration_sec": self.duration_sec,
            "components": self.components,
            "transcribe": self.transcribe,
            "audio_auto": self.AUDIO_AUTO,
            "test_auto": self.TEST_AUTO,
        }
        return json.dumps(metadata)


    def export_wav(self):
        """
        Export the concatenated audio to a wav file.
        """

        concatenated = np.concatenate(self.audio_list)
        output_path = f"{self.OUT_DIR}/wavs/{self.id:06d}.wav"
        sf.write(output_path, concatenated, self.SAMPLE_RATE)
        return output_path

    def appendaudio(self, audio_data, transcribe, path):
        """
        Append audio data to the entity.

        Args:
            audio_data: The audio data to append,
            it's assumed to be a list and already sampled to target sr
        """

        # Sample rate is in Hz (samples per second), not in seconds
        # To calculate duration in seconds, divide the number of samples by the sample rate
        cur_dur = len(audio_data) / self.SAMPLE_RATE
        if self.duration_sec >= self.AVG_DURATION:
            # get a random number [0, 1] to decide whether to append
            if random.random() < 0.5:
                self.finished = True
                return False
            self.duration_sec += cur_dur
            self.audio_list.append(audio_data)
            self.components.append(path)
            self.transcribe += " " + transcribe

        return True

