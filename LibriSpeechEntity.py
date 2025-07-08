
from LongSpeechEntity import LongSpeechEntity
from util import *

class LibriSpeechEntity(LongSpeechEntity):
    """
    LibriSpeech数据集的若干片段拼接的一条长音频。
    """

    """
    dataset metadata
    """
    AUDIO_AUTO = False
    TEST_AUTO = False

    def __init__(self, id):
        """
        Args:
            id: accending sequence_number
            ds: source_ds
        """
        super().__init__(id, "LibriSpeech")

    def appendaudio(self, audio_data, transcribe, path):
        transcribe = restore_punctuation(transcribe)
        super().appendaudio(audio_data, transcribe, path)
