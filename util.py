from deepmultilingualpunctuation import PunctuationModel
from sentence_transformers import SentenceTransformer
#from speechbrain.pretrained import EncoderClassifier
import torch
import torchaudio
import numpy as np

punct = PunctuationModel()
text_embedding = SentenceTransformer('sentence-transformers/all-MiniLM-L6-v2')
"""
speaker_embedding = EncoderClassifier.from_hparams(
        source="speechbrain/spkrec-ecapa-voxceleb",
        run_opts={"device": "cpu"}
    )
speaker_embedding.eval()
"""
def restore_punctuation(text: str) -> str:
    """
    标点恢复
    """
    punctuated_text = punct.restore_punctuation(text.lower())


    # capitailize the first letter of each sentence
    sentences = punctuated_text.split('. ')
    punctuated_text = '. '.join(sentence.capitalize() for sentence in sentences)

    return punctuated_text


def get_sentence_embeddings(texts: list):
    """
    获取句子嵌入
    Args:
        texts: list of sentences
    Return:
        (d,384) numpy array
    """
    return text_embedding.encode(texts)

def get_speaker_embedding(samples: np.ndarray, sampling_rate: int) -> np.ndarray:
    """
    获取说话人嵌入
    Args:
        samples: numpy ndarray，形状为 (1, num_samples)
    Return:
        numpy.ndarray，形状为 (1, 192)
    """
    samples = torch.tensor(samples, dtype=torch.float32)
    if samples.dim() == 1:
        samples = samples.unsqueeze(0)
    if sampling_rate != 16000:
        resampler = torchaudio.transforms.Resample(orig_freq=sampling_rate, new_freq=16000)
        samples = resampler(samples)

    assert samples.dim() == 2
    assert samples.shape[0] == 1
    return speaker_embedding.encode_batch(samples, wav_lens=torch.tensor([1.0], device="cpu")).squeeze(0).numpy()


def test_restore_punctuation():
    text  = "SO THERE CAME A STEP AND A LITTLE RUSTLING OF FEMININE DRAPERIES THE SMALL DOOR OPENED AND RACHEL ENTERED WITH HER HAND EXTENDED AND A PALE SMILE OF WELCOME WOMEN CAN HIDE THEIR PAIN BETTER THAN WE MEN AND BEAR IT BETTER TOO EXCEPT WHEN SHAME DROPS FIRE INTO THE DREADFUL CHALICE"
    print(restore_punctuation(text))


def test_get_sentence_embeddings():
    texts = ["Hello world", "This is a test sentence.", "Deep learning is fascinating!"]
    embeddings = get_sentence_embeddings(texts)
    print(type(embeddings))

def test_get_speaker_embedding():
    audio_path = "/mnt/d/voicedata/CommenVoice/delta/en/clips/common_voice_en_42696072.mp3"
    audio, sr = torchaudio.load(audio_path)
    speaker_embedding = get_speaker_embedding(audio.numpy(), sr)
    assert speaker_embedding.shape == (1, 192)


if __name__ == '__main__':
    #test_restore_punctuation()
    #test_get_sentence_embeddings()
    test_get_speaker_embedding()