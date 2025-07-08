
from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

_MODEL_ID = "oliverguhr/fullstop-punctuation-multilang-large"
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
_model     = AutoModelForTokenClassification.from_pretrained(_MODEL_ID)
_punctifier = pipeline(
    task="token-classification",
    model=_model,
    tokenizer=_tokenizer,
    aggregation_strategy="simple",  # 聚合成词级
    device_map="auto",
)

# 标签 → 标点 映射表
PUNCT_MAP = {
    "COMMA": ",",
    "PERIOD": ".",
    "QUESTION": "?",
    "EXCLAMATION": "!",
    "COLON": ":",
    "SEMICOLON": ";",
}

def add_punctuation(text: str, capitalize: bool = True) -> str:
    """
    使用 fullstop-punctuation 恢复英文标点
    Args:
        text: 原始大写/无标点文本
        capitalize: 是否把首字母大写
    Returns:
        带标点、正常大小写的文本
    """
    if not text:
        return text

    # 1) 统一小写可提升识别率
    text = text.lower().strip()

    # 2) 送模型预测标签
    preds = _punctifier(text)

    # 3) 根据标签插入标点
    words = []
    for tok in preds:
        word = tok["word"]
        label = tok.get("entity_group", tok.get("entity", "O"))
        words.append(word)
        if label in PUNCT_MAP:
            words[-1] += PUNCT_MAP[label]

    sent = " ".join(words)

    # 4) 去掉多余空格（标点前）
    sent = re.sub(r"\s+([,.!?;:])", r"\1", sent)

    # 5) 句首大写
    if capitalize and sent:
        sent = sent[0].upper() + sent[1:]

    return sent

plain = ("FROM THE UNDER SURFACE OF THE CLOUDS THERE ARE CONTINUAL EMISSIONS OF LURID LIGHT "
         "ELECTRIC MATTER IS IN CONTINUAL EVOLUTION FROM THEIR COMPONENT MOLECULES THE GASEOUS "
         "ELEMENTS OF THE AIR NEED TO BE SLAKED WITH MOISTURE FOR INNUMERABLE COLUMNS OF WATER "
         "RUSH UPWARDS INTO THE AIR AND FALL BACK AGAIN IN WHITE FOAM I REFER TO THE THERMOMETER "
         "IT INDICATES THE FIGURE IS OBLITERATED IS THE ATMOSPHERIC CONDITION HAVING ONCE REACHED "
         "THIS DENSITY TO BECOME FINAL")

print(add_punctuation(plain))