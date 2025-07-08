from transformers import AutoTokenizer, AutoModelForTokenClassification, pipeline
import re

print("abc")

_MODEL_ID = "oliverguhr/fullstop-punctuation-multilang-large"
_tokenizer = AutoTokenizer.from_pretrained(_MODEL_ID)
_model     = AutoModelForTokenClassification.from_pretrained(_MODEL_ID)
_punctifier = pipeline(
    task="token-classification",
    model=_model,
    tokenizer=_tokenizer,
    aggregation_strategy="simple",
    device_map="auto"
)
print("defg")
def add_punctuation(text: str, capitalize: bool = True) -> str:
    """
    标点恢复
    """
    if not text:
        return text

    text = text.lower().strip()

    tokens = _punctifier(text)

    words = []
    for tok in tokens:
        w = tok["word"]

        if w in {".", ",", "!", "?", ";", ":"}:

            if words:
                words[-1] = words[-1] + w
        else:
            words.append(w)

    sent = " ".join(words)


    sent = re.sub(r"\s+([,.!?;:])", r"\1", sent)

    if capitalize:
        sent = sent[:1].upper() + sent[1:]

    return sent

plain = "FROM THE UNDER SURFACE OF THE CLOUDS THERE ARE CONTINUAL EMISSIONS OF LURID LIGHT ELECTRIC MATTER IS IN CONTINUAL EVOLUTION FROM THEIR COMPONENT MOLECULES THE GASEOUS ELEMENTS OF THE AIR NEED TO BE SLAKED WITH MOISTURE FOR INNUMERABLE COLUMNS OF WATER RUSH UPWARDS INTO THE AIR AND FALL BACK AGAIN IN WHITE FOAM I REFER TO THE THERMOMETER IT INDICATES THE FIGURE IS OBLITERATED IS THE ATMOSPHERIC CONDITION HAVING ONCE REACHED THIS DENSITY TO BECOME FINAL"

print(add_punctuation(plain))