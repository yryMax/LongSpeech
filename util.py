from deepmultilingualpunctuation import PunctuationModel

punct = PunctuationModel()
def restore_punctuation(text: str) -> str:
    """
    标点恢复
    """
    punctuated_text = punct.restore_punctuation(text.lower())


    # capitailize the first letter of each sentence
    sentences = punctuated_text.split('. ')
    punctuated_text = '. '.join(sentence.capitalize() for sentence in sentences)

    return punctuated_text

def test_restore_punctuation():
    text  = "SO THERE CAME A STEP AND A LITTLE RUSTLING OF FEMININE DRAPERIES THE SMALL DOOR OPENED AND RACHEL ENTERED WITH HER HAND EXTENDED AND A PALE SMILE OF WELCOME WOMEN CAN HIDE THEIR PAIN BETTER THAN WE MEN AND BEAR IT BETTER TOO EXCEPT WHEN SHAME DROPS FIRE INTO THE DREADFUL CHALICE"
    print(restore_punctuation(text))

if __name__ == '__main__':
    test_restore_punctuation()