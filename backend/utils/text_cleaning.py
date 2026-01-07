# shared text cleaning
import re

def clean_text(text: str) -> str:
    """
    Cleans user input or dataset text for ML processing.
    """
    # if not isinstance(text, str):
    #     return ""

    # text = text.lower()
    # text = re.sub(r'\n', ' ', text)
    # text = re.sub(r'[^a-z0-9 ]', ' ', text)
    # text = re.sub(r'\s+', ' ', text).strip()
    # return text
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    return text
