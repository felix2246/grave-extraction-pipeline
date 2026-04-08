import os
import unicodedata
from functools import wraps
from time import time

import ftfy

from grave_extraction.logger import logger


def timed(f):
    @wraps(f)
    def wrapper(*args, **kwds):
        start = time()
        result = f(*args, **kwds)
        elapsed = time() - start
        filename = os.path.basename(f.__code__.co_filename)
        logger.info("%s:%s took %.3f seconds to finish", filename, f.__name__, elapsed)
        return result

    return wrapper


def normalize_text(text: str) -> str:
    text = text.lower()
    text = unicodedata.normalize("NFD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    text = " ".join(text.split())
    text = text.replace('"', "")
    text = text.replace(" ", "")
    text = text.replace("„", "").replace("\u201c", "")
    text = text.replace("–", "-")

    return text


def read_file(file_path: str) -> str:
    with open(file_path, "r", encoding="utf-8") as f:
        return f.read()


def fix_text_encoding(v):
    if isinstance(v, str):
        return ftfy.fix_text(v)
    return v
