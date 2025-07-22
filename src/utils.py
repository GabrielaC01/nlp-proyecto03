import unicodedata
import re
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sentencepiece as spm

def cargar_corpus(ruta: str) -> List[str]:
    """
    Carga el corpus desde un archivo de texto plano

    Args:
        ruta (str): ruta del archivo .txt

    Returns:
        List[str]: lista de frases
    """
    with open(ruta, "r", encoding="utf-8") as f:
        return [linea.strip() for linea in f if linea.strip()]


def tokenizar_por_espacios(frases):
    """
    Tokeniza una lista de frases por espacios, aplicando limpieza básica: eliminación de tildes, minúsculas y símbolos

    Args:
        frases (List[str]): lista de frases

    Returns:
        List[List[str]]: lista de tokens por frase
    """

    frases_limpias = []
    for frase in frases:
        frase = frase.lower()
        frase = unicodedata.normalize("NFD", frase)
        frase = "".join(c for c in frase if unicodedata.category(c) != "Mn")
        frase = re.sub(r"[^\w\s]", "", frase)
        frase = re.sub(r"\s+", " ", frase).strip()
        frases_limpias.append(frase.split())

    return frases_limpias

def tokenizar_con_bpe(frases, model_file):
    """
    Tokeniza una lista de frases usando un modelo BPE entrenado con SentencePiece

    Args:
        frases (List[str]): lista de frases
        model_file (str): ruta al modelo BPE (.model)

    Returns:
        List[List[str]]: lista de subpalabras por frase
    """

    sp = spm.SentencePieceProcessor(model_file=model_file)

    return [sp.encode(frase, out_type=str) for frase in frases] 

