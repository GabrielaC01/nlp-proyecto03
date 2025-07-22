import unicodedata
import re
from typing import List
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import sentencepiece as spm
import numpy as np
import torch
import pandas as pd

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


def tokenizar_por_espacios(frases: List[str]) -> List[List[str]]:
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

def tokenizar_con_bpe(frases: List[str], model_file: str) -> List[List[str]]:
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

def obtener_embedding_desde_tokens(tokens: List[str], tokenizer, modelo) -> np.ndarray:
    """
    Obtiene el embedding promedio de una lista de tokens usando BERT

    Args:
        tokens (List[str]): Lista de tokens de entrada
        tokenizer: Tokenizador de Hugging Face
        modelo: Modelo BERT cargado

    Returns:
        np.ndarray: Vector de embedding promedio de la frase
    """

    texto = " ".join(tokens)
    inputs = tokenizer(texto, return_tensors="pt", truncation=True, max_length=512)
    with torch.no_grad():
        outputs = modelo(**inputs)
        embedding = outputs.last_hidden_state[0][1:-1].mean(dim=0)
    return embedding.numpy()

def proyectar_embeddings_pca(embeddings: np.ndarray, etiquetas: List[str], titulo: str, nombre_archivo: str) -> None:
    """
    Proyecta los embeddings en 2D con PCA y guarda el gráfico

    Args:
        embeddings (np.ndarray): Matriz de embeddings
        etiquetas (List[str]): Lista de frases originales
        titulo (str): Título del gráfico
        nombre_archivo (str): Ruta del archivo de salida (.png)
    """

    pca = PCA(n_components=2)
    embeddings_2d = pca.fit_transform(embeddings)

    xs = embeddings_2d[:, 0]
    ys = embeddings_2d[:, 1]

    plt.figure(figsize=(20, 15))
    for punto, etiqueta in zip(embeddings_2d, etiquetas):
        plt.scatter(punto[0], punto[1])
        plt.annotate(etiqueta, (punto[0], punto[1]), fontsize=10)

    plt.xlim(xs.min() - 1, xs.max() + 2)
    plt.ylim(ys.min() - 1, ys.max() + 1)
    plt.title(titulo)
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(nombre_archivo)
    print(f"Gráfico guardado como: {nombre_archivo}")


def guardar_matriz_similitud(embeddings: np.ndarray, frases: List[str], nombre_archivo_csv: str) -> None:
    """
    Calcula y guarda la matriz de similitud coseno como CSV

    Args:
        embeddings (np.ndarray): Matriz de embeddings
        frases (List[str]): Frases originales del corpus
        nombre_archivo_csv (str): Ruta del archivo CSV de salida
    """

    sim_matrix = cosine_similarity(embeddings)
    df_sim = pd.DataFrame(sim_matrix, index=[f"Frase {i+1}" for i in range(len(frases))],
                          columns=[f"Frase {i+1}" for i in range(len(frases))])
    df_sim.to_csv(nombre_archivo_csv)
    print(f"Matriz de similitud guardada en: {nombre_archivo_csv}")