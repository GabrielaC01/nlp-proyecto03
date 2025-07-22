import os
import numpy as np
import pandas as pd
from transformers import AutoTokenizer, AutoModel
from utils import (
    cargar_corpus,
    tokenizar_por_espacios,
    tokenizar_con_bpe,
    obtener_embedding_desde_tokens,
    proyectar_embeddings_pca,
    guardar_matriz_similitud
)

# Crear carpeta de resultados
os.makedirs("resultados", exist_ok=True)

# Cargar modelo y tokenizer 
tokenizer_hf = AutoTokenizer.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
modelo_hf = AutoModel.from_pretrained("dccuchile/bert-base-spanish-wwm-uncased")
modelo_hf.eval()

# Cargar corpus
corpus = cargar_corpus("data/corpus.txt")

# Tokenización por espacios y BPE
tokenes_espacio = tokenizar_por_espacios(corpus)
tokenes_bpe = tokenizar_con_bpe(corpus, "data/bpe_model.model")

# Obtener embeddings
embeddings_espacio = [obtener_embedding_desde_tokens(tokens, tokenizer_hf, modelo_hf) for tokens in tokenes_espacio]
embeddings_bpe = [obtener_embedding_desde_tokens(tokens, tokenizer_hf, modelo_hf) for tokens in tokenes_bpe]

# Verificar formas
print("Forma embeddings (espacios):", np.array(embeddings_espacio).shape)
print("Forma embeddings (BPE):     ", np.array(embeddings_bpe).shape)

# Guardar matrices de similitud coseno
guardar_matriz_similitud(embeddings_espacio, corpus, "resultados/similitud_espacios.csv")
guardar_matriz_similitud(embeddings_bpe, corpus, "resultados/similitud_bpe.csv")

# Proyección PCA
proyectar_embeddings_pca(embeddings_espacio, corpus, "PCA - Tokenización por Espacios", "resultados/pca_espacios.png")
proyectar_embeddings_pca(embeddings_bpe, corpus, "PCA - Tokenización BPE", "resultados/pca_bpe.png")
