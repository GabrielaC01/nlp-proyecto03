<h2 align="center">
<p>Proyecto 3 - Tokenización y Embeddings Visuales</p>
</h2>

## Descripción
Este proyecto explora cómo diferentes tokenizadores afectan los embeddings y la similitud entre frases en español, comparando dos enfoques de tokenización: separación por espacios y Byte-Pair Encoding (BPE). Se utiliza un modelo preentrenado de Hugging Face para generar embeddings, y se proyectan en 2D con PCA para visualizar agrupaciones según su similitud semántica.

## Estructura del repositorio

- `src/`: Script principal y funciones auxiliares
- `data/`: Corpus de entrada y modelo BPE entrenado
- `resultados/`: Matrices de similitud y gráficos PCA
- `notebooks/`: Cuadernos exploratorios

## Cómo ejecutar

1. Clonar el repositorio
   ```bash
   git clone https://github.com/GabrielaC01/nlp-proyecto03.git
   cd nlp-proyecto03
  
2. Crear y activar un entorno virtual
    ```bash
   python3 -m venv .venv
   source .venv/bin/activate

3. Instalar dependencias
   ```bash
   pip install -r requirements.txt

4. Ejecutar el script principal
   ```bash
   python src/main_tokenizacion.py

5. Enlace al [video](https://drive.google.com/drive/folders/1MzI__laKMKhn5C1crOnJrHcYC0fC-TE2?usp=sharing)

## Autor
* Gabriela Colque  

