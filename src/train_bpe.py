import sentencepiece as spm

def entrenar_bpe(corpus_path: str, model_prefix: str, vocab_size: int = 100) -> None:
    """
    Entrena un modelo BPE con SentencePiece

    Args:
        corpus_path (str): Ruta al corpus preprocesado
        model_prefix (str): Prefijo para los archivos de salida
        vocab_size (int): Tamaño del vocabulario
    """
 
    spm.SentencePieceTrainer.train(
        input=corpus_path,
        model_prefix=model_prefix,
        vocab_size=vocab_size,
        model_type="bpe",
        character_coverage=1.0,
        pad_id=0,  # ID para el token de padding
        unk_id=1,  # ID para token desconocido
        bos_id=-1,  # No se usará token BOS
        eos_id=-1   # No se usará token EOS
    )

if __name__ == "__main__":
    # Ruta al corpus y prefijo de modelo
    corpus_path = "data/corpus.txt"
    model_prefix = "data/bpe_model"
    vocab_size = 100

    entrenar_bpe(corpus_path, model_prefix, vocab_size)
