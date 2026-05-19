import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm
from resize import apply_resize


EXTENSOES_VALIDAS = ('.jpg', '.jpeg', '.png', '.bmp')

def processar_imagem(caminho_imagem: str) -> np.ndarray:

    if not os.path.exists(caminho_imagem):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_imagem}")

    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        raise ValueError(f"Não foi possível ler a imagem: {caminho_imagem}")

    imagem = apply_resize(imagem)

    return imagem


def executar_pipeline(pasta_entrada: str, pasta_saida: str) -> None:
 
    total_processadas = 0
    total_erros = 0

    arquivos = []
    for raiz, _, arquivos_dir in os.walk(pasta_entrada):
        for arquivo in arquivos_dir:
            if arquivo.lower().endswith(EXTENSOES_VALIDAS):
                arquivos.append(os.path.join(raiz, arquivo))

    print(f"\n{len(arquivos)} imagens encontradas em '{pasta_entrada}'.")
    print("Iniciando pipeline\n")

    for caminho_entrada in tqdm(arquivos, desc="Processando"):
        try:
            caminho_relativo = os.path.relpath(caminho_entrada, pasta_entrada)
            caminho_saida = os.path.join(pasta_saida, caminho_relativo)

            os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

            imagem_processada = processar_imagem(caminho_entrada)

            cv2.imwrite(caminho_saida, imagem_processada[:, :, 0])

            total_processadas += 1

        except (FileNotFoundError, ValueError) as e:
            print(f"\n[ERRO] {e}")
            total_erros += 1

    print("\nPipeline concluido.")
    print(f"  OK - Processadas com sucesso : {total_processadas}")
    print(f"  ERRO - Falhas                : {total_erros}")
    print(f"  Saida salva em               : '{pasta_saida}'")


if __name__ == "__main__":
    executar_pipeline('data/raw/', 'data/processed/')
