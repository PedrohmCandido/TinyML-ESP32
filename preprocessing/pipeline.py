"""
pipeline.py — Orquestração do pipeline de pré-processamento
Responsáveis: Kaiki Andrade Silva e Pedro Henrique Mendes Cândido

Descrição:
    Integra as etapas de redimensionamento e conversão para tons de
    cinza, processando em lote todas as imagens do dataset original
    e salvando os resultados em disco.

    Ordem obrigatória do pipeline:
        1. apply_resize   → reduz para 96x96 px  (Kaiki)
        2. apply_grayscale → converte para 1 canal (Pedro)

    A ordem importa: realizar o grayscale antes do resize não altera
    o resultado final, mas o resize primeiro é mais eficiente pois
    reduz o volume de pixels antes da conversão de cor.

Uso via terminal:
    python preprocessing/pipeline.py \
        --input data/raw/ \
        --output data/processed/

Estrutura esperada de --input:
    data/raw/
    ├── rock/
    │   ├── imagem_001.jpg
    │   └── ...
    ├── paper/
    │   └── ...
    └── scissors/
        └── ...
"""

import os
import argparse
import cv2
import numpy as np
from tqdm import tqdm

from preprocessing.resize import apply_resize
from preprocessing.grayscale import apply_grayscale

# Extensões de imagem aceitas pelo pipeline
EXTENSOES_VALIDAS = ('.jpg', '.jpeg', '.png', '.bmp')


def processar_imagem(caminho_imagem: str) -> np.ndarray:
    """
    Executa o pipeline completo em uma única imagem.

    Etapas:
        1. Leitura da imagem em BGR (padrão OpenCV)
        2. Redimensionamento para 96x96 px
        3. Conversão para escala de cinza (1 canal)

    Parâmetros:
        caminho_imagem (str): Caminho absoluto ou relativo para o arquivo.

    Retorna:
        np.ndarray: Imagem processada com shape (96, 96, 1), dtype uint8.

    Exceções:
        FileNotFoundError: Se o arquivo não existir no caminho informado.
        ValueError: Se o arquivo não puder ser lido como imagem.
    """
    if not os.path.exists(caminho_imagem):
        raise FileNotFoundError(f"Arquivo não encontrado: {caminho_imagem}")

    imagem = cv2.imread(caminho_imagem)

    if imagem is None:
        raise ValueError(f"Não foi possível ler a imagem: {caminho_imagem}")

    # Etapa 1 — Resize (Kaiki)
    imagem = apply_resize(imagem)

    # Etapa 2 — Grayscale (Pedro)
    imagem = apply_grayscale(imagem)

    return imagem


def executar_pipeline(pasta_entrada: str, pasta_saida: str) -> None:
    """
    Processa em lote todas as imagens do dataset.

    Percorre recursivamente a pasta de entrada, mantendo a estrutura
    de subpastas (classes) na pasta de saída.

    Parâmetros:
        pasta_entrada (str): Caminho para a pasta com o dataset original.
        pasta_saida  (str): Caminho para salvar as imagens processadas.
    """
    total_processadas = 0
    total_erros = 0

    # Coleta todos os arquivos válidos antes de processar
    arquivos = []
    for raiz, _, arquivos_dir in os.walk(pasta_entrada):
        for arquivo in arquivos_dir:
            if arquivo.lower().endswith(EXTENSOES_VALIDAS):
                arquivos.append(os.path.join(raiz, arquivo))

    print(f"\n{len(arquivos)} imagens encontradas em '{pasta_entrada}'.")
    print("Iniciando pipeline: resize (96x96) → grayscale (1 canal)\n")

    for caminho_entrada in tqdm(arquivos, desc="Processando"):
        try:
            # Preserva a estrutura de subpastas (ex: rock/, paper/, scissors/)
            caminho_relativo = os.path.relpath(caminho_entrada, pasta_entrada)
            caminho_saida = os.path.join(pasta_saida, caminho_relativo)

            os.makedirs(os.path.dirname(caminho_saida), exist_ok=True)

            imagem_processada = processar_imagem(caminho_entrada)

            # Salva removendo a dimensão extra do canal para compatibilidade
            cv2.imwrite(caminho_saida, imagem_processada[:, :, 0])

            total_processadas += 1

        except (FileNotFoundError, ValueError) as e:
            print(f"\n[ERRO] {e}")
            total_erros += 1

    print(f"\nPipeline concluído.")
    print(f"  ✔ Processadas com sucesso : {total_processadas}")
    print(f"  ✘ Erros                   : {total_erros}")
    print(f"  Saída salva em            : '{pasta_saida}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Pipeline de pré-processamento de imagens para TinyML."
    )
    parser.add_argument(
        "--input",
        type=str,
        required=True,
        help="Caminho para a pasta com o dataset original (data/raw/)."
    )
    parser.add_argument(
        "--output",
        type=str,
        required=True,
        help="Caminho para salvar as imagens processadas (data/processed/)."
    )

    args = parser.parse_args()
    executar_pipeline(args.input, args.output)
