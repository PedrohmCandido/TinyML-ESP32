"""
grayscale.py — Conversão de imagens para tons de cinza
Responsável: Pedro Henrique Mendes Cândido
Branch: branch/pedro-grayscale

Descrição:
    Recebe uma imagem no formato NumPy array (BGR, formato padrão
    do OpenCV) e retorna a imagem convertida para escala de cinza
    com um único canal, reduzindo o consumo de memória em 66,6%
    em relação à representação RGB/BGR original.

    Esta etapa é executada APÓS o resize no pipeline, garantindo
    que a saída final seja sempre (96, 96, 1).
"""

import cv2
import numpy as np


def apply_grayscale(image: np.ndarray) -> np.ndarray:
    """
    Converte uma imagem BGR para escala de cinza (1 canal).

    A conversão utiliza a fórmula ponderada padrão do OpenCV:
        Gray = 0.114*B + 0.587*G + 0.299*R
    que respeita a sensibilidade do olho humano a cada canal,
    preservando melhor o contraste visual entre os gestos.

    O canal único é expandido de (H, W) para (H, W, 1) para
    manter compatibilidade com o formato de entrada da CNN,
    que espera tensores tridimensionais.

    Parâmetros:
        image (np.ndarray): Imagem de entrada em formato BGR
                            com shape (H, W, 3).

    Retorna:
        np.ndarray: Imagem em escala de cinza com shape (H, W, 1),
                    dtype uint8, valores no intervalo [0, 255].

    Exceções:
        ValueError: Se a entrada não for um np.ndarray válido.
        ValueError: Se a imagem não possuir 3 canais (BGR esperado).

    Exemplo de uso:
        import cv2
        from preprocessing.grayscale import apply_grayscale

        img = cv2.imread('data/raw/paper/imagem_001.jpg')
        img_gray = apply_grayscale(img)
        print(img_gray.shape)  # (300, 300, 1) antes do resize
                               # (96, 96, 1) após o resize no pipeline
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"Esperado np.ndarray, recebido {type(image).__name__}."
        )

    if len(image.shape) != 3 or image.shape[2] != 3:
        raise ValueError(
            f"Imagem com 3 canais BGR esperada, "
            f"recebida com shape {image.shape}."
        )

    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Expande de (H, W) para (H, W, 1) para compatibilidade com a CNN
    gray = np.expand_dims(gray, axis=-1)

    return gray
