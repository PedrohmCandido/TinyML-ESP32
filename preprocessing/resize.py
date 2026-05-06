"""
resize.py — Redimensionamento de imagens para 96x96 pixels
Responsável: Kaiki Andrade Silva
Branch: branch/kaiki-resize

Descrição:
    Recebe uma imagem no formato NumPy array (BGR ou RGB) em qualquer
    resolução e retorna a imagem redimensionada para 96x96 pixels,
    mantendo a compatibilidade com o pipeline de pré-processamento
    para TinyML no ESP32-CAM.
"""

import cv2
import numpy as np

# Resolução alvo definida pelo projeto
TARGET_SIZE = (96, 96)


def apply_resize(image: np.ndarray) -> np.ndarray:
    """
    Redimensiona uma imagem para 96x96 pixels.

    Utiliza interpolação bilinear (cv2.INTER_LINEAR), que oferece
    boa qualidade para redução de resolução com custo computacional
    baixo — adequado ao volume de imagens do dataset.

    Parâmetros:
        image (np.ndarray): Imagem de entrada em qualquer resolução.
                            Pode ser RGB ou BGR, com 1 ou 3 canais.

    Retorna:
        np.ndarray: Imagem redimensionada para (96, 96, C),
                    onde C é o número de canais da imagem original.

    Exceções:
        ValueError: Se a entrada não for um np.ndarray válido.
        ValueError: Se a imagem estiver vazia ou corrompida.

    Exemplo de uso:
        import cv2
        from preprocessing.resize import apply_resize

        img = cv2.imread('data/raw/rock/imagem_001.jpg')
        img_resized = apply_resize(img)
        print(img_resized.shape)  # (96, 96, 3)
    """
    if not isinstance(image, np.ndarray):
        raise ValueError(
            f"Esperado np.ndarray, recebido {type(image).__name__}."
        )

    if image.size == 0:
        raise ValueError("A imagem recebida está vazia ou corrompida.")

    resized = cv2.resize(image, TARGET_SIZE, interpolation=cv2.INTER_LINEAR)

    return resized
