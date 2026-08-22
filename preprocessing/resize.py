from collections.abc import Iterable

import cv2
import numpy as np

TARGET_SIZE = (96, 96)
DEFAULT_PADDING_VALUE = 0

# Estrategia para imagens que nao sao quadradas:
#
#   "crop"      -> recorta o centro no maior quadrado possivel e depois
#                  redimensiona. A imagem final fica 100% preenchida com
#                  conteudo, sem barras. Perde-se as laterais (ou o topo
#                  e a base) da imagem original.
#
#   "letterbox" -> preserva a imagem inteira e completa o que falta com
#                  DEFAULT_PADDING_VALUE. Nada eh descartado, mas parte
#                  da area final vira barra solida.
#
# Por que "crop" eh o padrao: uma imagem 300x200 em letterbox resulta em
# apenas 96x64 de conteudo, com 32 linhas pretas (33% da imagem). Isso
# reduz a resolucao vertical do gesto e, pior, introduz barras que NAO
# existem no quadro capturado pelo ESP32-CAM, criando um descasamento
# entre treino e inferencia. O recorte central resolve os dois problemas
# quando o objeto de interesse esta centralizado, o que eh o caso dos
# datasets utilizados.
DEFAULT_STRATEGY = "crop"


def select_interpolation(
    original_size: tuple[int, int],
    resized_size: tuple[int, int],
) -> int:
    original_width, original_height = original_size
    resized_width, resized_height = resized_size

    if resized_width < original_width or resized_height < original_height:
        return cv2.INTER_AREA

    return cv2.INTER_CUBIC


def padding_value(image: np.ndarray) -> int | tuple[int, ...]:
    if image.ndim == 2:
        return DEFAULT_PADDING_VALUE

    channels = image.shape[2]
    if channels == 1:
        return DEFAULT_PADDING_VALUE

    return tuple([DEFAULT_PADDING_VALUE] * channels)


def center_crop_square(image: np.ndarray) -> np.ndarray:
    """Recorta o maior quadrado centralizado possivel.

    Uma imagem 300x200 vira 200x200, descartando 50 pixels de cada lado.
    Imagens ja quadradas passam sem alteracao.
    """
    height, width = image.shape[:2]
    side = min(width, height)

    offset_x = (width - side) // 2
    offset_y = (height - side) // 2

    return image[offset_y:offset_y + side, offset_x:offset_x + side]


def apply_resize(image: np.ndarray, strategy: str = DEFAULT_STRATEGY) -> np.ndarray:
    if strategy not in ("crop", "letterbox"):
        raise ValueError(
            f"strategy deve ser 'crop' ou 'letterbox', recebido: {strategy!r}"
        )

    if strategy == "crop":
        image = center_crop_square(image)

    target_width, target_height = TARGET_SIZE
    original_height, original_width = image.shape[:2]

    scale = min(target_width / original_width, target_height / original_height)
    resized_width = max(1, int(round(original_width * scale)))
    resized_height = max(1, int(round(original_height * scale)))

    interpolation = select_interpolation(
        (original_width, original_height),
        (resized_width, resized_height),
    )

    resized = cv2.resize(
        image,
        (resized_width, resized_height),
        interpolation=interpolation,
    )

    # Apos o recorte central a imagem eh quadrada e o resultado ja tem
    # exatamente TARGET_SIZE, de modo que nenhum padding eh inserido.
    # O bloco abaixo continua necessario para a estrategia "letterbox".
    if resized.ndim == 2:
        canvas_shape = (target_height, target_width)
    else:
        canvas_shape = (target_height, target_width, resized.shape[2])

    canvas = np.full(
        canvas_shape,
        padding_value(image),
        dtype=image.dtype,
    )

    offset_x = (target_width - resized_width) // 2
    offset_y = (target_height - resized_height) // 2
    canvas[
        offset_y:offset_y + resized_height,
        offset_x:offset_x + resized_width,
    ] = resized

    return canvas
