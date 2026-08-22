"""
visualizar_camera.py

Le os frames em escala de cinza (96x96) transmitidos pelo firmware do
ESP32-CAM (inferencia_gestos.ino, com ENVIAR_IMAGEM_SERIAL = 1) e exibe
em tempo real usando OpenCV, junto com o gesto reconhecido e a confianca
reportada pelo proprio firmware.

Protocolo:
    Cada frame vem precedido pelo marcador de sincronizacao
    0xAA 0x55 0xAA 0x55, seguido por exatamente 96*96 = 9216 bytes
    (um byte por pixel, grayscale).

    Entre os frames trafegam as mensagens de texto do firmware, no
    formato "Gesto: rock | Confianca: 99.61%" ou "Nenhum gesto
    detectado". Este script aproveita esses bytes (que antes eram
    descartados) para montar a legenda exibida na janela.

Uso:
    pip install pyserial opencv-python numpy
    python visualizar_camera.py --port COM3 --baud 115200

Controles:
    q  -> encerra
    s  -> salva o frame atual em disco (pasta 'capturas/')
"""

import argparse
import os
import re
import sys
import time

import cv2
import numpy as np
import serial

MARCADOR = bytes([0xAA, 0x55, 0xAA, 0x55])
LARGURA = 96
ALTURA = 96
TAMANHO_FRAME = LARGURA * ALTURA
FATOR_ZOOM = 5          # amplia a exibicao (96px eh muito pequeno na tela)
ALTURA_LEGENDA = 70     # faixa abaixo da imagem onde a legenda eh desenhada
PASTA_CAPTURAS = "capturas"

# "Gesto: rock | Confianca: 99.61%" -> captura "rock" e "99.61".
# O .*? no meio evita depender do acento de "Confianca", que pode chegar
# como byte invalido dependendo da codificacao do firmware.
PADRAO_GESTO = re.compile(r"Gesto:\s*(\w+)\s*\|.*?([\d.]+)\s*%")
PADRAO_SEM_GESTO = re.compile(r"Nenhum gesto detectado")

# Cores em BGR (padrao do OpenCV)
COR_DETECTADO = (110, 220, 120)   # verde
COR_SEM_GESTO = (150, 150, 150)   # cinza
COR_AGUARDANDO = (60, 190, 240)   # ambar
COR_INFO = (170, 170, 170)


def abrir_serial(porta: str, baud: int, timeout: float = 2.0) -> serial.Serial:
    try:
        # Monta a porta sem abrir ainda, para poder zerar DTR/RTS
        # ANTES da abertura. Isso evita que o ESP32 seja forcado a
        # entrar em modo bootloader/gravacao (comportamento de
        # auto-reset comum em placas com chip CH340/CP2102), que faz
        # a porta abrir normalmente mas nenhum byte do firmware chegar.
        ser = serial.Serial()
        ser.port = porta
        ser.baudrate = baud
        ser.timeout = timeout
        ser.dtr = False
        ser.rts = False
        ser.open()
        time.sleep(0.3)
        ser.reset_input_buffer()
        return ser
    except serial.SerialException as e:
        print(f"[ERRO] Nao foi possivel abrir a porta {porta}: {e}")
        print("       Verifique se a porta esta correta e se o Arduino IDE "
              "(ou outro programa) nao esta com o Monitor Serial aberto.")
        sys.exit(1)


def processar_linha_texto(linha: str, estado: dict) -> None:
    """Interpreta uma linha de texto vinda do firmware e atualiza o estado."""
    linha = linha.strip()
    if not linha:
        return

    achou = PADRAO_GESTO.search(linha)
    if achou:
        estado["gesto"] = achou.group(1)
        try:
            estado["confianca"] = float(achou.group(2))
        except ValueError:
            estado["confianca"] = None
        estado["detectado"] = True
        print(f"[CLASSIFICACAO] {estado['gesto']} - {achou.group(2)}%")
        return

    if PADRAO_SEM_GESTO.search(linha):
        estado["gesto"] = None
        estado["confianca"] = None
        estado["detectado"] = False
        return

    # Qualquer outra mensagem do firmware (ex.: "Camera inicializada!")
    # eh apenas ecoada no terminal, sem virar legenda.
    print(f"[ESP32] {linha}")


def esperar_marcador(ser: serial.Serial, contadores: dict, estado: dict) -> bool:
    """Le byte a byte ate encontrar a sequencia MARCADOR no stream serial.

    Retorna False em caso de timeout (nenhum dado chegou), o que permite
    ao chamador continuar tentando sem travar o programa. Os bytes lidos
    antes do marcador sao acumulados como texto: quando uma linha
    completa se forma, ela eh interpretada para alimentar a legenda.
    """
    buffer = bytearray()
    while True:
        byte = ser.read(1)
        if not byte:
            return False

        contadores["total_bytes"] += 1
        contadores["preview"] += byte
        if len(contadores["preview"]) > 200:
            contadores["preview"] = contadores["preview"][-200:]

        # Acumula texto ate encontrar uma quebra de linha.
        if byte == b"\n":
            linha = estado["linha"].decode("utf-8", errors="replace")
            estado["linha"] = bytearray()
            processar_linha_texto(linha, estado)
        elif byte != b"\r":
            estado["linha"] += byte
            # Trava de seguranca: se muitos bytes vierem sem quebra de
            # linha, provavelmente sao dados binarios e nao texto.
            if len(estado["linha"]) > 300:
                estado["linha"] = bytearray()

        buffer += byte
        if len(buffer) > len(MARCADOR):
            buffer.pop(0)
        if bytes(buffer) == MARCADOR:
            # Descarta texto parcial acumulado imediatamente antes do frame.
            estado["linha"] = bytearray()
            return True


def ler_frame(ser: serial.Serial) -> np.ndarray | None:
    dados = ser.read(TAMANHO_FRAME)
    if len(dados) != TAMANHO_FRAME:
        return None
    return np.frombuffer(dados, dtype=np.uint8).reshape((ALTURA, LARGURA))


def montar_exibicao(frame: np.ndarray, contador: int, estado: dict) -> np.ndarray:
    """Amplia o frame e desenha a faixa de legenda embaixo."""
    largura = LARGURA * FATOR_ZOOM
    altura = ALTURA * FATOR_ZOOM

    ampliado = cv2.resize(
        frame, (largura, altura), interpolation=cv2.INTER_NEAREST
    )
    # Converte para BGR para permitir texto colorido sobre a imagem cinza.
    ampliado = cv2.cvtColor(ampliado, cv2.COLOR_GRAY2BGR)

    tela = np.zeros((altura + ALTURA_LEGENDA, largura, 3), dtype=np.uint8)
    tela[:altura] = ampliado
    tela[altura:] = (28, 28, 28)

    # Contador de frames, no canto superior da imagem.
    cv2.putText(
        tela, f"frame {contador}", (8, 20),
        cv2.FONT_HERSHEY_SIMPLEX, 0.45, COR_INFO, 1, cv2.LINE_AA,
    )

    # Faixa de legenda.
    if estado["detectado"] is None:
        texto = "aguardando classificacao..."
        cor = COR_AGUARDANDO
    elif estado["detectado"]:
        confianca = estado["confianca"]
        sufixo = f"   {confianca:.2f}%" if confianca is not None else ""
        texto = f"{estado['gesto'].upper()}{sufixo}"
        cor = COR_DETECTADO
    else:
        texto = "nenhum gesto detectado"
        cor = COR_SEM_GESTO

    cv2.putText(
        tela, texto, (12, altura + 30),
        cv2.FONT_HERSHEY_SIMPLEX, 0.7, cor, 2, cv2.LINE_AA,
    )

    # Barra de confianca proporcional, so quando ha deteccao.
    if estado["detectado"] and estado["confianca"] is not None:
        largura_util = largura - 24
        preenchido = int(largura_util * min(estado["confianca"], 100.0) / 100.0)
        y = altura + 46
        cv2.rectangle(tela, (12, y), (12 + largura_util, y + 10), (60, 60, 60), -1)
        cv2.rectangle(tela, (12, y), (12 + preenchido, y + 10), cor, -1)
        # Marca visual do limiar de 70% usado pelo firmware.
        x_limiar = 12 + int(largura_util * 0.70)
        cv2.line(tela, (x_limiar, y - 3), (x_limiar, y + 13), (200, 200, 200), 1)

    return tela


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Visualiza em tempo real as imagens capturadas pelo ESP32-CAM."
    )
    parser.add_argument("--port", default="COM3", help="Porta serial do ESP32 (padrao: COM3)")
    parser.add_argument(
        "--baud", type=int, default=115200,
        help="Baud rate (deve ser IGUAL ao Serial.begin() do firmware)"
    )
    args = parser.parse_args()

    ser = abrir_serial(args.port, args.baud)
    print(f"Conectado a {args.port} @ {args.baud} baud.")
    print("Aguardando frames... (janela de video vai abrir assim que o primeiro frame chegar)")
    print("Controles: 'q' sai | 's' salva o frame atual")

    contador_frames = 0
    ultimo_frame = None
    contadores = {"total_bytes": 0, "preview": bytearray()}
    # detectado: None = ainda nao chegou classificacao nenhuma
    #            True = gesto acima do limiar de 70%
    #            False = firmware reportou "Nenhum gesto detectado"
    estado = {"gesto": None, "confianca": None, "detectado": None,
              "linha": bytearray()}
    ultimo_aviso = time.time()

    try:
        while True:
            if not esperar_marcador(ser, contadores, estado):
                # timeout esperando o marcador (2s sem novos bytes).
                # A cada ~4s sem nenhum frame, mostra um diagnostico.
                if time.time() - ultimo_aviso > 4.0:
                    ultimo_aviso = time.time()
                    if contadores["total_bytes"] == 0:
                        print("[DEBUG] Nenhum byte chegou na porta ainda. "
                              "Confirme se o ESP32 esta ligado, rodando o "
                              "firmware (nao em modo de gravacao) e se "
                              "nenhum outro programa esta usando a porta.")
                    else:
                        texto = contadores["preview"].decode("ascii", errors="replace")
                        print(f"[DEBUG] {contadores['total_bytes']} bytes recebidos, "
                              f"mas nenhum frame valido ainda. Ultimo trecho recebido:")
                        print(f"        {texto!r}")
                continue

            frame = ler_frame(ser)
            if frame is None:
                print("[AVISO] Frame incompleto/corrompido, descartando.")
                continue

            contador_frames += 1
            ultimo_frame = frame

            tela = montar_exibicao(frame, contador_frames, estado)
            cv2.imshow("ESP32-CAM - Captura ao vivo (96x96)", tela)

            tecla = cv2.waitKey(1) & 0xFF
            if tecla == ord("q"):
                break
            elif tecla == ord("s") and ultimo_frame is not None:
                os.makedirs(PASTA_CAPTURAS, exist_ok=True)
                nome = os.path.join(PASTA_CAPTURAS, f"frame_{int(time.time())}.png")
                cv2.imwrite(nome, ultimo_frame)
                print(f"[OK] Frame salvo em {nome}")

    except KeyboardInterrupt:
        pass
    finally:
        ser.close()
        cv2.destroyAllWindows()
        print(f"\nEncerrado. Total de frames recebidos: {contador_frames}")


if __name__ == "__main__":
    main()