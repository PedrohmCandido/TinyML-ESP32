"""
avaliar_modelo.py

Diagnostico do estado atual do modelo (baseline, antes de qualquer
alteracao no treino). Mede:

  1. Acuracia do modelo Keras (.h5) no conjunto de teste
  2. Acuracia do modelo quantizado (.tflite) no MESMO conjunto
  3. Matriz de confusao de ambos
  4. Precisao / recall / f1 por classe
  5. Distribuicao das confiancas (quantas predicoes passam do limiar de 70%)

Os resultados sao impressos no terminal e salvos em
'resultados/diagnostico_baseline.txt'. A matriz de confusao tambem eh
salva como imagem, se o matplotlib estiver disponivel.

Uso:
    pip install scikit-learn matplotlib
    python avaliar_modelo.py

Execute a partir da RAIZ do projeto (a mesma pasta de onde voce roda o
train.py), para que os caminhos relativos funcionem.
"""

import os
import sys

import numpy as np
import tensorflow as tf
from tensorflow import keras

# ---------------------------------------------------------------------
# Caminhos (ajuste aqui se a sua estrutura for diferente)
# ---------------------------------------------------------------------
PASTA_TESTE = "data/processed/rps-real/test"
CAMINHO_H5 = "models/final/modelo_gestos.h5"
CAMINHO_TFLITE = "models/final/modelo_gestos.tflite"
PASTA_RESULTADOS = "resultados"
LIMIAR_FIRMWARE = 0.70  # mesmo valor usado no inferencia_gestos.ino


def verificar_caminhos() -> bool:
    """Confere os caminhos. Retorna True se o .tflite estiver disponivel."""
    faltando = [c for c in (PASTA_TESTE, CAMINHO_H5)
                if not os.path.exists(c)]
    if faltando:
        print("[ERRO] Nao encontrei estes caminhos:")
        for c in faltando:
            print(f"       - {c}")
        print("\n       Rode o script a partir da raiz do projeto (a mesma")
        print("       pasta de onde voce executa o train.py), ou ajuste as")
        print("       constantes no topo deste arquivo.")
        sys.exit(1)

    tem_tflite = os.path.exists(CAMINHO_TFLITE)
    if not tem_tflite:
        print(f"[AVISO] {CAMINHO_TFLITE} nao encontrado.")
        print("        Avaliando apenas o modelo Keras (.h5).")
        print("        Rode 'python convert.py' para incluir o TFLite.\n")
    return tem_tflite


def carregar_teste():
    """Carrega o conjunto de teste SEM embaralhar.

    O shuffle=False eh obrigatorio: sem ele, os rotulos lidos numa
    passagem nao corresponderiam as predicoes feitas em outra.
    """
    ds = keras.utils.image_dataset_from_directory(
        PASTA_TESTE,
        image_size=(96, 96),
        color_mode="grayscale",
        shuffle=False,
    )
    classes = ds.class_names
    # Mesma normalizacao usada no treino.
    ds = ds.map(lambda x, y: (keras.layers.Rescaling(1. / 255)(x), y))
    return ds, classes


def prever_keras(ds):
    modelo = keras.models.load_model(CAMINHO_H5)
    probs = modelo.predict(ds, verbose=0)
    return probs, modelo


def prever_tflite(ds):
    """Roda o interpretador TFLite imagem por imagem.

    Este eh o modelo que realmente esta no ESP32, entao esta medicao eh
    a que representa o desempenho embarcado.
    """
    it = tf.lite.Interpreter(model_path=CAMINHO_TFLITE)
    it.allocate_tensors()
    det_ent = it.get_input_details()[0]
    det_sai = it.get_output_details()[0]

    idx_ent = det_ent["index"]
    idx_sai = det_sai["index"]
    tipo_ent = det_ent["dtype"]

    saidas = []
    for lote, _ in ds:
        for img in lote.numpy():
            entrada = img[np.newaxis, ...].astype(tipo_ent)
            it.set_tensor(idx_ent, entrada)
            it.invoke()
            saidas.append(it.get_tensor(idx_sai)[0].copy())

    return np.array(saidas), det_ent, det_sai


def matriz_texto(matriz, classes) -> str:
    largura = max(10, max(len(c) for c in classes) + 2)
    linhas = []
    cabecalho = " " * largura + "".join(f"{c[:8]:>10}" for c in classes)
    linhas.append(cabecalho + "   <- predito")
    for i, c in enumerate(classes):
        linha = f"{c:<{largura}}" + "".join(f"{v:>10}" for v in matriz[i])
        total = matriz[i].sum()
        acerto = matriz[i][i] / total * 100 if total else 0.0
        linhas.append(linha + f"    ({acerto:.1f}% correto)")
    return "\n".join(linhas)


def salvar_figura(matriz, classes, nome_modelo, caminho):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return None

    fig, ax = plt.subplots(figsize=(5.5, 4.8))
    im = ax.imshow(matriz, cmap="Blues")
    ax.set_xticks(range(len(classes)), classes, rotation=45, ha="right")
    ax.set_yticks(range(len(classes)), classes)
    ax.set_xlabel("Classe predita")
    ax.set_ylabel("Classe verdadeira")
    ax.set_title(f"Matriz de confusao — {nome_modelo}")

    limite = matriz.max() / 2.0
    for i in range(len(classes)):
        for j in range(len(classes)):
            ax.text(j, i, str(matriz[i, j]), ha="center", va="center",
                    color="white" if matriz[i, j] > limite else "black")

    fig.colorbar(im, ax=ax)
    fig.tight_layout()
    fig.savefig(caminho, dpi=160)
    plt.close(fig)
    return caminho


def main() -> None:
    tem_tflite = verificar_caminhos()
    os.makedirs(PASTA_RESULTADOS, exist_ok=True)

    try:
        from sklearn.metrics import confusion_matrix, classification_report
    except ImportError:
        print("[ERRO] scikit-learn nao esta instalado.")
        print("       Rode: pip install scikit-learn")
        sys.exit(1)

    relatorio = []

    def registrar(texto=""):
        print(texto)
        relatorio.append(texto)

    ds, classes = carregar_teste()
    y_true = np.concatenate([y.numpy() for _, y in ds])

    registrar("=" * 66)
    registrar("DIAGNOSTICO BASELINE — modelo atual, sem alteracoes")
    registrar("=" * 66)
    registrar(f"\nClasses (ordem alfabetica do Keras): {classes}")
    registrar(f"Imagens no conjunto de teste: {len(y_true)}")
    for i, c in enumerate(classes):
        registrar(f"   {i} = {c:<20} {(y_true == i).sum()} imagens")

    # ---------------- Keras (.h5) ----------------
    registrar("\n" + "-" * 66)
    registrar("MODELO KERAS (.h5) — o que foi treinado no PC")
    registrar("-" * 66)

    probs_k, modelo = prever_keras(ds)
    y_k = np.argmax(probs_k, axis=1)
    acc_k = (y_k == y_true).mean()

    registrar(f"\nParametros totais: {modelo.count_params():,}")
    registrar(f"Acuracia: {acc_k:.4f}  ({acc_k * 100:.2f}%)")
    mat_k = confusion_matrix(y_true, y_k)
    registrar("\nMatriz de confusao:")
    registrar(matriz_texto(mat_k, classes))

    if not tem_tflite:
        registrar("\n" + "-" * 66)
        registrar("MODELO TFLITE — nao avaliado")
        registrar("-" * 66)
        registrar("\nArquivo .tflite ausente. Rode 'python convert.py'")
        registrar("e execute este script de novo para comparar os dois")
        registrar("modelos e medir o efeito da quantizacao.")
        mat_t = None
    else:
        # ---------------- TFLite ----------------
        registrar("\n" + "-" * 66)
        registrar("MODELO TFLITE (.tflite) — o que roda no ESP32")
        registrar("-" * 66)

        probs_t, det_ent, det_sai = prever_tflite(ds)
        y_t = np.argmax(probs_t, axis=1)
        acc_t = (y_t == y_true).mean()

        registrar(f"\nEntrada: shape={det_ent['shape']} dtype={det_ent['dtype'].__name__}")
        registrar(f"Saida  : shape={det_sai['shape']} dtype={det_sai['dtype'].__name__}")
        registrar(f"Tamanho do arquivo: {os.path.getsize(CAMINHO_TFLITE):,} bytes")
        registrar(f"\nAcuracia: {acc_t:.4f}  ({acc_t * 100:.2f}%)")
        mat_t = confusion_matrix(y_true, y_t)
        registrar("\nMatriz de confusao:")
        registrar(matriz_texto(mat_t, classes))
        registrar("\nRelatorio por classe:")
        registrar(classification_report(y_true, y_t, target_names=classes, digits=4))

        # ---------------- Impacto da quantizacao ----------------
        registrar("-" * 66)
        registrar("IMPACTO DA QUANTIZACAO")
        registrar("-" * 66)

        # Se o .tflite for mais antigo que o .h5, ele veio de OUTRO treino e a
        # comparacao entre os dois nao mede quantizacao nenhuma.
        mtime_h5 = os.path.getmtime(CAMINHO_H5)
        mtime_tfl = os.path.getmtime(CAMINHO_TFLITE)
        desatualizado = mtime_tfl < mtime_h5

        if desatualizado:
            from datetime import datetime
            fmt = "%d/%m/%Y %H:%M"
            registrar("\n!! ATENCAO: o .tflite eh MAIS ANTIGO que o .h5.")
            registrar(f"   .h5     modificado em {datetime.fromtimestamp(mtime_h5):{fmt}}")
            registrar(f"   .tflite modificado em {datetime.fromtimestamp(mtime_tfl):{fmt}}")
            registrar("   Os dois arquivos vieram de treinos DIFERENTES, entao a")
            registrar("   comparacao abaixo NAO mede o efeito da quantizacao.")
            registrar("   Rode 'python convert.py' e execute este script de novo.")

        delta = (acc_t - acc_k) * 100
        registrar(f"\n.h5     : {acc_k * 100:.2f}%")
        registrar(f".tflite : {acc_t * 100:.2f}%")
        registrar(f"Variacao: {delta:+.2f} pontos percentuais")
        discordancia = (y_k != y_t).sum()
        registrar(f"Imagens em que os dois modelos discordam: {discordancia} "
                  f"de {len(y_true)} ({discordancia / len(y_true) * 100:.1f}%)")

        if desatualizado:
            registrar("\n>> Veredito suspenso: arquivos de treinos diferentes.")
        elif delta < -3:
            registrar("\n>> A quantizacao degradou o modelo de forma relevante.")
            registrar("   Corrigir o representative_dataset do convert.py deve ajudar.")
        else:
            registrar("\n>> A quantizacao preservou bem a acuracia.")
            registrar("   O problema em campo nao vem daqui.")

        # ---------------- Confianca e limiar do firmware ----------------
        registrar("\n" + "-" * 66)
        registrar(f"CONFIANCA E LIMIAR DE {LIMIAR_FIRMWARE * 100:.0f}% (TFLite)")
        registrar("-" * 66)

        conf = probs_t.max(axis=1)
        acima = conf > LIMIAR_FIRMWARE
        registrar(f"\nConfianca media : {conf.mean() * 100:.2f}%")
        registrar(f"Confianca mediana: {np.median(conf) * 100:.2f}%")
        registrar(f"Predicoes acima do limiar: {acima.sum()} de {len(conf)} "
                  f"({acima.mean() * 100:.1f}%)")

        if acima.sum():
            acc_acima = (y_t[acima] == y_true[acima]).mean()
            registrar(f"Acuracia SO nas predicoes acima do limiar: {acc_acima * 100:.2f}%")
        erradas_confiantes = ((y_t != y_true) & acima).sum()
        registrar(f"Erros COM alta confianca (>{LIMIAR_FIRMWARE * 100:.0f}%): "
                  f"{erradas_confiantes}")
        registrar("   (este numero mostra o quanto o limiar do firmware NAO"
                  " protege contra erros confiantes)")

    # ---------------- Saidas em disco ----------------
    caminho_txt = os.path.join(PASTA_RESULTADOS, "diagnostico_baseline.txt")
    with open(caminho_txt, "w", encoding="utf-8") as f:
        f.write("\n".join(relatorio) + "\n")

    print("\n" + "=" * 66)
    print(f"Relatorio salvo em: {caminho_txt}")

    fig_k = salvar_figura(mat_k, classes, "Keras (.h5)",
                          os.path.join(PASTA_RESULTADOS, "confusao_h5.png"))
    fig_t = None
    if mat_t is not None:
        fig_t = salvar_figura(mat_t, classes, "TFLite quantizado",
                              os.path.join(PASTA_RESULTADOS, "confusao_tflite.png"))
    if fig_k:
        print(f"Figura salva em   : {fig_k}")
        if fig_t:
            print(f"Figura salva em   : {fig_t}")
    else:
        print("(matplotlib nao instalado — figuras nao geradas)")


if __name__ == "__main__":
    main()
